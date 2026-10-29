require "json"
require "http/client"
require "sync"
require "sync/exclusive"

class Agent
  # Raised when trying to use an Agent that has been closed.
  class ClosedError < Error
    def initialize
      super("Agent has been closed")
    end
  end

  # Internal request types sent to the processing fiber.
  private record AskRequest, new_messages : Array(Message), tools : Array(Tool)?, response : Response

  private alias Request = AskRequest

  getter config : Config
  getter session_id : String
  getter cache_key : String

  # Return a snapshot of the conversation history.
  # Thread-safe; acquires @history internally.
  def history : Array(Message)
    @history.lock(&.dup)
  end

  @request_channel : Channel(Request)
  @fiber : Fiber
  @state_mutex : Sync::Mutex # protects @closed
  @closed = false

  # Protected shared state via Sync::Exclusive.
  # Each value is bundled with its own Mutex — the lock relationship is
  # encoded in the type.
  @history : Sync::Exclusive(Array(Message))
  @registered_tools : Sync::Exclusive(Hash(String, NamedTuple(tool: Tool, callback: (Hash(String, JSON::Any) -> String)?, enabled: Bool)))

  # Persistent HTTP client for connection pooling across requests.
  @http_client : HTTP::Client
  # The per-read timeout value that was last applied to @http_client's socket at
  # open time. Used by #apply_client_read_timeout to decide whether the client
  # must be rebuilt so a new timeout value actually takes effect on the socket.
  @applied_read_timeout : Time::Span?
  # The provider handles all wire-format concerns.
  @provider : Provider::Base
  # Chain of handler decorators.
  @handlers : ChainHandler

  def initialize(@config : Config, provider : Provider::Base? = nil)
    @session_id = Random::Secure.hex(16)
    @cache_key = @config.prompt_cache_key || "agent-cr:#{@session_id}"
    @history = Sync::Exclusive(Array(Message)).new([] of Message)
    @request_channel = Channel(Request).new
    @registered_tools = Sync::Exclusive(Hash(String, NamedTuple(tool: Tool, callback: (Hash(String, JSON::Any) -> String)?, enabled: Bool))).new(
      {} of String => NamedTuple(tool: Tool, callback: (Hash(String, JSON::Any) -> String)?, enabled: Bool)
    )
    @state_mutex = Sync::Mutex.new
    @provider = provider || Provider::OpenAI.new(@config, @cache_key)
    @applied_read_timeout = nil
    @http_client = build_http_client
    @handlers = ChainHandler.new
    @fiber = spawn { run_loop }
  end

  # Append a `Handler` to the chain. Handlers run left-to-right in the order
  # they were added, wrapping each stage of the request pipeline (turn,
  # chunk, tool call, error). See `Agent::Handler`.
  def use(handler : Handler) : self
    @handlers << handler
    self
  end

  # Register a tool with a callback that will be called automatically when the
  # model requests this tool (when `auto_execute_tools` is true in config).
  #
  # The callback receives the parsed JSON arguments and returns a string result.
  # The tool definition is automatically included in all subsequent #ask calls.
  #
  # Safe to call from any fiber, including the agent fiber itself.
  #
  # ```
  # agent.register_tool("get_weather", "Get the weather for a city",
  #   parameters: {
  #     "type"       => JSON::Any.new("object"),
  #     "properties" => JSON::Any.new({
  #       "city" => JSON::Any.new({"type" => JSON::Any.new("string")}),
  #     }),
  #     "required" => JSON::Any.new([] of JSON::Any),
  #   }
  # ) do |args|
  #   city = args["city"]?.try(&.as_s) || "unknown"
  #   "The weather in #{city} is sunny."
  # end
  # ```
  def register_tool(name : String, description : String? = nil, parameters : Hash(String, JSON::Any)? = nil, enabled : Bool = true, &block : Hash(String, JSON::Any) -> String) : Nil
    @state_mutex.synchronize { raise ClosedError.new if @closed }
    raise ArgumentError.new("Tool name must not be empty") if name.empty?

    tool = Tool.new(Tool::FunctionDef.new(name: name, description: description, parameters: parameters))
    entry = {tool: tool, callback: block, enabled: enabled}

    @registered_tools.lock { |t| t[name] = entry }
  end

  # Enable a previously registered tool, making it available to the model.
  # Returns `false` if no tool with that name is registered.
  # Safe to call from any fiber.
  def enable_tool(name : String) : Bool
    @state_mutex.synchronize { raise ClosedError.new if @closed }

    @registered_tools.lock do |t|
      return false unless t.has_key?(name)
      t[name] = t[name].merge({enabled: true})
      true
    end
  end

  # Disable a registered tool, hiding it from the model without unregistering it.
  # Returns `false` if no tool with that name is registered.
  # Safe to call from any fiber.
  def disable_tool(name : String) : Bool
    @state_mutex.synchronize { raise ClosedError.new if @closed }

    @registered_tools.lock do |t|
      return false unless t.has_key?(name)
      t[name] = t[name].merge({enabled: false})
      true
    end
  end

  # Returns the names of all currently enabled tools.
  # Thread-safe; acquires @registered_tools internally.
  def enabled_tools : Array(String)
    @registered_tools.lock(&.select { |_, v| v[:enabled] }.keys)
  end

  # Close the agent, shutting down the background fiber.
  # Any pending or future #ask calls will get a closed-error response.
  # Safe to call multiple times.
  #
  # NOTE: Do not call #close from inside a tool callback — if you need
  # to stop the agent mid-callback, prefer response.cancel on the
  # current Response. Calling #close from a tool callback will close
  # the HTTP client underneath the in-flight request.
  def close : Nil
    @state_mutex.synchronize { return if @closed; @closed = true }
    @request_channel.close
    @http_client.close
    @provider.close
  end

  # Send a user message and return a Response immediately.
  # The actual HTTP call happens in a background fiber.
  #
  # If `auto_execute_tools` is true (default) and the model requests registered
  # tools, the agent will automatically execute them and re-ask the model in
  # the background. The final Response (after all tool loops) is returned.
  #
  # Pass `attachments` as an array of file paths, URLs, or data URIs.
  # Local files are read and auto-typed by MIME (image → image_url,
  # text → inline text, audio → input_audio, other → file).
  #
  # ```
  # resp = agent.ask("What is in this image?", attachments: ["photo.jpg"])
  # resp.stream { |chunk| print chunk }
  # puts resp.message.content
  # ```
  #
  # Raises Agent::ClosedError if the agent has been closed via #close.
  def ask(content : String, attachments : Array(String)? = nil, tools : Array(Tool)? = nil) : Response
    @state_mutex.synchronize { raise ClosedError.new if @closed }

    msg = if atts = attachments
            parts = [ContentPart.new(type: :text, text: content)] + atts.map { |path| ContentPart.from_path(path) }
            Message.new(role: Role::User, content_parts: parts)
          else
            Message.new(role: Role::User, content: content)
          end

    response = Response.new
    @request_channel.send(AskRequest.new([msg], tools, response))
    response
  rescue Channel::ClosedError
    raise ClosedError.new
  end

  # Send tool result messages and return a Response immediately.
  # Used to feed tool call results back to the model after it requests them.
  #
  # Raises Agent::ClosedError if the agent has been closed via #close.
  def ask(tool_results : Array(Message), tools : Array(Tool)? = nil) : Response
    @state_mutex.synchronize { raise ClosedError.new if @closed }

    response = Response.new
    @request_channel.send(AskRequest.new(tool_results, tools, response))
    response
  rescue Channel::ClosedError
    raise ClosedError.new
  end

  # Restore session state from previously dumped session data.
  #
  # This restores session_id, cache_key, conversation history, and enabled-tool
  # state — everything except the system prompt and tool definitions (those
  # come from Config and from register_tool calls). Call this after creating
  # a fresh Agent and registering all tools.
  #
  # Accepts a raw JSON string (shorthand) or a pre-parsed `JSON::Any` hash
  # (useful when the agent data lives inside a larger document).
  #
  # Raises `Agent::SessionLoadError` if the session data is corrupt or
  # malformed.
  #
  # NOTE: Calling #load while an #ask is in-flight is undefined behavior.
  # Wait for any in-flight Response to complete (via #join) first.
  #
  # ```
  # agent = Agent.new(config)
  # agent.register_tool(...)
  # agent.load(File.read("session.json"))
  # agent.ask("Continue where we left off")
  # ```
  def load(data : String | JSON::Any) : Nil
    parsed = parse_session_data(data)

    id = parse_str_field(parsed, "session_id")
    key = parse_str_field(parsed, "cache_key")

    history_raw = parsed["history"]?
    raise SessionLoadError.new("'history' field missing") if history_raw.nil?
    history = begin
      Array(Message).from_json(history_raw.to_json)
    rescue ex
      raise SessionLoadError.new("'history' field is malformed", cause: ex)
    end

    enabled_names = parse_enabled_tools(parsed)

    @session_id = id
    @cache_key = key

    load_history(history)

    @registered_tools.lock do |tools|
      enabled_names.each do |name|
        if tools.has_key?(name)
          tools[name] = tools[name].merge({enabled: true})
        else
          Log.warn { "Agent#load: tool '#{name}' was enabled in saved session but is not registered — skipping" }
        end
      end
    end
  end

  private def parse_session_data(data : String | JSON::Any) : Hash(String, JSON::Any)
    h = begin
      data.is_a?(String) ? JSON.parse(data) : data
    rescue ex
      raise SessionLoadError.new("not a valid JSON object", cause: ex)
    end
    parsed = h.as_h?
    raise SessionLoadError.new("not a valid JSON object") if parsed.nil?
    parsed
  end

  private def parse_str_field(parsed : Hash(String, JSON::Any), field : String) : String
    val = begin
      parsed[field]?.try(&.as_s)
    rescue ex
      raise SessionLoadError.new("'#{field}' field missing or not a string", cause: ex)
    end
    raise SessionLoadError.new("'#{field}' field missing or not a string") if val.nil?
    val
  end

  private def parse_enabled_tools(parsed : Hash(String, JSON::Any)) : Array(String)
    if names = parsed["enabled_tools"]?
      names.as_a.map(&.as_s)
    else
      [] of String
    end
  rescue ex
    raise SessionLoadError.new("'enabled_tools' field is not an array of strings", cause: ex)
  end

  # Reset the conversation history back to the system prompt only.
  #
  # NOTE: Calling #reset while an #ask is in-flight is undefined behavior.
  # Wait for any in-flight Response to complete (via #join) first.
  #
  # Raises Agent::ClosedError if the agent has been closed.
  def reset : Nil
    @state_mutex.synchronize { raise ClosedError.new if @closed }
    @history.lock(&.clear)
  end

  # Restore the full conversation history.
  # This replaces the current history with the given messages (e.g. from a
  # previous `#dump`). The system prompt is NOT included — `build_messages`
  # prepends it from Config as usual.
  #
  # NOTE: Calling #load_history while an #ask is in-flight is undefined behavior.
  # Wait for any in-flight Response to complete (via #join) first.
  #
  # Raises Agent::ClosedError if the agent has been closed.
  def load_history(messages : Array(Message)) : Nil
    @state_mutex.synchronize { raise ClosedError.new if @closed }
    @history.lock { |h| h.clear; h.concat(messages) }
  end

  # Serialise the current session fields into an open JSON object.
  # Call this inside a `json.object` block managed by the caller.
  #
  # Thread-safe; acquires @history and @registered_tools internally.
  #
  # ```
  # JSON.build do |json|
  #   json.object do
  #     json.field "hero", hero
  #     agent.dump(json)
  #   end
  # end
  # ```
  def dump(json : JSON::Builder) : Nil
    result = snapshot_session

    json.field "version", 1
    json.field "session_id", result.session_id
    json.field "cache_key", result.cache_key
    json.field "history", result.history

    # Persist only the names of enabled tools.
    # The actual tool definitions (with callbacks) are application code
    # and must be re-registered before load.
    tools = result.enabled_tools
    unless tools.empty?
      json.field "enabled_tools", tools
    end
  end

  # Serialise the current session to a JSON string.
  # Includes session_id, cache_key, and the full message history.
  # The output is suitable for later use with `Agent.load`.
  #
  # ```
  # File.write("session.json", agent.dump)
  # ```
  def dump : String
    JSON.build do |json|
      json.object do
        dump(json)
      end
    end
  end

  # Internal: fetch session snapshot.
  # Replaces the earlier channel-based DumpRequest round-trip.
  private def snapshot_session : DumpResult
    history_dup = @history.lock(&.dup)
    enabled_names = @registered_tools.lock(&.select { |_, v| v[:enabled] }.keys)
    DumpResult.new(
      session_id: @session_id,
      cache_key: @cache_key,
      history: history_dup,
      enabled_tools: enabled_names,
    )
  end

  # Result carrier for session snapshot.
  private record DumpResult,
    session_id : String,
    cache_key : String,
    history : Array(Message),
    enabled_tools : Array(String)

  # Create an Agent and restore session state from previously dumped data.
  #
  # Convenience shorthand equivalent to:
  # ```
  # agent = Agent.new(config, provider: provider)
  # agent.register_tool(...)
  # agent.load(data)
  # ```
  #
  # Accepts a raw JSON string (shorthand) or a pre-parsed `JSON::Any` hash.
  # Raises `Agent::SessionLoadError` if the session data is corrupt.
  def self.load(config : Config, data : String | JSON::Any, provider : Provider::Base? = nil) : Agent
    agent = new(config, provider: provider)
    agent.load(data)
    agent
  end

  # ---------------------------------------------------------------------------
  # Internal
  # ---------------------------------------------------------------------------

  private def build_messages(history : Array(Message), msgs_to_append : Array(Message)) : Array(Message)
    msgs = [] of Message

    if (sys = @config.system_prompt) && !sys.empty?
      msgs << Message.new(role: Role::System, content: sys)
    end

    msgs.concat(history)
    msgs.concat(msgs_to_append)
    msgs
  end

  private def run_loop : Nil
    loop do
      request = @request_channel.receive
      case request
      in AskRequest
        process_request_loop(request)
      end
    end
  rescue Channel::ClosedError
    # agent fiber exits on close — any fiber blocked in a send to the
    # request channel will also see ClosedError and raise to its caller.
  end

  # Process a request with automatic tool resolution.
  # When the model returns tool calls and auto_execute_tools is enabled,
  # registered tools are executed inline and the result is sent back to the
  # model — all within this fiber, without returning to the caller.
  #
  # @history and @registered_tools are accessed through Sync::Exclusive.
  # The long HTTP call (http_post_stream) happens outside any lock.
  private def process_request_loop(request : AskRequest) : Nil
    response = request.response
    tools = request.tools
    max_iter = @config.max_tool_iterations
    iteration = 0

    # Append new messages to history under the lock, then snapshot for the HTTP call.
    history_snapshot = @history.lock do |h|
      h.concat(request.new_messages)
      h.dup
    end
    messages = build_messages(history_snapshot, [] of Message)

    # Accumulates each iteration's usage so `Response#metadata` reports the
    # whole turn — every provider call of the tool loop — not just the
    # finishing iteration's counts.
    total_usage = Usage.new

    loop do
      iteration += 1

      # Pre-flight bail-out checks (closed / cancelled / iteration limit).
      # Returns true if the loop should break.
      if preflight_bailout?(response, iteration, max_iter)
        break
      end

      response.chunk_handler = ->(chunk : Response::Chunk) {
        @handlers.decorate(ChunkContext.new(chunk), &.chunk)
      }
      msg, usage, finish_reason = http_post_stream(messages, tools, response)

      # Cancellation is a user-initiated abort, not a failure: the partial
      # assistant reply and the user's messages are real conversation context
      # and must be preserved so the next #ask has continuity.
      if response.cancelled?
        handle_cancellation(msg, response)
        break
      end

      # On a genuine error (network/API/JSON), http_post_stream already called
      # response.finish_with_error. Roll back the user messages that were
      # appended at the start of this request so history is not corrupted for
      # subsequent #ask calls (the turn never produced a usable assistant reply).
      if response.error?
        @history.lock { |h| request.new_messages.each { h.pop } }
        break
      end

      # Append the assistant message to history under the lock.
      @history.lock { |h| h << msg }
      total_usage = total_usage + usage

      # Determine whether to auto-resolve tools (read under the lock).
      has_registered_tools = @registered_tools.lock(&.empty?.!)

      # If no tool calls, or auto_execute is disabled, or no registered tools — done.
      no_tools = !msg.has_tool_calls? || !@config.auto_execute_tools? || !has_registered_tools
      if no_tools
        trim_history!
        response.finish(msg, total_usage, finish_reason: finish_reason)
        break
      end

      # ameba:disable Lint/NotNil
      tool_calls = msg.tool_calls.not_nil!
      results = execute_registered_tools(tool_calls, response)

      # If some tools had no registered handler, stop and let the caller handle it.
      # NOTE: The assistant(tool_calls) is already in @history. The caller MUST
      # follow up with #ask(tool_results) so the tool messages follow the
      # assistant(tool_calls) — otherwise the next API request will be invalid.
      if results.empty?
        trim_history!
        response.finish(msg, total_usage, finish_reason: finish_reason)
        break
      end

      # Append tool results to history under the lock, then rebuild messages for next iteration.
      @history.lock(&.concat(results))
      history_snapshot = @history.lock(&.dup)
      messages = build_messages(history_snapshot, [] of Message)
    end
  end

  # Preserve the canceled turn in history.
  #
  # Cancellation is a user-initiated abort, not a failure: whatever assistant
  # content was streamed before the cancel is real conversation context and is
  # appended to history so the next #ask has continuity (the user's question
  # was already appended at the start of #process_request_loop and is kept).
  # The response is then finished with a CancelledError.
  private def handle_cancellation(partial_msg : Message, response : Response) : Nil
    has_content = !partial_msg.content.nil? || !partial_msg.tool_calls.nil?
    @history.lock { |h| h << partial_msg } if has_content
    trim_history!
    response.finish_with_error(CancelledError.new)
  end

  # Checks performed at the top of each tool-resolution iteration.
  # Returns true (and finishes the response) if the loop should break.
  private def preflight_bailout?(response : Response, iteration : Int32, max_iter : Int32?) : Bool
    if @state_mutex.synchronize { @closed }
      response.finish_with_error(Agent::Error.new("Agent was closed"))
      return true
    end

    if response.cancelled?
      response.finish_with_error(CancelledError.new)
      return true
    end

    if max_iter && iteration > max_iter
      err_msg = "Agent error: tool call iteration limit (#{max_iter}) exceeded"
      response.finish_with_error(ToolLoopError.new(max_iter, err_msg))
      return true
    end

    false
  end

  # Trim history to respect max_history config, applied only after a
  # complete user->assistant turn (not during intermediate tool-call steps).
  #
  # A turn is a user message and everything up to the next user message:
  # the assistant's reply, and with tools, its tool-call messages and their
  # results. The oldest turns are dropped whole until at most `max_history`
  # remain, so the kept history always starts at a user message and no
  # tool result is ever left without the assistant message that called it.
  private def trim_history! : Nil
    @history.replace do |h|
      max = @config.max_history
      next h unless max && max > 0

      starts = [] of Int32
      h.each_with_index { |m, idx| starts << idx if m.role == Role::User }
      next h if starts.size <= max

      first_kept = starts[starts.size - max]
      h[first_kept..]
    end
  end

  # Execute all tool calls that have registered callbacks.
  # Returns the array of tool-result messages. Returns an **empty** array if any
  # tool call has no registered handler.
  # If a callback raises, the error is caught and returned as a tool-result
  # message with an error description, so the agent fiber never dies.
  private def execute_registered_tools(tool_calls : Array(ToolCall), response : Response) : Array(Message)
    results = [] of Message

    tool_calls.each do |tc|
      entry = @registered_tools.lock { |t| t[tc.name]? }
      if entry.nil? || !entry[:enabled]
        return [] of Message
      end

      # Parse the JSON arguments string into a hash for the callback.
      args_hash = begin
        parsed = JSON.parse(tc.arguments)
        parsed.as_h? || {} of String => JSON::Any
      rescue JSON::ParseException
        results << Message.new(
          role: Role::Tool,
          content: "Error parsing arguments for tool '#{tc.name}': invalid JSON",
          tool_call_id: tc.id,
          name: tc.name,
          is_error: true,
        )
        next
      end

      # Validate arguments against the tool's parameter schema.
      if params_schema = entry[:tool].function.parameters
        errors = validate_tool_args(tc.name, args_hash, params_schema)
        unless errors.empty?
          results << Message.new(
            role: Role::Tool,
            content: "Error validating arguments for tool '#{tc.name}': #{errors.join("; ")}",
            tool_call_id: tc.id,
            name: tc.name,
            is_error: true,
          )
          next
        end
      end

      result_msg = @handlers.decorate(ToolCallContext.new(tc, response)) do |ctx|
        failed = false
        result = if cb = entry[:callback]
                   begin
                     cb.call(args_hash)
                   rescue ex
                     failed = true
                     "Error executing tool '#{ctx.tool_call.name}': #{ex.message}"
                   end
                 else
                   failed = true
                   "Error executing tool '#{ctx.tool_call.name}': no callback registered"
                 end

        Message.new(
          role: Role::Tool,
          content: result,
          tool_call_id: ctx.tool_call.id,
          name: ctx.tool_call.name,
          is_error: failed ? true : nil,
        )
      end

      results << result_msg
    end

    results
  end

  # Validate tool call arguments against the tool's parameter JSON Schema.
  # Returns an array of error messages (empty = valid).
  private def validate_tool_args(tool_name : String, args : Hash(String, JSON::Any), schema : Hash(String, JSON::Any)) : Array(String)
    errors = [] of String

    props = schema["properties"]?.try(&.as_h?) || {} of String => JSON::Any
    required = schema["required"]?.try(&.as_a?.try(&.map(&.as_s))) || [] of String

    # Check required fields are present.
    required.each do |field|
      unless args.has_key?(field)
        errors << "Missing required field '#{field}'"
      end
    end

    # Check that all provided arguments have known types.
    args.each do |key, value|
      next unless prop = props[key]?

      # `as_s?` (not `as_s`): a schema may declare `type` as an array of
      # alternatives (e.g. ["string", "null"]); casting that to String would
      # raise here and kill the agent fiber.
      expected = prop["type"]?.try(&.as_s?)
      next unless expected

      actual = type_name(value)
      if actual && !type_matches?(actual, expected)
        errors << "Field '#{key}' expected type '#{expected}', got '#{actual}'"
      end

      errors.concat(array_item_errors(key, value, prop)) if actual == "array"
    end

    errors
  end

  # Validate array element types against a property schema's `items` type.
  private def array_item_errors(key : String, value : JSON::Any, prop : JSON::Any) : Array(String)
    errors = [] of String
    return errors unless elements = value.as_a?

    items = prop["items"]?.try(&.as_h?)
    item_type = items.try { |h| h["type"]?.try(&.as_s?) }
    return errors unless item_type

    elements.each_with_index do |item, index|
      item_actual = type_name(item)
      if item_actual && !type_matches?(item_actual, item_type)
        errors << "Field '#{key}' expected array items of type '#{item_type}', got '#{item_actual}' at index #{index}"
      end
    end

    errors
  end

  private def type_matches?(actual_type : String, expected_type : String) : Bool
    case expected_type
    when "integer"
      actual_type == "integer" || actual_type == "number"
    when "number"
      actual_type == "integer" || actual_type == "number"
    when "array"
      actual_type == "array"
    when "object"
      actual_type == "object"
    when "string"
      actual_type == "string"
    when "boolean"
      actual_type == "boolean"
    when "null"
      actual_type == "null"
    else
      true
    end
  end

  # Return the JSON type name of a JSON::Any value.
  private def type_name(value : JSON::Any) : String?
    case value.raw
    when String
      "string"
    when Int64, Int32, Int16, Int8
      "number"
    when Float64, Float32
      "number"
    when Bool
      "boolean"
    when Array
      "array"
    when Hash
      "object"
    when Nil
      # JSON null is a real type: a property declaring `type` never accepts
      # null (JSON Schema), so it must be reported and fail the type check
      # like any other mismatch instead of passing validation silently.
      "null"
    end
  end

  # Returns the combined tool list: per-request tools merged with registered tools.
  # Registered tools take precedence over per-request tools with the same name.
  private def combined_tools(tools : Array(Tool)?) : Array(Tool)?
    enabled_reg = @registered_tools.lock { |t| t.select { |_, v| v[:enabled] } }

    if enabled_reg.empty?
      tools
    else
      reg = enabled_reg.values.map(&.[:tool])

      if tools
        # Filter out per-request tools whose name collides with enabled registered tools
        filtered = tools.reject { |t| enabled_reg.has_key?(t.function.name) }
        filtered + reg
      else
        reg
      end
    end
  end

  # Perform the HTTP POST to the provider and parse the streaming response.
  # This is a thin shim that delegates to the provider for wire-format concerns.
  #
  # Per-read SSE timeouts (first-byte vs idle) come from `Agent::Config` and
  # are applied both to the HTTP client (so the underlying socket honours them)
  # and threaded through to `Provider#parse_stream` (which can re-apply per
  # read). `IO::TimeoutError` and `IdleTimeoutError` are caught here and surface
  # as `Agent::IdleTimeoutError` via the response's error path, mirroring the
  # `CancelledError` recovery (rebuild the connection, keep the partial msg).
  private def http_post_stream(
    messages : Array(Message),
    tools : Array(Tool)?,
    response : Response,
  ) : {Message, Usage, String?}
    @handlers.decorate(TurnContext.new(messages, tools)) do |ctx|
      all_tools = combined_tools(ctx.tools)
      req = @provider.build_request(ctx.messages, all_tools)

      first_byte_timeout, idle_timeout = apply_request_timeouts(req[:body].bytesize)
      client = @http_client
      partial_msg = nil

      begin
        client.post(req[:path], headers: req[:headers], body: req[:body]) do |http_resp|
          unless http_resp.status.ok?
            raise ApiError.new(http_resp.status_code, "#{http_resp.status_code} #{http_resp.status_message}")
          end

          msg, usage, finish_reason = @provider.parse_stream(
            http_resp.body_io,
            response,
            -> { response.cancelled? },
            first_byte_timeout: first_byte_timeout,
            idle_timeout: idle_timeout,
          )

          # If the response was cancelled mid-stream, treat it as an error.
          # Keep the partial assistant message so the caller can preserve it
          # in history, and raise to abort the HTTP block.
          if response.cancelled?
            partial_msg = msg
            raise CancelledError.new
          end

          {msg, usage, finish_reason}
        end
      rescue ex
        handle_post_error(ex, response, partial_msg, first_byte_timeout, idle_timeout)
      end
    end
  end

  # Centralised error handler for #http_post_stream. Dispatches on the
  # exception type, decorates it through the handler chain, calls
  # `response.finish_with_error`, and returns the canonical error tuple to the
  # caller (preserving any partial assistant message captured before the
  # failure). Active connections are rebuilt after cancellation / timeout /
  # mid-line read failures because the socket may be in an unknown state.
  private def handle_post_error(
    ex : Exception,
    response : Response,
    partial_msg : Message?,
    first_byte_timeout : Time::Span?,
    idle_timeout : Time::Span,
  ) : {Message, Usage, String?}
    case ex
    when ApiError
      err = @handlers.decorate(ErrorContext.new(ex)) { |c| c.error.as(Agent::Error) }
      response.finish_with_error(err)
      error_tuple(err)
    when CancelledError, IdleTimeoutError
      # Cancellation / per-read SSE timeout was already wrapped by the parser
      # for the idle case, or raised here directly for cancellation. Either
      # leaves the persistent HTTP connection with an unconsumed/partial
      # response body, which would corrupt the next request on the same client
      # ("Invalid HTTP response"). Discard and build a fresh one.
      rebuild_http_client
      err = @handlers.decorate(ErrorContext.new(ex)) { |c| c.error.as(Agent::Error) }
      response.finish_with_error(err)
      error_tuple(err, partial_msg)
    when IO::TimeoutError
      # Defensive fallback: the parser wraps its IO reads and raises
      # IdleTimeoutError directly, but other IO operations in this shim (eg.
      # request body upload) might still raise IO::TimeoutError. Recover the
      # same way as IdleTimeoutError with phase inferred from partial_msg.
      rebuild_http_client
      phase = partial_msg.nil? ? :first_byte : :idle
      err = @handlers.decorate(ErrorContext.new(ex)) do |c|
        IdleTimeoutError.new(phase, first_byte_timeout || idle_timeout, cause: c.error)
      end
      response.finish_with_error(err)
      error_tuple(err, partial_msg)
    else
      err = @handlers.decorate(ErrorContext.new(ex)) do |c|
        ConnectionError.new(c.error.message || c.error.class.name, cause: c.error)
      end
      response.finish_with_error(err)
      error_tuple(err)
    end
  end

  # Compute the per-request SSE read timeouts from `Agent::Config` and apply
  # them to the persistent HTTP client. Returns the {first_byte, idle} timeouts
  # to thread through to the provider's `#parse_stream`.
  #
  # `prompt_bytes` is the size of the wire-format request body; the first-byte
  # budget scales with prompt size per AGENTS.md. When dynamic first-byte is
  # disabled (`compute_first_byte_timeout? == false`), `first_byte` is nil and
  # both reads fall back to `idle_byte_timeout`.
  #
  # The HTTP client's `read_timeout` is set to `Config#effective_read_timeout`
  # (the smaller of first_byte_timeout and idle_byte_timeout on a real
  # connection), because HTTP::Client only honours `read_timeout` at
  # socket-open time — there's no portable way to mutate the per-read timeout
  # mid-stream on `http_resp.body_io`. The provider's `parse_stream` re-applies
  # per-read timeouts on the body_io when it supports `read_timeout=` (a no-op
  # for HTTP::ChunkedContent) so the first-byte-generous-then-idle-tight
  # switch is honored on IOs that surface the setter.
  #
  # Also: rebuild the client when the desired value changes so the new value
  # sticks on a fresh socket (pooled connections don't pick up a new
  # `client.read_timeout` after open).
  private def apply_request_timeouts(prompt_bytes : Int) : {Time::Span?, Time::Span}
    first_byte_timeout = @config.compute_first_byte_timeout(prompt_bytes)
    idle_timeout = @config.idle_byte_timeout
    desired_rt = @config.effective_read_timeout(prompt_bytes)
    if desired_rt && desired_rt != @applied_read_timeout
      apply_client_read_timeout(desired_rt)
    end
    {first_byte_timeout, idle_timeout}
  end

  # Build the canonical error return tuple for #http_post_stream: an assistant
  # message whose content is the prefix-decorated error message, plus empty
  # usage and nil finish_reason. If `partial` is provided (e.g. a partial
  # assistant message captured before cancellation / timeout), it is
  # preserved so the caller can keep any streamed tokens in history.
  private def error_tuple(err : Agent::Error, partial : Message? = nil) : {Message, Usage, String?}
    msg = partial || Message.new(role: Role::Assistant, content: "Agent error: #{err.message || err.class.name}")
    {msg, Usage.new, nil}
  end

  private def build_http_client : HTTP::Client
    uri = @provider.base_uri
    client = HTTP::Client.new(uri)

    # The legacy read_timeout from config (defaults to nil). When callers do
    # not set it explicitly, it remains nil and the dynamic per-read SSE
    # timeouts (set in #http_post_stream via #apply_client_read_timeout) take
    # over. If both are set, apply_client_read_timeout overrides this on the
    # first request whose dynamic timeout differs.
    if (rt = @config.read_timeout) && rt > Time::Span.zero
      client.read_timeout = rt
      @applied_read_timeout = rt
    else
      @applied_read_timeout = nil
    end
    if (ct = @config.connect_timeout) && ct > Time::Span.zero
      client.connect_timeout = ct
    end

    client
  end

  # Apply a per-read timeout to the persistent HTTP client. Because the
  # underlying socket only honours `read_timeout` at open time, a value change
  # requires rebuilding the client so the next POST opens a fresh socket with
  # the new budget. The parser (StreamParser.parse) additionally re-applies
  # `read_timeout=` to the body_io per successful read on IOs that support it
  # (true sockets); this handles the first-byte → idle timeout switch.
  private def apply_client_read_timeout(timeout : Time::Span) : Nil
    return unless timeout > Time::Span.zero
    @http_client.read_timeout = timeout
    # If the timeout differs from what was applied at socket open, rebuild so
    # the new value sticks on a fresh connection.
    if timeout != @applied_read_timeout
      rebuild_http_client
      @http_client.read_timeout = timeout
      @applied_read_timeout = timeout
    end
  end

  # Replace the persistent HTTP client with a fresh one.
  # Used after a cancelled stream leaves the previous connection with an
  # unconsumed response body, which would corrupt the next request.
  private def rebuild_http_client : Nil
    @http_client.close
  rescue IO::Error
    # ignore close errors on a possibly-broken socket
  ensure
    @http_client = build_http_client
  end
end

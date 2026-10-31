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
  private record AskRequest, new_messages : Array(Message), tools : Array(Tool)?, response : Response, options : RequestOptions

  # Version written by #dump; #load refuses a newer one.
  DUMP_VERSION = 2

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
  # A request in flight finishes its current provider call; the turn then
  # ends with a closed error instead of continuing its tool loop. The agent
  # fiber releases the HTTP client and provider as it exits: closing a TLS
  # socket from here, while that fiber is mid-read on it, would re-enter
  # OpenSSL.
  def close : Nil
    @state_mutex.synchronize { return if @closed; @closed = true }
    @request_channel.close
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
  # `reasoning_effort` overrides `Config#reasoning_effort` for this request.
  #
  # Raises Agent::ClosedError if the agent has been closed via #close.
  def ask(content : String, attachments : Array(String)? = nil, tools : Array(Tool)? = nil, reasoning_effort : ReasoningEffort? = nil) : Response
    @state_mutex.synchronize { raise ClosedError.new if @closed }

    msg = if atts = attachments
            parts = [ContentPart.new(type: :text, text: content)] + atts.map { |path| ContentPart.from_path(path) }
            Message.new(role: Role::User, content_parts: parts)
          else
            Message.new(role: Role::User, content: content)
          end

    response = Response.new
    @request_channel.send(AskRequest.new([msg], tools, response, RequestOptions.new(reasoning_effort: reasoning_effort)))
    response
  rescue Channel::ClosedError
    raise ClosedError.new
  end

  # Send tool result messages and return a Response immediately.
  # Used to feed tool call results back to the model after it requests them.
  #
  # Raises Agent::ClosedError if the agent has been closed via #close.
  def ask(tool_results : Array(Message), tools : Array(Tool)? = nil, reasoning_effort : ReasoningEffort? = nil) : Response
    @state_mutex.synchronize { raise ClosedError.new if @closed }

    response = Response.new
    @request_channel.send(AskRequest.new(tool_results, tools, response, RequestOptions.new(reasoning_effort: reasoning_effort)))
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
    check_dump_version(parsed)

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

  # A dump without a version predates versioning (version 1).
  private def check_dump_version(parsed : Hash(String, JSON::Any)) : Nil
    raw = parsed["version"]?
    return if raw.nil?
    version = raw.as_i?
    raise SessionLoadError.new("'version' field is not an integer") if version.nil?
    if version > DUMP_VERSION
      raise SessionLoadError.new("version #{version} is newer than this agent-cr supports (#{DUMP_VERSION})")
    end
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

    json.field "version", DUMP_VERSION
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
  ensure
    @http_client.close
    @provider.close
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
    # History size before this turn, so a failed first call can be rolled back.
    start_size : Int32? = nil

    # Accumulates each iteration's usage so `Response#metadata` reports the
    # whole turn — every provider call of the tool loop — not just the
    # finishing iteration's counts. Reported on failure too.
    total_usage = Usage.new

    begin
      # Append new messages to history under the lock, then snapshot for the HTTP call.
      history_snapshot = @history.lock do |h|
        close_dangling_tool_calls(h, request.new_messages)
        start_size = h.size
        h.concat(request.new_messages)
        h.dup
      end
      messages = build_messages(history_snapshot, [] of Message)

      loop do
        iteration += 1

        # Pre-flight bail-out checks (closed / cancelled / iteration limit).
        # Returns true if the loop should break.
        if preflight_bailout?(response, iteration, max_iter, total_usage)
          break
        end

        response.chunk_handler = ->(chunk : Response::Chunk) {
          @handlers.decorate(ChunkContext.new(chunk), &.chunk)
        }
        msg, usage, finish_reason = http_post_stream(messages, tools, request.options, response, total_usage)
        total_usage = total_usage + usage

        # Cancellation is a user-initiated abort, not a failure: the partial
        # assistant reply and the user's messages are real conversation context
        # and must be preserved so the next #ask has continuity.
        if response.cancelled?
          handle_cancellation(msg, response, total_usage)
          break
        end

        # On a genuine error (network/API/JSON), http_post_stream already called
        # response.finish_with_error.
        if response.error?
          rollback_failed_turn(start_size, request.new_messages.size)
          break
        end

        # Append the assistant message to history under the lock.
        @history.lock { |h| h << msg }

        # Done unless the loop runs this message's tools itself.
        tool_calls = auto_tool_calls(msg, tools)
        unless tool_calls
          trim_history!
          response.finish(msg, total_usage, finish_reason: finish_reason)
          break
        end

        results = execute_registered_tools(tool_calls, response)

        # Append tool results to history under the lock, then rebuild messages for next iteration.
        @history.lock(&.concat(results))
        history_snapshot = @history.lock(&.dup)
        messages = build_messages(history_snapshot, [] of Message)
      end
    rescue ex
      # Anything escaping the loop (a handler, a provider, a tool schema) must
      # finish the response; an unhandled exception would kill this fiber and
      # hang every caller.
      Log.error(exception: ex) { "Agent: request failed" } unless ex.is_a?(Agent::Error)
      err = ex.is_a?(Agent::Error) ? ex : Agent::Error.new(ex.message || ex.class.name, cause: ex)
      rollback_failed_turn(start_size, request.new_messages.size) unless response.finished?
      response.finish_with_error(err, total_usage)
    end
  end

  # A turn that fails before the model replied produced nothing: drop the
  # messages it added. Once a reply is in history, tools may have run, so
  # the turn is kept; any call left unanswered is answered at the start of
  # the next turn.
  private def rollback_failed_turn(start_size : Int32?, added : Int32) : Nil
    return unless start_size
    @history.lock { |h| h.pop(h.size - start_size) if start_size < h.size <= start_size + added }
  end

  # Give every tool call of the last assistant message that has no result
  # (and none among `incoming`) an error result, so the next request is valid.
  # This happens when a caller skips the #ask(tool_results) follow-up or a
  # turn fails midway through its tools.
  private def close_dangling_tool_calls(history : Array(Message), incoming : Array(Message)) : Nil
    idx = history.rindex { |m| m.role == Role::Assistant } || return
    calls = history[idx].tool_calls || return

    pos = idx + 1
    answered = Set(String).new
    while pos < history.size && history[pos].role == Role::Tool
      history[pos].tool_call_id.try { |id| answered << id }
      pos += 1
    end
    incoming.each { |m| m.tool_call_id.try { |id| answered << id } if m.role == Role::Tool }

    calls.each do |tc|
      next if answered.includes?(tc.id)
      history.insert(pos, Message.tool_result(tc, "Error: this tool call was not completed", is_error: true))
      pos += 1
    end
  end

  # The tool calls the loop should execute, or nil to hand the message back
  # to the caller: no tool calls, auto-execution off, no registered tools,
  # or a call to a tool passed to #ask(tools:) that is not an enabled
  # registered one. Such a call is the caller's to dispatch, so the whole
  # message goes back before anything runs; no registered tool runs and
  # then has its result discarded.
  # NOTE: The assistant(tool_calls) is already in @history. The caller
  # should follow up with #ask(tool_results); any call left unanswered gets
  # an error result at the start of the next turn.
  private def auto_tool_calls(msg : Message, tools : Array(Tool)?) : Array(ToolCall)?
    tool_calls = msg.tool_calls
    return if tool_calls.nil? || tool_calls.empty? || !@config.auto_execute_tools?

    enabled = @registered_tools.lock do |t|
      next if t.empty?
      t.select { |_, v| v[:enabled] }.keys.to_set
    end
    return unless enabled

    if tools
      request_names = tools.map(&.function.name).to_set
      return if tool_calls.any? { |tc| request_names.includes?(tc.name) && !enabled.includes?(tc.name) }
    end
    tool_calls
  end

  # Preserve the canceled turn in history.
  #
  # Cancellation is a user-initiated abort, not a failure: whatever assistant
  # content was streamed before the cancel is real conversation context and is
  # appended to history so the next #ask has continuity (the user's question
  # was already appended at the start of #process_request_loop and is kept).
  # Tool calls are dropped: they never ran, and their arguments may be cut
  # off mid-stream. So is the provider's replay data: a reasoning block cut
  # off before its signature would be refused when sent back. The response
  # is then finished with a CancelledError.
  private def handle_cancellation(partial_msg : Message, response : Response, usage : Usage) : Nil
    kept = Message.new(role: Role::Assistant, content: partial_msg.content, reasoning: partial_msg.reasoning)
    @history.lock { |h| h << kept } unless kept.content.nil?
    trim_history!
    response.finish_with_error(CancelledError.new, usage)
  end

  # Checks performed at the top of each tool-resolution iteration.
  # Returns true (and finishes the response) if the loop should break.
  private def preflight_bailout?(response : Response, iteration : Int32, max_iter : Int32?, usage : Usage) : Bool
    if @state_mutex.synchronize { @closed }
      response.finish_with_error(Agent::Error.new("Agent was closed"), usage)
      return true
    end

    if response.cancelled?
      response.finish_with_error(CancelledError.new, usage)
      return true
    end

    if max_iter && iteration > max_iter
      err_msg = "tool call iteration limit (#{max_iter}) exceeded"
      response.finish_with_error(ToolLoopError.new(max_iter, err_msg), usage)
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
  # Returns one tool-result message per call. A call to a tool that is not
  # registered or is disabled (usually a hallucinated name) gets an error
  # result so the model can correct itself.
  # If a callback raises, the error is caught and returned as a tool-result
  # message, so the agent fiber never dies. Only an `Agent::ToolError`'s
  # message reaches the model; anything else is logged.
  private def execute_registered_tools(tool_calls : Array(ToolCall), response : Response) : Array(Message)
    results = [] of Message

    tool_calls.each do |tc|
      # A cancel stops the remaining calls; each still gets a result so the
      # history stays valid.
      if response.cancelled?
        results << Message.tool_result(tc, "Error: cancelled before this tool ran", is_error: true)
        next
      end

      entry = @registered_tools.lock { |t| t[tc.name]? }
      if entry.nil? || !entry[:enabled]
        results << Message.new(
          role: Role::Tool,
          content: "Error: tool '#{tc.name}' is not available",
          tool_call_id: tc.id,
          name: tc.name,
          is_error: true,
        )
        next
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
                   rescue ex : ToolError
                     failed = true
                     ctx.error = ex
                     "Error executing tool '#{ctx.tool_call.name}': #{ex.message}"
                   rescue ex
                     failed = true
                     ctx.error = ex
                     Log.error(exception: ex) { "Agent: tool '#{ctx.tool_call.name}' raised" }
                     "Error executing tool '#{ctx.tool_call.name}': the tool failed"
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
    options : RequestOptions,
    response : Response,
    turn_usage : Usage,
  ) : {Message, Usage, String?}
    @handlers.decorate(TurnContext.new(messages, tools, options)) do |ctx|
      all_tools = combined_tools(ctx.tools)
      req = @provider.build_request(ctx.messages, all_tools, ctx.options)
      post_with_retries(req, response, turn_usage)
    end
  end

  # POST the request and parse its stream. A failure is retried up to
  # `Config#max_retries` times with backoff, honouring `Retry-After`, when it
  # is transient and no output has streamed yet; once it has, nothing is
  # retried. A failure after the provider accepted the request (HTTP 200)
  # is a StreamError, whatever broke.
  private def post_with_retries(
    req : NamedTuple(path: String, headers: HTTP::Headers, body: String),
    response : Response,
    turn_usage : Usage,
  ) : {Message, Usage, String?}
    first_byte_timeout, idle_timeout = apply_request_timeouts(req[:body].bytesize)
    attempt = 0

    loop do
      client = @http_client
      partial_msg = nil
      accepted = false
      chunks_before = response.chunks_pushed

      begin
        raise CancelledError.new if response.cancelled?

        return client.post(req[:path], headers: req[:headers], body: req[:body]) do |http_resp|
          raise api_error(http_resp) unless http_resp.status.ok?
          accepted = true
          raise json_instead_of_stream(http_resp) if json_body?(http_resp)

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
        ex = stream_failure(ex) if accepted
        # Retry only while this call has shown nothing; an earlier call of
        # the tool loop may have.
        output = response.chunks_pushed != chunks_before
        delay = retry_delay(ex, attempt) unless output || response.cancelled?
        return handle_post_error(ex, response, partial_msg, first_byte_timeout, idle_timeout, turn_usage) unless delay

        attempt += 1
        rebuild_http_client
        wait_unless_cancelled(delay, response)
      end
    end
  end

  # A failure once the request was accepted, as a StreamError: the socket
  # dropping (plain or TLS) is transient, anything else (a malformed stream)
  # is not. Agent errors, and timeouts, keep their own type.
  private def stream_failure(ex : Exception) : Exception
    case ex
    when Agent::Error, IO::TimeoutError
      ex
    when IO::Error, OpenSSL::Error
      StreamError.new("connection lost during the stream: #{ex.message}", cause: ex, retryable: true)
    else
      StreamError.new("could not read the stream: #{ex.message || ex.class.name}", cause: ex)
    end
  end

  # How long to wait before retrying after `ex`, or nil to give up. A
  # `Retry-After` longer than `Config#retry_max_delay` is not waited out.
  private def retry_delay(ex : Exception, attempt : Int32) : Time::Span?
    return if attempt >= @config.max_retries
    return unless retryable?(ex)

    if ex.is_a?(ApiError) && (wait = ex.retry_after)
      return wait <= @config.retry_max_delay ? wait : nil
    end
    backoff = @config.retry_base_delay * (2 ** attempt) * Random.rand(0.5..1.0)
    backoff < @config.retry_max_delay ? backoff : @config.retry_max_delay
  end

  # Connection errors are transient. Of timeouts, only connecting is: a
  # provider slow to answer is not retried, since that doubles an already
  # long wait. Of TLS errors, only a session the peer cut off: a bad
  # certificate fails the same way again.
  private def retryable?(ex : Exception) : Bool
    case ex
    when ApiError, StreamError then ex.retryable?
    when IO::TimeoutError      then ex.message.try(&.downcase.starts_with?("connect timed out")) || false
    when IO::Error             then true
    when OpenSSL::SSL::Error   then ex.underlying_eof?
    else                            false
    end
  end

  private def wait_unless_cancelled(delay : Time::Span, response : Response) : Nil
    deadline = Time.instant + delay
    while !response.cancelled? && (left = deadline - Time.instant) > Time::Span.zero
      sleep(left < 50.milliseconds ? left : 50.milliseconds)
    end
  end

  # An ApiError for a non-2xx response, with the start of its body, its
  # headers, and the provider's error message when the body has one.
  private def api_error(http_resp : HTTP::Client::Response) : ApiError
    body = IO::Memory.new
    if io = http_resp.body_io?
      begin
        IO.copy(io, body, ApiError::BODY_LIMIT)
      rescue IO::Error
        # keep what was read
      end
    end
    text = body.to_s

    message = "#{http_resp.status_code} #{http_resp.status_message}"
    if detail = provider_error_message(text)
      message += ": #{detail}"
    end
    ApiError.new(http_resp.status_code, message, body: text.presence, headers: http_resp.headers)
  end

  # Some gateways answer a streaming request with a 200 and a JSON error
  # body instead of an event stream.
  private def json_body?(http_resp : HTTP::Client::Response) : Bool
    http_resp.headers["Content-Type"]?.try(&.downcase.starts_with?("application/json")) || false
  end

  private def json_instead_of_stream(http_resp : HTTP::Client::Response) : StreamError
    text = String.build { |io| http_resp.body_io?.try { |b| IO.copy(b, io, ApiError::BODY_LIMIT) } }
    detail = provider_error_message(text) || "no event stream"
    StreamError.new("provider replied with JSON instead of a stream: #{detail}")
  end

  # `error.message` from an OpenAI-style JSON error body.
  private def provider_error_message(text : String) : String?
    error = JSON.parse(text).as_h?.try(&.["error"]?) || return
    error.as_s? || error.as_h?.try(&.["message"]?).try(&.as_s?)
  rescue JSON::ParseException
    nil
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
    turn_usage : Usage,
  ) : {Message, Usage, String?}
    # Every failure can leave an unread body (an API error's JSON, a
    # half-read stream) on the kept-alive connection, which the next request
    # would read as its response ("Unsupported HTTP version: {...").
    rebuild_http_client

    # A failure racing a cancel is the cancel; nothing streamed is kept.
    if response.cancelled? && !ex.is_a?(CancelledError)
      ex = CancelledError.new
      partial_msg ||= Message.new(role: Role::Assistant)
    end

    case ex
    when CancelledError, IdleTimeoutError
      err = @handlers.decorate(ErrorContext.new(ex)) { |c| c.error.as(Agent::Error) }
      response.finish_with_error(err, turn_usage)
      error_tuple(err, partial_msg)
    when IO::TimeoutError
      # Defensive fallback: the parser wraps its IO reads and raises
      # IdleTimeoutError directly, but other IO operations in this shim (eg.
      # request body upload) might still raise IO::TimeoutError. Recover the
      # same way as IdleTimeoutError with phase inferred from partial_msg.
      phase = partial_msg.nil? ? :first_byte : :idle
      err = @handlers.decorate(ErrorContext.new(ex)) do |c|
        IdleTimeoutError.new(phase, first_byte_timeout || idle_timeout, cause: c.error)
      end
      response.finish_with_error(err, turn_usage)
      error_tuple(err, partial_msg)
    when Agent::Error
      # ApiError, StreamError, or one a handler raised: passed through.
      err = @handlers.decorate(ErrorContext.new(ex)) { |c| c.error.as(Agent::Error) }
      response.finish_with_error(err, turn_usage)
      error_tuple(err)
    else
      err = @handlers.decorate(ErrorContext.new(ex)) do |c|
        ConnectionError.new(c.error.message || c.error.class.name, cause: c.error)
      end
      response.finish_with_error(err, turn_usage)
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

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

  # What an agent holds in memory, roughly: the history's messages and
  # their bytes, and the background tasks (running, and finished but not
  # yet delivered) with the bytes of their results. Cheap enough to poll,
  # for finding the sessions that have grown large. The byte counts are of
  # the strings held, not of the heap: close enough to compare sessions and
  # set an alarm on, not exact.
  record Footprint, messages : Int32, history_bytes : Int64, tasks : Int32, task_bytes : Int64 do
    def bytes : Int64
      history_bytes + task_bytes
    end
  end

  # See `Footprint`. Thread-safe; takes each lock briefly.
  def footprint : Footprint
    messages, history_bytes = @history.lock { |h| {h.size, h.sum(0_i64, &.bytesize)} }
    tasks = @tasks.lock(&.all)
    task_bytes = tasks.sum(0_i64) { |t| (t.result.try(&.bytesize) || 0) + (t.description.try(&.bytesize) || 0) }
    Footprint.new(messages, history_bytes, tasks.size, task_bytes)
  end

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

  # Background tasks (see `Agent#background`): those running, and those
  # finished whose outcome the model has not seen yet. One lock for both,
  # so a check-and-move between them is atomic.
  private class TaskState
    getter running = [] of Task
    getter finished = [] of Task

    def all : Array(Task)
      running + finished
    end

    def find(id : String) : Task?
      all.find { |t| t.id == id }
    end
  end

  @tasks : Sync::Exclusive(TaskState)
  @task_callbacks : Sync::Exclusive(Array(Task ->))
  # Named kinds of background work, relaunched after a restart.
  @task_kinds : Sync::Exclusive(Hash(String, (Task -> String)))
  # The tool call being executed, so a task it starts records its name.
  # Agent fiber only.
  @current_tool : String?

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
    @tasks = Sync::Exclusive(TaskState).new(TaskState.new)
    @task_callbacks = Sync::Exclusive(Array(Task ->)).new([] of Task ->)
    @task_kinds = Sync::Exclusive(Hash(String, (Task -> String))).new({} of String => (Task -> String))
    @current_tool = nil
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
    cancel_tasks
  end

  # Run `block` in its own fiber and return at once, so a tool can start slow
  # work (a condition to poll, a long command) and answer the model with
  # `Task#accepted` instead of blocking the turn. The block gets the `Task`,
  # for `Task#sleep` and `Task#cancelled?`, and returns the text the model
  # will see. When it returns, raises or hits its timeout, the outcome is
  # queued and delivered to the model as a `<task_result>` message at the
  # start of the next model call: the next `#ask` or `#resume`, or the next
  # iteration of a tool loop still running. `#on_task_finished` says when
  # that is worth doing.
  #
  # Only an `Agent::ToolError`'s message reaches the model; any other
  # exception is logged and reported as "the task failed". A `Handler` sees
  # the exception on `TaskContext#error` as the outcome is delivered and can
  # set `TaskContext#result` to tell the model why. The default timeout is
  # `Config#background_task_timeout`. A task cannot be killed:
  # after its timeout or a cancel, the block should see `Task#cancelled?`
  # and stop.
  #
  # Raises `TaskLimitError` when `Config#max_background_tasks` tasks are
  # running, and `TaskIdError` for an id in use; from a tool callback both
  # become an error result for the model.
  #
  # `id` names the task instead of `Config#task_id_generator` (or the
  # default); it must not be that of a task running or awaiting delivery.
  #
  # ```
  # agent.register_tool("wait_for_build", "Wait for a build to finish", parameters: schema) do |args|
  #   id = args["id"].as_s
  #   task = agent.background("build #{id} to finish", timeout: 30.minutes) do |t|
  #     until build_done?(id)
  #       break unless t.sleep(10.seconds)
  #     end
  #     build_done?(id) ? build_summary(id) : "build still running"
  #   end
  #   task.accepted
  # end
  # ```
  def background(description : String? = nil, timeout : Time::Span? = nil, id : String? = nil, &block : Task -> String) : Task
    start_task(description, timeout, id, nil, nil, block)
  end

  # Start a task the application finishes: no block runs, and the outcome
  # comes later through `Task#complete` or `Task#fail` (or
  # `#complete_task`/`#fail_task` by id), from any fiber. For results
  # produced outside the agent: a webhook, a scheduler checking state, a
  # person. Until then it counts against the limit and the timeout, and
  # `#cancel_task` and `Task#on_cancel` work as usual. One still running
  # when dumped stays running after `#load`, with the time it had left.
  #
  # ```
  # task = agent.background("deploy 42 to settle", timeout: 45.minutes)
  # # ... later, from the checker:
  # agent.complete_task(task.id, "deploy 42 is serving on 3 nodes")
  # ```
  def background(description : String? = nil, timeout : Time::Span? = nil, id : String? = nil) : Task
    start_task(description, timeout, id, nil, nil, nil)
  end

  # `Task#complete` on the running task with this id. Returns false when
  # none has it.
  def complete_task(id : String, result : String) : Bool
    @tasks.lock { |t| t.running.find { |x| x.id == id } }.try(&.complete(result)) || false
  end

  # `Task#fail` on the running task with this id. Returns false when none
  # has it.
  def fail_task(id : String, reason : String, error : Exception? = nil) : Bool
    @tasks.lock { |t| t.running.find { |x| x.id == id } }.try(&.fail(reason, error)) || false
  end

  # Queue the outcome of a task the application ran and finished entirely
  # on its own, for the next `#ask` or `#resume`: `Task.new(...)` then
  # `Task#complete` or `#fail`, then this. For an application that keeps
  # its own record of pending work and loads a conversation only to run a
  # turn, so no agent was there when the outcome arrived. The model reads
  # the same `<task_result>` as for any task. `#on_task_finished` callbacks
  # are not called: the caller knows.
  #
  # Raises `ArgumentError` for a task still running or cancelled, and
  # `TaskIdError` for an id in use.
  #
  # ```
  # task = Agent::Task.new(45.minutes, tool: "rollback", description: "rollback of 42 to settle",
  #   id: row.task_id, started_at: row.started_at)
  # task.fail("the rollback did not take: 2 containers still Down")
  # agent.queue_task(task)
  # agent.resume
  # ```
  def queue_task(task : Task) : Nil
    raise ArgumentError.new("queue_task takes a finished task; #{task.id} is still running") if task.running?
    raise ArgumentError.new("a cancelled task has no outcome to queue") if task.status.cancelled?
    @tasks.lock do |t|
      raise TaskIdError.new("task id #{task.id} is already in use") if t.find(task.id)
      t.finished << task
    end
  end

  # Start a task of a registered kind (see `#register_task_kind`) with the
  # data its block needs. Unlike a task given a block, it survives a
  # restart: `#load` relaunches it from `kind` and `data` when the kind is
  # registered. Raises `ArgumentError` for a kind that is not.
  #
  # ```
  # agent.register_task_kind("wait_for_build") do |task|
  #   id = task.data.not_nil!["id"].as_s
  #   ...
  # end
  # agent.background("build 42 to finish", kind: "wait_for_build", data: JSON::Any.new({"id" => JSON::Any.new("42")}))
  # ```
  def background(description : String? = nil, timeout : Time::Span? = nil, id : String? = nil, *, kind : String, data : JSON::Any? = nil) : Task
    block = @task_kinds.lock(&.[kind]?) || raise ArgumentError.new("no task kind '#{kind}' is registered")
    start_task(description, timeout, id, kind, data, block)
  end

  # Name a kind of background work, so tasks of it can be started with
  # `#background(kind:, data:)` and relaunched after a restart. The block
  # gets the `Task`, whose `data` tells it what to do. Register kinds
  # before `#load`, like tools.
  def register_task_kind(kind : String, &block : Task -> String) : Nil
    raise ArgumentError.new("task kind must not be empty") if kind.empty?
    @task_kinds.lock { |k| k[kind] = block }
  end

  private def start_task(description : String?, timeout : Time::Span?, id : String?, kind : String?, data : JSON::Any?, block : (Task -> String)?) : Task
    @state_mutex.synchronize { raise ClosedError.new if @closed }
    span = timeout || @config.background_task_timeout
    raise ArgumentError.new("timeout must be positive, got #{span}") unless span > Time::Span.zero
    id ||= @config.task_id_generator.try(&.call) || Task.generate_id
    raise TaskIdError.new("task id must not be empty") if id.empty?

    tool = Fiber.current == @fiber ? @current_tool : nil
    task = Task.new(span, tool: tool, description: description, id: id, kind: kind, data: data, external: block.nil?)
    task.on_finish = ->task_finished(Task)
    @tasks.lock do |t|
      raise TaskIdError.new("task id #{id} is already in use") if t.find(id)
      raise TaskLimitError.new(@config.max_background_tasks) if t.running.size >= @config.max_background_tasks
      t.running << task
    end
    run_task(task, block)
    task
  end

  # Two fibers per task: the worker runs the block; the watcher finishes
  # the task on the outcome, the timeout or a cancel, whichever is first.
  # Run `block` (if any) in a fiber of its own, and watch the task for its
  # outcome, the timeout (`remaining` of it) or an end from outside: a
  # cancel, or `Task#complete`/`#fail`, both of which close the cancel
  # channel.
  private def run_task(task : Task, block : (Task -> String)?, remaining : Time::Span = task.timeout) : Nil
    done = Channel({Task::Status, String?, Exception?}).new(1)
    if block
      spawn(name: "agent task #{task.id}") do
        outcome = begin
          {Task::Status::Done, block.call(task), nil}
        rescue ex : ToolError
          {Task::Status::Failed, ex.message, ex}
        rescue ex
          Log.error(exception: ex) { "Agent: background task #{task.id} raised" }
          {Task::Status::Failed, "the task failed", ex}
        end
        done.send(outcome)
      end
    end
    spawn(name: "agent task watch #{task.id}") do
      select
      when outcome = done.receive
        task.finish(outcome[0], outcome[1], outcome[2])
      when timeout(remaining)
        task.finish(Task::Status::TimedOut, "timed out after #{Task.humanize(task.timeout)}")
      when task.cancel_signal.receive?
        # ended from outside: already finished
      end
      task_finished(task)
    end
  end

  # Background tasks still running.
  def running_tasks : Array(Task)
    @tasks.lock(&.running.dup)
  end

  # Cancel the running task with this id (see `Task#cancel`); it leaves
  # `#running_tasks` at once. Returns false when no running task has it.
  # For a tool that lets the model stop a task it started.
  def cancel_task(id : String) : Bool
    task = @tasks.lock { |t| t.running.find { |x| x.id == id }.try { |x| t.running.delete(x) } } || return false
    task.cancel
    true
  end

  # Drop a finished task's outcome before the model sees it. Returns false
  # when no finished task has the id.
  def discard_task(id : String) : Bool
    !@tasks.lock { |t| t.finished.find { |x| x.id == id }.try { |x| t.finished.delete(x) } }.nil?
  end

  # Background tasks that ended whose outcome the model has not seen yet.
  # `#resume` delivers them; so does the next `#ask`.
  def finished_tasks : Array(Task)
    @tasks.lock(&.finished.dup)
  end

  # Whether the conversation is waiting on background work: tasks running,
  # or outcomes not yet delivered. A scheduler can park the conversation
  # while this is true and come back on `#on_task_finished`.
  def awaiting_tasks? : Bool
    @tasks.lock { |t| !t.running.empty? || !t.finished.empty? }
  end

  # Call `block` from the task's fiber whenever a background task ends with
  # an outcome for the model: done, failed or timed out, not cancelled. The
  # usual follow-up is `#resume`. Returns the callback, for
  # `#off_task_finished`.
  def on_task_finished(&block : Task ->) : Task ->
    @task_callbacks.lock(&.<< block)
    block
  end

  # Remove a callback given to `#on_task_finished`. Returns false when it
  # was not registered.
  def off_task_finished(callback : Task ->) : Bool
    !@task_callbacks.lock(&.delete(callback)).nil?
  end

  # Send the model the outcomes of finished background tasks, with no new
  # user message, and return a Response immediately. Safe to call on a
  # hunch: when nothing is waiting (a turn in progress delivered it, or
  # nothing has finished) no request is made and the response finishes at
  # once with the model's latest reply, `Response#from_history?` set and
  # no usage; with no reply to repeat it finishes with an `Agent::Error`.
  #
  # Raises Agent::ClosedError if the agent has been closed via #close.
  def resume(tools : Array(Tool)? = nil, reasoning_effort : ReasoningEffort? = nil) : Response
    @state_mutex.synchronize { raise ClosedError.new if @closed }
    refuse_from_agent_fiber("resume")

    response = Response.new
    @request_channel.send(AskRequest.new([] of Message, tools, response, RequestOptions.new(reasoning_effort: reasoning_effort)))
    response
  rescue Channel::ClosedError
    raise ClosedError.new
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
  # `source` records who wrote the message (see `Message::Source`): a
  # person by default, or the application feeding the model an event.
  # `Task` is the agent's own; passing it raises `ArgumentError`.
  #
  # Outcomes of finished background tasks (`#background`) are delivered
  # ahead of the message.
  #
  # Raises Agent::ClosedError if the agent has been closed via #close.
  def ask(content : String, attachments : Array(String)? = nil, tools : Array(Tool)? = nil, reasoning_effort : ReasoningEffort? = nil, source : Message::Source = Message::Source::Human) : Response
    @state_mutex.synchronize { raise ClosedError.new if @closed }
    refuse_from_agent_fiber("ask")
    raise ArgumentError.new("source :task is reserved for background task results") if source.task?

    msg = if atts = attachments
            parts = [ContentPart.new(type: :text, text: content)] + atts.map { |path| ContentPart.from_path(path) }
            Message.new(role: Role::User, content_parts: parts, source: source)
          else
            Message.new(role: Role::User, content: content, source: source)
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
    refuse_from_agent_fiber("ask")

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
    tasks = parse_tasks(parsed)

    @session_id = id
    @cache_key = key

    @history.lock { |h| h.clear; h.concat(history) }
    adopt_tasks(tasks)

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

  # Background tasks from a dump, one per id. A cancelled one is nothing
  # the model should hear about and is dropped.
  private def parse_tasks(parsed : Hash(String, JSON::Any)) : Array(Task)
    if tasks = parsed["tasks"]?
      tasks.as_a.map { |t| Task.from_dump(t) }.reject(&.status.cancelled?).uniq!(&.id)
    else
      [] of Task
    end
  rescue ex
    raise SessionLoadError.new("'tasks' field is malformed", cause: ex)
  end

  # Make the dumped tasks this agent's. A task the dump had running is kept
  # if this agent is still running it (the dump was its own snapshot);
  # relaunched if its kind is registered; kept waiting, for the time it had
  # left, if the application finishes it; otherwise its fiber is gone and
  # it is queued as `Lost`, so the model learns the result is never coming.
  # Running tasks the dump does not know are the old conversation's and are
  # cancelled.
  private def adopt_tasks(dumped : Array(Task)) : Nil
    kinds = @task_kinds.lock(&.dup)
    relaunch = [] of {Task, (Task -> String)?, Time::Span}
    cancelled = @tasks.lock do |t|
      alive = t.running.select { |x| dumped.any? { |d| d.running? && d.id == x.id } }
      gone = t.running - alive
      t.running.clear
      t.running.concat(alive)
      t.finished.clear
      dumped.each do |d|
        next if alive.any? { |x| x.id == d.id }
        if d.running? && (block = d.kind.try { |k| kinds[k]? })
          t.running << d
          relaunch << {d, block, d.timeout}
        elsif d.running? && d.external?
          t.running << d
          relaunch << {d, nil, {d.timeout - d.elapsed, Time::Span.zero}.max}
        else
          d.finish(Task::Status::Lost, "this task did not survive a reload of the conversation") if d.running?
          t.finished << d
        end
      end
      gone
    end
    cancelled.each(&.cancel)
    relaunch.each do |(task, block, remaining)|
      task.on_finish = ->task_finished(Task)
      run_task(task, block, remaining)
    end
  end

  # Reset the conversation history back to the system prompt only. Running
  # background tasks are cancelled and undelivered outcomes dropped.
  #
  # NOTE: Calling #reset while an #ask is in-flight is undefined behavior.
  # Wait for any in-flight Response to complete (via #join) first.
  #
  # Raises Agent::ClosedError if the agent has been closed.
  def reset : Nil
    @state_mutex.synchronize { raise ClosedError.new if @closed }
    @history.lock(&.clear)
    cancel_tasks
  end

  # Restore the full conversation history.
  # This replaces the current history with the given messages (e.g. from a
  # previous `#dump`). The system prompt is NOT included — `build_messages`
  # prepends it from Config as usual. Background tasks of the replaced
  # conversation are cancelled.
  #
  # NOTE: Calling #load_history while an #ask is in-flight is undefined behavior.
  # Wait for any in-flight Response to complete (via #join) first.
  #
  # Raises Agent::ClosedError if the agent has been closed.
  def load_history(messages : Array(Message)) : Nil
    @state_mutex.synchronize { raise ClosedError.new if @closed }
    @history.lock { |h| h.clear; h.concat(messages) }
    cancel_tasks
  end

  # Background tasks belong to the conversation: a reset, a load or a close
  # cancels the running ones and drops undelivered outcomes.
  private def cancel_tasks : Nil
    running = @tasks.lock { |t| taken = t.running.dup; t.running.clear; t.finished.clear; taken }
    running.each(&.cancel)
  end

  # `#ask` and `#resume` wait for the agent fiber to take the request; from
  # that fiber (a tool callback, a handler) they would wait for themselves.
  private def refuse_from_agent_fiber(method : String) : Nil
    return unless Fiber.current == @fiber
    raise ArgumentError.new("Agent##{method} must not be called from the agent's own fiber (a tool callback or handler): it would wait for itself")
  end

  # Called from a task's watch fiber once it ended. A task no longer in the
  # running list was taken by a cancel and is not the model's business.
  protected def task_finished(task : Task) : Nil
    queued = @tasks.lock do |t|
      next false unless t.running.delete(task)
      next false if task.status.cancelled?
      t.finished << task
      true
    end
    return unless queued
    @task_callbacks.lock(&.dup).each do |cb|
      cb.call(task)
    rescue ex
      Log.error(exception: ex) { "Agent: on_task_done callback raised" }
    end
  end

  # Serialise the current session fields into an open JSON object.
  # Call this inside a `json.object` block managed by the caller.
  #
  # Thread-safe; acquires @history and @registered_tools internally. A
  # dump taken while a turn is in flight holds that turn's messages as
  # sent so far, task outcomes delivered to it included.
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

    # Running tasks are dumped as they are; `load` marks them lost, since
    # their fibers don't survive.
    unless result.tasks.empty?
      json.field "tasks", result.tasks
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
    tasks = @tasks.lock(&.all)
    DumpResult.new(
      session_id: @session_id,
      cache_key: @cache_key,
      history: history_dup,
      enabled_tools: enabled_names,
      tasks: tasks,
    )
  end

  # Result carrier for session snapshot.
  private record DumpResult,
    session_id : String,
    cache_key : String,
    history : Array(Message),
    enabled_tools : Array(String),
    tasks : Array(Task)

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

    # Task outcomes delivered with this turn's first call; a rolled-back
    # turn puts them back so the next one delivers them again.
    delivered = [] of Task
    incoming = request.new_messages

    begin
      incoming = with_task_results(request.new_messages, delivered)
      if incoming.empty?
        answer_from_history(response)
        return
      end

      # Append new messages to history under the lock, then snapshot for the HTTP call.
      history_snapshot = @history.lock do |h|
        close_dangling_tool_calls(h, incoming)
        start_size = h.size
        h.concat(incoming)
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
          rollback_failed_turn(start_size, incoming.size, delivered)
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
        # Tasks that ended during this turn report after the tool results.
        results = with_task_results(results)

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
      rollback_failed_turn(start_size, incoming.size, delivered) unless response.finished?
      response.finish_with_error(err, total_usage)
    end
  end

  # A resume with nothing to deliver: a turn in progress got there first,
  # or nothing had finished. The model's latest reply answers it without a
  # request; a conversation with no reply (empty, or waiting on the
  # caller's own tool results) gets an error, never a request it did not
  # ask for.
  private def answer_from_history(response : Response) : Nil
    last = @history.lock(&.last?)
    if last && last.role.assistant?
      response.mark_from_history
      response.finish(last, Usage.new)
    else
      response.finish_with_error(Agent::Error.new("nothing to deliver: no finished task, and no reply to repeat"))
    end
  end

  # A turn that fails before the model replied produced nothing: drop the
  # messages it added, and queue the task outcomes among them again. Once a
  # reply is in history, tools may have run, so the turn is kept; any call
  # left unanswered is answered at the start of the next turn.
  private def rollback_failed_turn(start_size : Int32?, added : Int32, delivered : Array(Task)) : Nil
    return unless start_size
    rolled_back = @history.lock do |h|
      next false unless start_size < h.size <= start_size + added
      h.pop(h.size - start_size)
      true
    end
    return if !rolled_back || delivered.empty?
    @tasks.lock do |t|
      later = t.finished.dup
      t.finished.clear
      t.finished.concat(delivered).concat(later)
    end
  end

  # `incoming` with a `<task_result>` message per finished background task
  # added. A batch with tool results gets them only when
  # `Config#deliver_tasks_mid_turn?` allows, after the last tool result:
  # those must directly follow the assistant message that called them, and
  # some backends refuse a user message right after them. Delivered tasks
  # are appended to `delivered`.
  private def with_task_results(incoming : Array(Message), delivered : Array(Task)? = nil) : Array(Message)
    return incoming if incoming.any?(&.role.tool?) && !@config.deliver_tasks_mid_turn?
    finished = @tasks.lock(&.finished.dup)
    return incoming if finished.empty?

    messages = finished.map { |task| task_message(task) }
    @tasks.lock(&.finished.reject! { |t| finished.includes?(t) })
    delivered.try(&.concat(finished))

    at = (incoming.rindex(&.role.tool?) || -1) + 1
    incoming[0, at] + messages + incoming[at..]
  end

  # The task's outcome through the `TaskContext` handlers. A handler that
  # raises is logged and the outcome delivered as it is: one bad handler
  # must not block the conversation.
  private def task_message(task : Task) : Message
    ctx = TaskContext.new(task, @config.max_task_result_chars)
    begin
      @handlers.decorate(ctx) { |c| c.task.to_message(c.result, c.warnings) }
    rescue ex
      Log.error(exception: ex) { "Agent: TaskContext handler raised for task #{task.id}; delivering the outcome as it is" }
      fallback = TaskContext.new(task, @config.max_task_result_chars)
      task.to_message(fallback.result, fallback.warnings)
    end
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
  # A background task's result belongs to the turn it arrived in: ahead of
  # the next user message, or after the tool results of a turn in progress.
  private def trim_history! : Nil
    @history.replace do |h|
      max = @config.max_history
      next h unless max && max > 0

      starts = turn_starts(h)
      next h if starts.size <= max

      first_kept = starts[starts.size - max]
      h[first_kept..]
    end
  end

  # Where each turn begins: a run of consecutive user messages starts a
  # turn at its first message, unless the run follows tool results, where
  # the task results at its head still belong to the turn in progress and
  # the first other message starts the new one.
  private def turn_starts(h : Array(Message)) : Array(Int32)
    starts = [] of Int32
    idx = 0
    while idx < h.size
      unless h[idx].role.user?
        idx += 1
        next
      end
      run_end = idx
      while run_end < h.size && h[run_end].role.user?
        run_end += 1
      end
      if idx > 0 && h[idx - 1].role.tool?
        (idx...run_end).find { |i| !h[i].task_result? }.try { |i| starts << i }
      else
        starts << idx
      end
      idx = run_end
    end
    starts
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
                     @current_tool = ctx.tool_call.name
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
                   ensure
                     @current_tool = nil
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
          check_tool_calls_complete(msg, finish_reason)

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

  # Finish reasons that end a reply early. Tool calls in such a reply may
  # have half-written arguments (OpenAI "length", Anthropic "max_tokens"),
  # or have been cut off by a refusal: none of them is run.
  CUT_OFF_REASONS = {"length", "max_tokens", "model_context_window_exceeded", "refusal", "content_filter"}

  private def check_tool_calls_complete(msg : Message, finish_reason : String?) : Nil
    return unless msg.has_tool_calls? && finish_reason.in?(CUT_OFF_REASONS)
    raise StreamError.new("the reply ended early (#{finish_reason}) with tool calls in it; none was run", finish_reason)
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

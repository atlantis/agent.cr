require "json"
require "sync"

class Agent
  # Raised by `Agent#background` when `Config#max_background_tasks` tasks are
  # already running. A `ToolError`, so from a tool callback it becomes an
  # error result the model can act on.
  class TaskLimitError < ToolError
    getter limit : Int32

    def initialize(@limit : Int32)
      if @limit.zero?
        super("background tasks are turned off (max_background_tasks is 0)")
      else
        super("background task limit (#{@limit}) reached; wait for a task to finish")
      end
    end
  end

  # Raised by `Agent#background` for a task id that is empty or already in
  # use, usually a `Config#task_id_generator` that repeats itself. A
  # `ToolError`, so from a tool callback the model sees it too.
  class TaskIdError < ToolError
  end

  # Work a tool started but did not wait for: a condition to poll, a slow
  # command, a timer. The tool answers at once with `#accepted`, the turn
  # ends, and the task runs in its own fiber under `Agent#background`. Its
  # outcome is delivered to the model as a `Message` whose `source` is
  # `Task`, at the start of the conversation's next model call (the next
  # `Agent#ask` or `Agent#resume`, or the next iteration of a tool loop still
  # running). The content is the outcome wrapped in a `<task_result>` tag, so
  # the model can tell it from what a person typed; see `PROMPT_NOTE`.
  class Task
    enum Status
      Running
      # The block returned.
      Done
      # The block raised.
      Failed
      TimedOut
      # Stopped by `#cancel`, `Agent#reset` or `Agent#close`.
      Cancelled
      # The agent was dumped while it ran; the fiber did not survive.
      Lost

      def to_s : String
        case self
        in Running   then "running"
        in Done      then "done"
        in Failed    then "failed"
        in TimedOut  then "timed_out"
        in Cancelled then "cancelled"
        in Lost      then "lost"
        end
      end

      def self.parse(value : String) : Status
        case value
        when "running"   then Running
        when "done"      then Done
        when "failed"    then Failed
        when "timed_out" then TimedOut
        when "cancelled" then Cancelled
        when "lost"      then Lost
        else                  raise ArgumentError.new("Unknown task status: '#{value}'")
        end
      end
    end

    # A sentence for the system prompt of an agent whose tools start tasks.
    PROMPT_NOTE = "Some tools start a background task instead of answering at once; they reply " \
                  "with the task's id. The outcome arrives later as a message wrapped in " \
                  "<task_result id=\"...\"> tags. That message comes from the system, not from " \
                  "the user. Never poll for a result: end your turn and go on when it arrives. " \
                  "A warning attribute means the outcome was shortened or otherwise qualified."

    # Something the model should know about an outcome that is otherwise
    # fine, sent as the tag's `warning` attribute.
    enum Warning
      # The body is the start and end of a longer output; see
      # `Config#max_task_result_chars`.
      Truncated

      def to_s : String
        case self
        in Truncated then "truncated"
        end
      end
    end

    getter id : String
    # The tool call that started the task, when one did.
    getter tool : String?
    # What the task is doing, for the model and for logs.
    getter description : String?
    getter timeout : Time::Span
    getter started_at : Time
    # The registered kind of work this task runs, and the data it needs
    # (see `Agent#register_task_kind`); what a restart relaunches it from.
    # `data` is for the application; the model never sees it.
    getter kind : String?
    getter data : JSON::Any?

    @status = Status::Running
    @result : String? = nil
    @error : Exception? = nil
    @finished_at : Time? = nil
    @cancel = Channel(Nil).new
    @on_cancel = [] of ->
    @mutex = Sync::Mutex.new

    def initialize(@timeout : Time::Span, @tool : String? = nil, @description : String? = nil,
                   @id : String = Task.generate_id, @started_at : Time = Time.utc,
                   @kind : String? = nil, @data : JSON::Any? = nil)
    end

    # Random, 12 hex characters: short enough for the model to quote, and
    # 48 bits, so an application store keyed by them across many agents
    # stays collision-free into the millions.
    def self.generate_id : String
      "t-#{Random::Secure.hex(6)}"
    end

    def status : Status
      @mutex.synchronize { @status }
    end

    # The block's return value, or what went wrong: a `ToolError`'s message,
    # "the task failed" for any other exception, the timeout.
    def result : String?
      @mutex.synchronize { @result }
    end

    # The exception that failed the task, for `TaskContext` handlers and
    # `Agent#on_task_finished` callbacks. Nil unless `status` is `Failed`.
    def error : Exception?
      @mutex.synchronize { @error }
    end

    def finished_at : Time?
      @mutex.synchronize { @finished_at }
    end

    def running? : Bool
      status.running?
    end

    # True once the task was cancelled or timed out: the block should stop.
    def cancelled? : Bool
      status.in?(Status::Cancelled, Status::TimedOut)
    end

    # :nodoc:
    def cancel_signal : Channel(Nil)
      @cancel
    end

    # Stop the task. The block cannot be killed; it sees `#cancelled?` and
    # `#sleep` returning early, and its `#on_cancel` hooks run. Nothing is
    # delivered to the model.
    def cancel : Nil
      finish(Status::Cancelled, "cancelled")
    end

    # Run `hook` when the task is cancelled or times out, so the block can
    # tear down what it is blocked on: kill a process, close a socket. Runs
    # at once, in the caller's fiber, if the task is already stopped;
    # otherwise in a fiber of its own, so a slow hook never holds up the
    # cancel. An exception is logged, not raised.
    #
    # ```
    # agent.background("the test suite", timeout: 20.minutes) do |task|
    #   process = Process.new("make", ["test"], output: out = IO::Memory.new, error: out)
    #   task.on_cancel { process.terminate }
    #   status = process.wait
    #   raise Agent::ToolError.new("make test exited #{status.exit_code}:\n#{out}") unless status.success?
    #   out.to_s
    # end
    # ```
    def on_cancel(&hook : ->) : Nil
      run_now = @mutex.synchronize do
        next true if @status.in?(Status::Cancelled, Status::TimedOut)
        @on_cancel << hook
        false
      end
      run_hook(hook) if run_now
    end

    # Sleep for `span`, or until the task is cancelled or times out. Returns
    # false when cut short, so a polling loop can stop:
    #
    # ```
    # agent.background("build #42 to finish", timeout: 30.minutes) do |task|
    #   until build_done?(42)
    #     break unless task.sleep(10.seconds)
    #   end
    #   build_done?(42) ? build_summary(42) : "build still running"
    # end
    # ```
    def sleep(span : Time::Span) : Bool
      select
      when @cancel.receive?
        false
      when timeout(span)
        !@cancel.closed?
      end
    end

    # Block until the task ends, or `span` passes. Returns whether it ended.
    def wait(span : Time::Span? = nil) : Bool
      if span
        select
        when @cancel.receive? then true
        when timeout(span) then !running?
        end
      else
        @cancel.receive?
        true
      end
    end

    def elapsed : Time::Span
      (finished_at || Time.utc) - @started_at
    end

    def inspect(io : IO) : Nil
      io << "#<Agent::Task " << @id << ' ' << status
      io << " tool=" << @tool if @tool
      io << ' ' << Task.humanize(elapsed) << '>'
    end

    # The tool result to give the model when a tool starts this task.
    def accepted : String
      String.build do |s|
        s << "Started background task " << @id
        s << " (" << @description << ")" if @description
        s << ". It times out after " << Task.humanize(@timeout)
        s << ". Its result will arrive in a later message as <task_result id=\"" << @id << "\">; do not poll for it."
      end
    end

    # The message that tells the model how the task ended. `body` replaces
    # `result` as the text inside the tag and `warnings` qualify it; a
    # `TaskContext` handler sets them.
    def to_message(body : String? = result, warnings : Array(Warning) = [] of Warning) : Message
      Message.new(role: Role::User, content: wrapped_result(body, warnings), source: Message::Source::Task)
    end

    # The body goes in as it is, except that a closing tag inside it (a log
    # tail, a page) is defused so the frame stays whole.
    # :nodoc:
    def wrapped_result(body : String? = result, warnings : Array(Warning) = [] of Warning) : String
      String.build do |s|
        s << "<task_result id=" << @id.inspect
        s << " tool=" << tool.inspect if tool
        s << " description=" << description.inspect if description
        s << " status=" << status.to_s.inspect
        s << " elapsed=" << Task.humanize(elapsed).inspect
        s << " warning=" << warnings.uniq.join(",", &.to_s).inspect unless warnings.empty?
        s << ">\n" << (body || "").gsub("</task_result", "<\\/task_result") << "\n</task_result>"
      end
    end

    # `text` cut to at most `max` characters, as its start and end around a
    # line saying how much is missing, and whether it was cut. The end is
    # kept because that is where a log says how things went.
    # :nodoc:
    def self.truncate(text : String, max : Int32) : {String, Bool}
      return {text, false} if text.size <= max
      # The marker's own length counts against the cap, and its digit count
      # depends on what is kept, so settle the two together.
      keep = max
      marker = ""
      5.times do
        marker = "\n[... #{text.size - keep} of #{text.size} characters omitted ...]\n"
        next_keep = max - marker.size
        break if next_keep == keep
        keep = next_keep
      end
      # A cap too small for the marker keeps the start alone.
      return {text[0, max], true} if keep < 2
      head = keep // 2
      tail = keep - head
      {text[0, head] + marker + text[text.size - tail, tail], true}
    end

    # :nodoc:
    def finish(status : Status, result : String?, error : Exception? = nil) : Bool
      hooks = [] of ->
      changed = @mutex.synchronize do
        next false unless @status.running?
        @status = status
        @result = result
        @error = error
        @finished_at = Time.utc
        hooks, @on_cancel = @on_cancel, hooks if status.in?(Status::Cancelled, Status::TimedOut)
        true
      end
      @cancel.close unless @cancel.closed?
      spawn(name: "agent task #{@id} on_cancel") { hooks.each { |hook| run_hook(hook) } } unless hooks.empty?
      changed
    end

    private def run_hook(hook : ->) : Nil
      hook.call
    rescue ex
      Log.error(exception: ex) { "Agent: on_cancel hook of task #{@id} raised" }
    end

    # :nodoc:
    def self.humanize(span : Time::Span) : String
      return "#{span.total_milliseconds.round.to_i64}ms" if span < 1.second
      total = span.total_seconds.round.to_i64
      return "#{total}s" if total < 60
      minutes, seconds = total.divmod(60)
      return seconds.zero? ? "#{minutes}m" : "#{minutes}m #{seconds}s" if minutes < 60
      hours, minutes = minutes.divmod(60)
      minutes.zero? ? "#{hours}h" : "#{hours}h #{minutes}m"
    end

    # :nodoc:
    def to_json(json : JSON::Builder) : Nil
      json.object do
        json.field "id", @id
        json.field "tool", @tool if @tool
        json.field "description", @description if @description
        json.field "kind", @kind if @kind
        json.field "data", @data if @data
        json.field "timeout_ms", @timeout.total_milliseconds.to_i64
        json.field "started_at", @started_at.to_rfc3339(fraction_digits: 3)
        json.field "status", status.to_s
        json.field "result", result if result
        if at = finished_at
          json.field "finished_at", at.to_rfc3339(fraction_digits: 3)
        end
      end
    end

    # Rebuild a task from a dump. One dumped while running comes back
    # `Running` with no fiber behind it; `Agent#load` decides whether it is
    # still alive in this agent or `Lost`.
    # :nodoc:
    def self.from_dump(data : JSON::Any) : Task
      h = data.as_h
      id = h["id"].as_s
      timeout = h["timeout_ms"].as_i64
      raise ArgumentError.new("task id must not be empty") if id.empty?
      raise ArgumentError.new("task timeout must be positive") unless timeout > 0
      task = new(
        timeout.milliseconds,
        h["tool"]?.try(&.as_s?),
        h["description"]?.try(&.as_s?),
        id,
        Time.parse_rfc3339(h["started_at"].as_s),
        h["kind"]?.try(&.as_s?),
        h["data"]?,
      )
      status = Status.parse(h["status"].as_s)
      unless status.running?
        task.finish(status, h["result"]?.try(&.as_s?))
        if at = h["finished_at"]?.try(&.as_s?)
          task.restore_finished_at(Time.parse_rfc3339(at))
        end
      end
      task
    end

    # :nodoc:
    def restore_finished_at(at : Time) : Nil
      @mutex.synchronize { @finished_at = at }
    end
  end
end

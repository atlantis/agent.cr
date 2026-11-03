class Agent
  # Mutable context for the full turn (request → LLM response).
  # Handlers may mutate messages/tools before the HTTP request is built.
  # (Post-processing of the returned {Message, Usage, String?} tuple is not
  # supported by the current shape; the block returns the tuple directly.)
  class TurnContext
    property messages : Array(Message)
    property tools : Array(Tool)?
    # Per-request settings of the #ask; a handler may replace them for this
    # provider call (e.g. a different reasoning effort per lane).
    property options : RequestOptions

    def initialize(@messages, @tools, @options = RequestOptions.new)
    end
  end

  # Mutable context for a single streaming chunk.
  # Handlers should reassign `ctx.chunk = Chunk.new(...)` rather than
  # attempting to mutate fields in place.
  class ChunkContext
    property chunk : Response::Chunk

    def initialize(@chunk)
    end
  end

  # Mutable context for a single tool call, before execution.
  # `response` is provided so a handler can cancel the in-flight request.
  # After `next_proc` returns, `error` holds the exception the callback
  # raised, if any: a handler can record it, or match on an `Agent::ToolError`
  # subclass and return a different result for the model.
  class ToolCallContext
    property tool_call : ToolCall
    property response : Response
    property error : Exception?

    def initialize(@tool_call, @response)
    end
  end

  # Mutable context for a background task's outcome, as it is about to be
  # delivered to the model. `result` is the text the model will read inside
  # the `<task_result>` tag: `Task#result`, cut to
  # `Config#max_task_result_chars` with `Task::Warning::Truncated` in
  # `warnings` when it was longer. A handler can replace it, say to explain
  # a failure from `error` (the exception that failed the task; an
  # `Agent::ToolError`'s message is already there) or to shorten it
  # differently, and add warnings of its own. The status attributes come
  # from the task and are not changed by it.
  class TaskContext
    getter task : Task
    property result : String?
    property warnings : Array(Task::Warning)

    def initialize(@task, max_chars : Int32? = nil)
      @result = @task.result
      @warnings = [] of Task::Warning
      if max_chars && (text = @result)
        @result, cut = Task.truncate(text, max_chars)
        @warnings << Task::Warning::Truncated if cut
      end
    end

    def error : Exception?
      @task.error
    end
  end

  # Mutable context for an exception that escaped the pipeline.
  # Handlers may wrap or replace the error before it's recorded.
  # `error` is typed `Exception` because the generic rescue branch can land
  # any exception here; handler blocks must produce an `Agent::Error`.
  class ErrorContext
    property error : Exception

    def initialize(@error)
    end
  end

  # Base class for a handler in the chain.
  #
  # Each `handle` overload receives the context for one stage of the pipeline
  # and a `next` proc that invokes the rest of the chain. The default
  # implementation is a pass-through: it simply calls `next.call(ctx)` and
  # returns the result. Subclasses override the overloads they care about and
  # may inspect/mutate the context before and/or after yielding to `next`.
  #
  # A handler that does not want to forward to the rest of the chain can
  # short-circuit by returning a value without calling `next`.
  class Handler
    def handle(ctx : TurnContext, next_proc : TurnContext -> {Message, Usage, String?}) : {Message, Usage, String?}
      next_proc.call(ctx)
    end

    def handle(ctx : ChunkContext, next_proc : ChunkContext -> Response::Chunk) : Response::Chunk
      next_proc.call(ctx)
    end

    def handle(ctx : ToolCallContext, next_proc : ToolCallContext -> Message) : Message
      next_proc.call(ctx)
    end

    def handle(ctx : TaskContext, next_proc : TaskContext -> Message) : Message
      next_proc.call(ctx)
    end

    def handle(ctx : ErrorContext, next_proc : ErrorContext -> Agent::Error) : Agent::Error
      next_proc.call(ctx)
    end
  end

  # Runs a chain of `Handler`s around a block.
  #
  # `decorate` builds a nested chain of `handle` calls ending in the supplied
  # block, then invokes the outermost handler. Each handler can yield to the
  # next by calling `next.call(ctx)`. The chain is iterated left-to-right in
  # the order handlers were added via `Agent#use`.
  class ChainHandler
    @chain : Array(Handler)

    def initialize
      @chain = [] of Handler
    end

    def <<(handler : Handler) : self
      @chain << handler
      self
    end

    def empty? : Bool
      @chain.empty?
    end

    def decorate(context : TurnContext, &block : TurnContext -> {Message, Usage, String?}) : {Message, Usage, String?}
      build_turn(@chain.each, block).call(context)
    end

    def decorate(context : ChunkContext, &block : ChunkContext -> Response::Chunk) : Response::Chunk
      build_chunk(@chain.each, block).call(context)
    end

    def decorate(context : ToolCallContext, &block : ToolCallContext -> Message) : Message
      build_tool_call(@chain.each, block).call(context)
    end

    def decorate(context : TaskContext, &block : TaskContext -> Message) : Message
      build_task(@chain.each, block).call(context)
    end

    def decorate(context : ErrorContext, &block : ErrorContext -> Agent::Error) : Agent::Error
      build_error(@chain.each, block).call(context)
    end

    # Each `build_*` walks the handler iterator and returns a proc that, when
    # called with a context, invokes the current handler's `handle` with a
    # `next` proc continuing the chain. When the iterator is exhausted the
    # leaf block is the next call.
    private def build_turn(iter, leaf : TurnContext -> {Message, Usage, String?}) : TurnContext -> {Message, Usage, String?}
      case current = iter.next
      when Iterator::Stop
        leaf
      else
        rest = build_turn(iter, leaf)
        ->(ctx : TurnContext) { current.handle(ctx, rest) }
      end
    end

    private def build_chunk(iter, leaf : ChunkContext -> Response::Chunk) : ChunkContext -> Response::Chunk
      case current = iter.next
      when Iterator::Stop
        leaf
      else
        rest = build_chunk(iter, leaf)
        ->(ctx : ChunkContext) { current.handle(ctx, rest) }
      end
    end

    private def build_tool_call(iter, leaf : ToolCallContext -> Message) : ToolCallContext -> Message
      case current = iter.next
      when Iterator::Stop
        leaf
      else
        rest = build_tool_call(iter, leaf)
        ->(ctx : ToolCallContext) { current.handle(ctx, rest) }
      end
    end

    private def build_task(iter, leaf : TaskContext -> Message) : TaskContext -> Message
      case current = iter.next
      when Iterator::Stop
        leaf
      else
        rest = build_task(iter, leaf)
        ->(ctx : TaskContext) { current.handle(ctx, rest) }
      end
    end

    private def build_error(iter, leaf : ErrorContext -> Agent::Error) : ErrorContext -> Agent::Error
      case current = iter.next
      when Iterator::Stop
        leaf
      else
        rest = build_error(iter, leaf)
        ->(ctx : ErrorContext) { current.handle(ctx, rest) }
      end
    end
  end
end

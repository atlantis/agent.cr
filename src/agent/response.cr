require "sync"

class Agent
  # A handle on the pending response from #ask.
  # Wraps the ongoing fiber-based HTTP streaming and provides
  # streaming chunks, final message, metadata, and wait/join operations.
  #
  # Synchronization:
  #   - `chunk_channel` (size 256): legitimate stream queue with backpressure.
  #   - `message_channel` (size 1) and `usage_channel` (size 1): legitimate
  #     one-shot futures for the final message and usage metadata.
  #   - Shared flags (@done, @cancelled, @stream_state, @finish_reason, @error)
  #     are protected by a Mutex. The old `cancel_channel` (unconsumed) has
  #     been removed in favor of a mutex-protected bool.
  class Response
    CHUNK_BUFFER = 256

    private enum StreamState
      Waiting
      Streaming
      Left
    end

    # Tags a streamed chunk with its origin.
    enum ChunkKind
      Content
      Reasoning
      ToolCallArgs
      ToolCallName
    end

    # A streamed text chunk tagged with its origin.
    #
    # This is a class (not a struct) so that handler chains can replace the
    # chunk object on `ChunkContext` and have the mutation be visible to the
    # caller. Handlers should reassign `ctx.chunk = Chunk.new(...)` rather
    # than attempting to mutate fields in place.
    class Chunk
      property text : String
      property kind : ChunkKind

      def initialize(@text : String, @kind : ChunkKind = ChunkKind::Content)
      end

      def content? : Bool
        @kind == ChunkKind::Content
      end

      def reasoning? : Bool
        @kind == ChunkKind::Reasoning
      end

      def tool_call_name? : Bool
        @kind == ChunkKind::ToolCallName
      end

      def tool_call_args? : Bool
        @kind == ChunkKind::ToolCallArgs
      end
    end

    # --- channels used internally by the agent ---

    # The fiber pushes streamed Chunks here.
    private getter chunk_channel : Channel(Chunk)
    # Signals the final Message once fully built.
    private getter message_channel : Channel(Message)
    # Signals the final Usage metadata.
    private getter usage_channel : Channel(Usage)

    # Optional callback: transform each chunk before it reaches the stream consumer.
    # Set by Agent to wire the on_chunk handler. Returns the (possibly mutated) chunk.
    #
    # Cleared by #finish and #finish_with_error so the proc does not retain a
    # closure reference to the Agent after the response completes. Callers
    # should not reassign this once the response is in flight.
    property chunk_handler : (Chunk -> Chunk)?

    # --- mutex-protected shared state ---

    @mutex : Sync::Mutex
    @done = false
    @cancelled = false
    # Waiting: nobody has called #stream yet, so chunks queue in @backlog.
    # Streaming: chunks go to the channel. Left: the consumer left early,
    # so chunks are dropped.
    @stream_state = StreamState::Waiting
    # Consecutive chunks of a kind are joined, so a long reply waiting for
    # #stream costs its text, not an object per token.
    @backlog = [] of {ChunkKind, String::Builder}
    @chunks_pushed = 0_i64
    @finish_reason : String?
    @error : Agent::Error?
    @chunk_handler : (Chunk -> Chunk)?

    # Cached values so the second call to #message / #metadata
    # does not block on the channel again.
    @message : Message?
    @metadata : Usage?

    def initialize
      @chunk_channel = Channel(Chunk).new(CHUNK_BUFFER)
      @message_channel = Channel(Message).new(1)
      @usage_channel = Channel(Usage).new(1)
      @chunk_handler = nil
      @mutex = Sync::Mutex.new
    end

    # Shorthand for @mutex.synchronize — used by all flag accessors.
    private def synchronize(&)
      @mutex.synchronize { yield }
    end

    # Request cancellation of this in-flight response.
    # The HTTP fiber will abort after the current SSE line is processed.
    # Safe to call from any fiber. Idempotent.
    # After cancellation, #message returns a "Agent error: cancelled" message
    # and #error returns a CancelledError.
    def cancel : Nil
      synchronize { @cancelled = true }
    end

    # Returns true if cancel was requested.
    def cancelled? : Bool
      synchronize { @cancelled }
    end

    # Returns true when the response has finished assembling.
    def finished? : Bool
      synchronize { @done }
    end

    # The reason the stream finished ("stop", "length", "tool_calls", etc.)
    # or nil if not known / the request errored.
    def finish_reason : String?
      synchronize { @finish_reason }
    end

    # Returns the error if this response represents a failed request, or nil.
    def error : Agent::Error?
      synchronize { @error }
    end

    # Returns true if this response represents a failed request.
    def error? : Bool
      synchronize { !@error.nil? }
    end

    # True once any chunk (content, reasoning, tool call) has been produced
    # for this response, by any model call of the turn.
    def output_started? : Bool
      chunks_pushed > 0
    end

    # True when the response failed or was cancelled after output started:
    # what was streamed is an incomplete reply, which a UI should mark.
    def partial? : Bool
      synchronize { !@error.nil? && @chunks_pushed > 0 }
    end

    # Internal: chunks produced so far, across every model call of the turn.
    def chunks_pushed : Int64
      synchronize { @chunks_pushed }
    end

    # Return the fully assembled message (blocks until ready).
    def message : Message
      @message ||= begin
        msg = @message_channel.receive
        synchronize { @message = msg }
        msg
      end
    rescue Channel::ClosedError
      @message || raise("BUG: message channel closed without delivering a message")
    end

    # Return the metadata / usage (blocks until ready).
    def metadata : Usage
      @metadata ||= begin
        meta = @usage_channel.receive
        synchronize { @metadata = meta }
        meta
      end
    rescue Channel::ClosedError
      @metadata || Usage.new
    end

    # Wait for the response to complete (both message and usage).
    def join : Nil
      message
      metadata
    end

    # Yields each chunk as it arrives from the API.
    # Use `chunk.text` for the string and `chunk.kind` for its origin.
    def stream(& : Chunk -> _) : Nil
      backlog = synchronize do
        @stream_state = StreamState::Streaming
        @backlog.tap { @backlog = [] of {ChunkKind, String::Builder} }
      end
      backlog.each { |kind, text| yield Chunk.new(text.to_s, kind) }
      loop do
        chunk = @chunk_channel.receive
        yield chunk
      end
    rescue Channel::ClosedError
      # streaming finished
    ensure
      stop_streaming
    end

    # A consumer that leaves #stream early (break, or an exception such as a
    # disconnected client) must not leave the agent fiber blocked on a full
    # buffer: stop accepting chunks and drain what is queued.
    private def stop_streaming : Nil
      synchronize { @stream_state = StreamState::Left }
      loop do
        select
        when chunk = @chunk_channel.receive?
          break if chunk.nil?
        else
          break
        end
      end
    end

    # A tool call name stays its own chunk: two calls' names must not join.
    private def queue(chunk : Chunk) : Nil
      last = @backlog.last?
      if last && last[0] == chunk.kind && !chunk.kind.tool_call_name?
        last[1] << chunk.text
      else
        @backlog << {chunk.kind, String::Builder.new.tap(&.<<(chunk.text))}
      end
    end

    # Internal: push a chunk (safe to call from any fiber).
    #
    # Until #stream is called, chunks queue in memory without blocking, so
    # a caller that starts streaming late still gets every chunk, and one
    # that only calls #message never blocks the agent fiber. While #stream
    # runs, chunks go through the 256-slot channel and the fiber blocks if
    # the consumer falls behind. After the consumer leaves, they are dropped.
    def push_chunk(chunk : Chunk) : Nil
      if handler = @chunk_handler
        chunk = handler.call(chunk)
      end

      queued = synchronize do
        @chunks_pushed += 1
        case @stream_state
        in .waiting?   then queue(chunk); true
        in .left?      then true
        in .streaming? then false
        end
      end
      @chunk_channel.send(chunk) unless queued
    rescue Channel::ClosedError
      # already finished
    end

    # Internal: signal that the response is complete.
    # Safe to call multiple times — subsequent calls are no-ops.
    def finish(message : Message, usage : Usage, finish_reason : String? = nil) : Nil
      synchronize do
        return if @done
        @done = true
        @finish_reason = finish_reason
        @message = message
        @metadata = usage
      end
      @message_channel.send(message)
      @usage_channel.send(usage)
    ensure
      @chunk_handler = nil
      @chunk_channel.close
    end

    # Internal: signal that the response failed with an error.
    # Stores the error and sends an error message through the normal channels
    # so consumers can still call #message. `usage` is what the turn's
    # provider calls consumed before the failure.
    def finish_with_error(err : Agent::Error, usage : Usage = Usage.new) : Nil
      error_msg = Message.new(role: Role::Assistant, content: "Agent error: #{err.message}")

      synchronize do
        return if @done
        @done = true
        @error = err
        @message = error_msg
        @metadata = usage
      end
      @message_channel.send(error_msg)
      @usage_channel.send(usage)
    ensure
      @chunk_handler = nil
      @chunk_channel.close
    end
  end
end

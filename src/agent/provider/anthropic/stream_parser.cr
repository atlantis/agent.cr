require "json"

class Agent
  module Provider
    class Anthropic
      # One content block of a streamed reply, rebuilt from its deltas.
      private class Block
        # Blocks whose input streams as `input_json_delta`.
        INPUT_TYPES = {"tool_use", "server_tool_use", "mcp_tool_use"}

        getter type : String
        # The block as `content_block_start` sent it; other types (server
        # tool results, ...) arrive whole and go back as they came.
        getter start : Hash(String, JSON::Any)
        getter text = IO::Memory.new
        getter json = IO::Memory.new
        getter signature = IO::Memory.new
        getter citations = [] of JSON::Any

        def initialize(@start)
          @type = @start["type"]?.try(&.as_s?) || "text"
          @start["text"]?.try(&.as_s?).try { |t| @text << t }
          @start["thinking"]?.try(&.as_s?).try { |t| @text << t }
          @start["signature"]?.try(&.as_s?).try { |s| @signature << s }
        end

        def id : String
          @start["id"]?.try(&.as_s?) || ""
        end

        def name : String
          @start["name"]?.try(&.as_s?) || ""
        end

        def arguments : String
          @json.to_s.presence || "{}"
        end

        def to_any : JSON::Any
          case @type
          when "text"
            h = {"type" => JSON::Any.new("text"), "text" => JSON::Any.new(@text.to_s)}
            h["citations"] = JSON::Any.new(@citations) unless @citations.empty?
            JSON::Any.new(h)
          when "thinking"
            JSON::Any.new({
              "type"      => JSON::Any.new("thinking"),
              "thinking"  => JSON::Any.new(@text.to_s),
              "signature" => JSON::Any.new(@signature.to_s),
            })
          when .in?(INPUT_TYPES)
            h = @start.dup
            h["input"] = RequestBody.parse_input(arguments)
            JSON::Any.new(h)
          else # redacted_thinking (its data), server tool results, ...
            JSON::Any.new(@start)
          end
        end
      end

      # What a stream has said so far.
      private class StreamState
        getter blocks = {} of Int32 => Block
        getter usage = {} of String => JSON::Any
        property stop_reason : String?
        property? stopped = false

        def ordered_blocks : Array(Block)
          @blocks.keys.sort!.map { |i| @blocks[i] }
        end
      end

      # Parses a Messages API event stream.
      module StreamParser
        extend self

        def parse(
          io : IO,
          response : Response,
          cancel : -> Bool,
          *,
          first_byte_timeout : Time::Span? = nil,
          idle_timeout : Time::Span? = nil,
          endpoint : String? = nil,
        ) : {Message, Usage, String?}
          state = StreamState.new
          reads = 0
          first = first_byte_timeout || idle_timeout
          read_timeout(io, first)

          loop do
            break if cancel.call
            line = begin
              io.gets(chomp: true)
            rescue ex : IO::TimeoutError
              raise IdleTimeoutError.new(reads.zero? ? :first_byte : :idle, reads.zero? ? first : idle_timeout, cause: ex)
            end
            break if line.nil?
            reads += 1
            read_timeout(io, idle_timeout) if idle_timeout && idle_timeout != first

            parse_event(line).try { |event| handle_event(event, state, response) }
          end

          # A stream cut off before message_stop is not a reply, however
          # much of one came.
          unless state.stopped? || cancel.call
            raise StreamError.new("stream ended before the model finished", retryable: true)
          end
          {message(state.ordered_blocks, endpoint), usage(state.usage), state.stop_reason}
        end

        private def handle_event(event : Hash(String, JSON::Any), state : StreamState, response : Response) : Nil
          case event["type"]?.try(&.as_s?)
          when "message_start"
            event["message"]?.try(&.["usage"]?).try(&.as_h?).try { |u| state.usage.merge!(u) }
          when "content_block_start"
            start_block(event, state.blocks, response)
          when "content_block_delta"
            apply_delta(event, state.blocks, response)
          when "message_delta"
            state.stop_reason = event["delta"]?.try(&.["stop_reason"]?).try(&.as_s?) || state.stop_reason
            event["usage"]?.try(&.as_h?).try { |u| state.usage.merge!(u) }
          when "error"
            raise error_event(event)
          when "message_stop"
            # Read on to EOF, so the chunked body is drained and the
            # connection can be reused.
            state.stopped = true
          end
        end

        private def parse_event(line : String) : Hash(String, JSON::Any)?
          return unless line.starts_with?("data:")
          JSON.parse(line[5..].strip).as_h?
        rescue JSON::ParseException
          nil
        end

        private def start_block(event : Hash(String, JSON::Any), blocks : Hash(Int32, Block), response : Response) : Nil
          start = event["content_block"]?.try(&.as_h?) || return
          block = Block.new(start.dup)
          index = event["index"]?.try(&.as_i?) || (blocks.keys.max? || -1) + 1
          blocks[index] = block
          if block.type == "tool_use"
            response.push_chunk(Response::Chunk.new(block.name, Response::ChunkKind::ToolCallName))
          end
        end

        private def apply_delta(event : Hash(String, JSON::Any), blocks : Hash(Int32, Block), response : Response) : Nil
          index = event["index"]?.try(&.as_i?) || blocks.keys.max? || return
          block = blocks[index]? || return
          delta = event["delta"]?.try(&.as_h?) || return

          case delta["type"]?.try(&.as_s?)
          when "text_delta"
            append_text(block, delta["text"]?, Response::ChunkKind::Content, response)
          when "thinking_delta"
            append_text(block, delta["thinking"]?, Response::ChunkKind::Reasoning, response)
          when "signature_delta"
            delta["signature"]?.try(&.as_s?).try { |s| block.signature << s }
          when "input_json_delta"
            part = delta["partial_json"]?.try(&.as_s?) || return
            block.json << part
            if block.type == "tool_use"
              response.push_chunk(Response::Chunk.new(part, Response::ChunkKind::ToolCallArgs))
            end
          when "citations_delta"
            delta["citation"]?.try { |c| block.citations << c }
          end
        end

        private def append_text(block : Block, value : JSON::Any?, kind : Response::ChunkKind, response : Response) : Nil
          text = value.try(&.as_s?) || return
          block.text << text
          response.push_chunk(Response::Chunk.new(text, kind))
        end

        # The error type is the code (`overloaded_error` is retried).
        private def error_event(event : Hash(String, JSON::Any)) : StreamError
          detail = event["error"]?.try(&.as_h?)
          kind = detail.try(&.["type"]?).try(&.as_s?) || "error"
          message = detail.try(&.["message"]?).try(&.as_s?) || "the stream reported an error"
          StreamError.new("provider error in stream: #{kind}: #{message}", kind)
        end

        private def message(blocks : Array(Block), endpoint : String?) : Message
          text = blocks.select(&.type.==("text")).join(&.text.to_s)
          tool_calls = blocks.select(&.type.==("tool_use")).map do |b|
            ToolCall.new(id: b.id, name: b.name, arguments: b.arguments)
          end
          thinking = blocks.select(&.type.==("thinking")).compact_map(&.text.to_s.presence).join("\n\n")
          Message.new(
            role: Role::Assistant,
            content: text.empty? && !tool_calls.empty? ? nil : text,
            tool_calls: tool_calls.empty? ? nil : tool_calls,
            reasoning: thinking.presence,
            provider_data: blocks.empty? || endpoint.nil? ? nil : Anthropic.replay_data(endpoint, blocks.map(&.to_any)),
          )
        end

        # Everything read counts as prompt, cache reads and writes included,
        # as OpenAI's prompt_tokens does; the cache figures are kept beside
        # it.
        def usage(raw : Hash(String, JSON::Any)) : Usage
          read = raw["cache_read_input_tokens"]?.try(&.as_i?)
          written = raw["cache_creation_input_tokens"]?.try(&.as_i?)
          prompt = (raw["input_tokens"]?.try(&.as_i?) || 0) + (read || 0) + (written || 0)
          completion = raw["output_tokens"]?.try(&.as_i?) || 0
          Usage.new(
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cached_tokens: read,
            cache_creation_tokens: written,
          )
        end

        private def read_timeout(io : IO, timeout : Time::Span?) : Nil
          return unless timeout
          io.read_timeout = timeout if io.responds_to?(:read_timeout=)
        end
      end
    end
  end
end

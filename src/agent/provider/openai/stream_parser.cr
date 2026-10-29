class Agent
  module Provider
    class OpenAI
      # Accumulates a tool call across multiple SSE deltas.
      # Must be a class (reference type) so the hash entry is mutated in place.
      private class ToolCallDelta
        property id : String
        property name : String
        property arguments : String

        def initialize
          @id = ""
          @name = ""
          @arguments = ""
        end
      end

      # Routes tool-call fragments to per-call accumulator entries.
      #
      # The OpenAI wire format tags every tool-call delta with an integer
      # `index`, and most providers also repeat the call `id` on every
      # fragment. Some upstreams (e.g. OpenRouter providers) omit `index`,
      # and argument fragments typically carry neither `id` nor `name`, so
      # routing degrades through three tiers:
      #
      #   1. fragment carries `id` -> join (or open) the entry with that id.
      #   2. fragment carries `index` (no id) -> join (or open) the entry
      #      first opened under that index.
      #   3. fragment carries neither -> continue the most recently routed
      #      entry, unless its `name` is set and the fragment names a
      #      different function — in that case open a new entry rather than
      #      merging two distinct calls (see route_by_cursor).
      #
      # Internal keys only need to be unique; entries surface in
      # first-appearance order because Crystal hashes keep insertion order.
      private class ToolCallAccumulator
        getter deltas : Hash(Int32, ToolCallDelta)

        @keys_by_id = {} of String => Int32
        @keys_by_index = {} of Int32 => Int32
        @cursor : Int32?
        @next_key = 0

        def initialize
          @deltas = {} of Int32 => ToolCallDelta
        end

        def route(tcd : Hash(String, JSON::Any)) : ToolCallDelta
          id = tcd["id"]?.try(&.as_s?)
          index = tcd["index"]?.try(&.as_i)

          key = if id && !id.empty?
                  route_by_id(id, index)
                elsif idx = index
                  route_by_index(idx)
                else
                  route_by_cursor(tcd)
                end

          @cursor = key
          @deltas[key] ||= ToolCallDelta.new
        end

        # Smallest internal key not already occupied. Synthetic keys may be
        # needed when the provider reuses `index` values across distinct
        # calls or omits `index` entirely.
        private def new_key : Int32
          while @deltas.has_key?(@next_key)
            @next_key += 1
          end
          @next_key
        end

        # An id is strong identity: fragments sharing an id always belong to
        # the same call, and two different ids must never share an entry —
        # a provider that reuses `index` values across parallel calls would
        # otherwise concatenate their arguments. A fresh id opens a new
        # entry under its `index` when that key is free, adopts an id-less
        # entry already opened under that index, or takes a synthetic key.
        private def route_by_id(id : String, index : Int32?) : Int32
          if key = @keys_by_id[id]?
            return key
          end

          key = if idx = index
                  if entry = @deltas[idx]?
                    if entry.id.empty?
                      # Adopt the id-less entry opened under this index.
                      @keys_by_index[idx] ||= idx
                      idx
                    else
                      # Occupied by a different call — never merge ids.
                      new_key
                    end
                  else
                    idx
                  end
                else
                  new_key
                end

          @keys_by_id[id] = key
          @keys_by_index[index] = key if index
          key
        end

        # Explicit `index` is authoritative for id-less fragments — this
        # keeps standard OpenAI-style streams on their previous code path.
        private def route_by_index(index : Int32) : Int32
          if key = @keys_by_index[index]?
            return key
          end

          # Prefer the literal index as the internal key; if a previous
          # entry already occupies it, allocate a fresh synthetic key.
          key = @deltas.has_key?(index) ? new_key : index
          @keys_by_index[index] = key
          key
        end

        # Neither id nor index: continue the entry routed most recently.
        # If that entry already carries a function name and the fragment
        # names a different function, the provider has moved on to another
        # call without emitting any identifier — open a new entry instead
        # of concatenating two calls' argument fragments (which can parse
        # as valid JSON while silently losing keys) or letting the later
        # call's name overwrite the earlier one. The known cost: a provider
        # that streams a single name in pieces *and* omits id and index
        # would split into multiple entries — far rarer than the collisions
        # this guard prevents.
        private def route_by_cursor(tcd : Hash(String, JSON::Any)) : Int32
          if key = @cursor
            entry_name = @deltas[key].name
            incoming_name = tcd["function"]?.try(&.as_h?).try(&.["name"]?).try(&.as_s?)
            return key if entry_name.empty? || incoming_name.nil? ||
                          incoming_name.empty? || entry_name == incoming_name
          end
          new_key
        end
      end

      # Parses an OpenAI SSE streaming response.
      module StreamParser
        extend self

        def parse(
          body_io : IO,
          response : Response,
          cancel : -> Bool,
          *,
          first_byte_timeout : Time::Span? = nil,
          idle_timeout : Time::Span? = nil,
        ) : {Message, Usage, String?}
          tool_calls = ToolCallAccumulator.new
          content_buffer = String::Builder.new
          reasoning_buffer = String::Builder.new
          usage = Usage.new
          finish_reason = nil
          read_count = 0

          # Apply per-read timeouts to the IO when it supports them. Real
          # HTTP::Client body_io (HTTP::ChunkedContent / SSL sockets) honours
          # the timeout set on the underlying socket via HTTP::Client#read_timeout.
          # Custom IOs (tests, IO::Memory) silently ignore these setters.
          #
          # The first read uses `first_byte_timeout`; after every successful
          # read we switch to `idle_timeout`. When `first_byte_timeout` is nil
          # (dynamic first-byte disabled), both reads use `idle_timeout`.
          effective_first = first_byte_timeout || idle_timeout
          apply_read_timeout(body_io, effective_first) if effective_first

          loop do
            # Check for cancellation before each blocking read — the cancel
            # flag and the IO timeout race independently, but cancel should
            # win when both fire (see process_request_loop's order).
            break if cancel.call

            # `#gets` consults `read_timeout` each invocation on IOs that
            # support it. Catch IO::TimeoutError and re-raise as IdleTimeoutError
            # with the right phase so callers get a consistent Agent::Error type
            # (this also lets http_post_stream match phase without inspecting
            # the partial message — partial_msg is only set on cancellation).
            line = begin
              body_io.gets(chomp: true)
            rescue ex : IO::TimeoutError
              phase = read_count.zero? ? :first_byte : :idle
              timeout = read_count.zero? ? effective_first : idle_timeout
              raise IdleTimeoutError.new(phase, timeout, cause: ex)
            end
            break if line.nil? # EOF
            read_count += 1

            # Switch to the idle (inter-byte) budget for subsequent reads.
            # Any byte arrival — including comment lines like
            # `: OPENROUTER PROCESSING`, empty lines, and the `[DONE]`
            # sentinel — resets the timer because `read_timeout` is per-read.
            if idle_timeout && idle_timeout != effective_first
              apply_read_timeout(body_io, idle_timeout)
            end

            usage, finish_reason = handle_line(
              line,
              response,
              content_buffer,
              reasoning_buffer,
              tool_calls,
              usage,
              finish_reason,
            )
          end

          final_message = build_final_message(content_buffer, reasoning_buffer, tool_calls.deltas)
          {final_message, usage, finish_reason}
        end

        # Process a single SSE line. Returns the updated {usage, finish_reason}
        # tuple so the caller can thread state across iterations. Lines that are
        # empty, non-`data:` prefixed, the `[DONE]` sentinel, or unparseable
        # JSON are no-ops (per SSE convention, any byte arrival resets the idle
        # timer — that reset happens in `#parse` after each successful read,
        # regardless of whether the line produced deltas).
        private def handle_line(
          line : String,
          response : Response,
          content_buffer : String::Builder,
          reasoning_buffer : String::Builder,
          tool_calls : ToolCallAccumulator,
          usage : Usage,
          finish_reason : String?,
        ) : {Usage, String?}
          line = line.strip
          return {usage, finish_reason} if line.empty?
          return {usage, finish_reason} unless line.starts_with?("data:")

          # SSE sentinel — skip [DONE] (with or without space after data:)
          rest = line[5..].lstrip(' ')
          return {usage, finish_reason} if rest.starts_with?("[DONE]")

          json = begin
            JSON.parse(rest)
          rescue JSON::ParseException
            return {usage, finish_reason}
          end
          parsed = json.as_h?
          return {usage, finish_reason} unless parsed

          new_usage = parse_usage(parsed, usage)
          new_finish = process_deltas(parsed, response, content_buffer, reasoning_buffer, tool_calls) || finish_reason
          {new_usage, new_finish}
        end

        # Best-effort per-read timeout application. Uses runtime dispatch
        # because the compile-time type of `io` is `IO+` and most concrete
        # IOs (including HTTP::ChunkedContent + SSL sockets) implement
        # `read_timeout=` even when the abstract type does not surface it.
        # IOs that don't (e.g. IO::Memory) silently no-op via the unmatched
        # case branch.
        private def apply_read_timeout(io : IO, timeout : Time::Span?) : Nil
          return unless timeout
          case io
          when .responds_to?(:read_timeout=)
            io.read_timeout = timeout
          end
        end

        private def parse_usage(parsed : Hash(String, JSON::Any), prev_usage : Usage) : Usage
          if usage_data = parsed["usage"]?
            if u = usage_data.as_h?
              cached = u["prompt_tokens_details"]?.try(&.as_h?).try(&.["cached_tokens"]?).try(&.as_i?)
              return Usage.new(
                prompt_tokens: u["prompt_tokens"]?.try(&.as_i),
                completion_tokens: u["completion_tokens"]?.try(&.as_i),
                total_tokens: u["total_tokens"]?.try(&.as_i),
                cached_tokens: cached,
              )
            end
          elsif timings = parsed["timings"]?
            if timings_h = timings.as_h?
              prompt_n = timings_h["prompt_n"]?.try(&.as_i)
              predicted_n = timings_h["predicted_n"]?.try(&.as_i)
              return Usage.new(
                prompt_tokens: prev_usage.prompt_tokens || prompt_n,
                completion_tokens: prev_usage.completion_tokens || predicted_n,
                total_tokens: prev_usage.total_tokens || (prompt_n && predicted_n ? prompt_n + predicted_n : nil)
              )
            end
          end
          prev_usage
        end

        private def process_deltas(
          parsed : Hash(String, JSON::Any),
          response : Response,
          content_buffer : String::Builder,
          reasoning_buffer : String::Builder,
          tool_calls : ToolCallAccumulator,
        ) : String?
          finish_reason = nil

          choices = parsed["choices"]?.try(&.as_a?) || [] of JSON::Any
          choices.each do |choice|
            delta = choice["delta"]?.try(&.as_h?) || next

            if reason = choice["finish_reason"]?.try(&.as_s?)
              finish_reason = reason
            end

            if c = delta["content"]?.try(&.as_s?)
              content_buffer << c
              response.push_chunk(Response::Chunk.new(c, Response::ChunkKind::Content))
            end

            if rc = reasoning_delta_text(delta)
              reasoning_buffer << rc
              response.push_chunk(Response::Chunk.new(rc, Response::ChunkKind::Reasoning))
            end

            process_tool_call_deltas(delta, tool_calls, response)
          end

          finish_reason
        end

        # Extracts the reasoning text for a single delta across the field shapes
        # used by different providers. Pushes exactly one chunk per delta: the
        # first non-empty form found in a fixed precedence order, so providers
        # that emit both a string field and `reasoning_details` (e.g. OpenRouter)
        # are not double-counted.
        #
        # Order: `reasoning_content` (DeepSeek native / vLLM) -> `reasoning`
        # (OpenRouter, Gemini-compat) -> `thinking` (Anthropic-via-proxy) ->
        # `reasoning_details[].text` (OpenAI Responses / OpenRouter structured).
        private def reasoning_delta_text(delta : Hash(String, JSON::Any)) : String?
          {"reasoning_content", "reasoning", "thinking"}.each do |key|
            if text = delta[key]?.try(&.as_s?)
              return text
            end
          end

          # OpenAI Responses / OpenRouter structured form. Concatenate all
          # `reasoning.text` items in this delta; other types in the array
          # (e.g. `summary`, `encrypted`) are intentionally skipped.
          if details = delta["reasoning_details"]?.try(&.as_a?)
            joined = details.each_with_object(String::Builder.new) do |item, buf|
              next unless item_h = item.as_h?
              next unless item_h["type"]?.try(&.as_s?) == "reasoning.text"
              if t = item_h["text"]?.try(&.as_s?)
                buf << t
              end
            end
            s = joined.to_s
            return s unless s.empty?
          end

          nil
        end

        private def process_tool_call_deltas(
          delta : Hash(String, JSON::Any),
          tool_calls : ToolCallAccumulator,
          response : Response,
        ) : Nil
          tc_delta = delta["tool_calls"]?.try(&.as_a?) || return
          tc_delta.each do |tcd_any|
            tcd = tcd_any.as_h? || next
            # Route the fragment to its per-call entry (by id, then index,
            # then most-recent — see ToolCallAccumulator).
            entry = tool_calls.route(tcd)
            update_tool_call_id(entry, tcd)
            update_tool_call_function(entry, tcd, response)
          end
        end

        private def update_tool_call_id(entry : ToolCallDelta, tcd : Hash(String, JSON::Any)) : Nil
          if id = tcd["id"]?
            if id_s = id.as_s?
              entry.id = id_s
            end
          end
        end

        private def update_tool_call_function(entry : ToolCallDelta, tcd : Hash(String, JSON::Any), response : Response) : Nil
          fn = tcd["function"]? || return
          fn_h = fn.as_h? || return

          if fn_h_name = fn_h["name"]?
            if name_str = fn_h_name.as_s?
              if entry.name.empty?
                response.push_chunk(Response::Chunk.new(name_str, Response::ChunkKind::ToolCallName))
              end
              entry.name = name_str
            end
          end

          if fn_h_args = fn_h["arguments"]?
            if args_str = fn_h_args.as_s?
              entry.arguments += args_str
              response.push_chunk(Response::Chunk.new(args_str, Response::ChunkKind::ToolCallArgs))
            end
          end
        end

        private def build_final_message(
          content_buffer : String::Builder,
          reasoning_buffer : String::Builder,
          tool_call_deltas : Hash(Int32, ToolCallDelta),
        ) : Message
          full_content = content_buffer.to_s
          reasoning_content = reasoning_buffer.to_s

          tool_calls = if tool_call_deltas.empty?
                         nil
                       else
                         tool_call_deltas.map do |_idx, delta|
                           ToolCall.new(id: delta.id, name: delta.name, arguments: delta.arguments)
                         end
                       end

          content = full_content
          content = nil if tool_calls && full_content.empty?

          Message.new(
            role: Role::Assistant,
            content: content,
            tool_calls: tool_calls,
            reasoning: reasoning_content.empty? ? nil : reasoning_content,
          )
        end
      end
    end
  end
end

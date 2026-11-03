require "json"

class Agent
  module Provider
    class Anthropic
      # Builds the Messages API request body from the shard's flat history.
      module RequestBody
        extend self

        # The API has no level below "low" and can't turn thinking off on
        # current models, so :off and :minimal ask for the least thinking.
        EFFORT = {
          ReasoningEffort::Off     => "low",
          ReasoningEffort::Minimal => "low",
          ReasoningEffort::Low     => "low",
          ReasoningEffort::Medium  => "medium",
          ReasoningEffort::High    => "high",
          ReasoningEffort::Xhigh   => "xhigh",
          ReasoningEffort::Max     => "max",
        }

        # The API allows 4 markers; the system prompt and the latest turn
        # take 2, which leaves 2 for a long turn (see #mark_cache).
        MAX_MARKERS = 4
        # A marker finds the previous cache entry only within 20 positions
        # back, so a turn longer than that gets a marker every 15.
        LOOKBACK      = 20
        MARKER_STRIDE = 15

        def marker(ttl : CacheTTL) : JSON::Any
          h = {"type" => JSON::Any.new("ephemeral")}
          h["ttl"] = JSON::Any.new("1h") if ttl.one_hour?
          JSON::Any.new(h)
        end

        def build(
          messages : Array(Message),
          tools : Array(Tool)?,
          *,
          model : String,
          max_tokens : Int32,
          effort : ReasoningEffort? = nil,
          thinking : Bool = true,
          display : String? = nil,
          prefix_mismatch : String? = nil,
          eager_input_streaming : Bool = false,
          extra_body : Hash(String, JSON::Any)? = nil,
          endpoint : String? = nil,
          caching : PromptCaching = PromptCaching::Explicit,
          ttl : CacheTTL = CacheTTL::FiveMinutes,
        ) : Hash(String, JSON::Any)
          body = {
            "model"      => JSON::Any.new(model),
            "max_tokens" => JSON::Any.new(max_tokens.to_i64),
            "stream"     => JSON::Any.new(true),
          }

          if thinking
            config = {"type" => JSON::Any.new("adaptive")}
            display.try { |d| config["display"] = JSON::Any.new(d) }
            if pm = prefix_mismatch
              config["block_binding"] = JSON::Any.new({"prefix_mismatch_behavior" => JSON::Any.new(pm)})
            end
            body["thinking"] = JSON::Any.new(config)
          end
          if e = effort
            body["output_config"] = JSON::Any.new({"effort" => JSON::Any.new(EFFORT[e])})
          end

          # Every system message goes in `system`, a later one included: the
          # API takes one inside the conversation only from some models, and
          # only right after a user message. Adding one rewrites the prompt,
          # so the cache restarts and earlier thinking is dropped (see
          # `prefix_mismatch`).
          system = messages.select(&.role.system?).map { |m| text_block(m.content.to_s) }
          body["system"] = JSON::Any.new(system) unless system.empty?
          tool_defs = tools.try(&.map { |t| tool(t, eager_input_streaming) }) || [] of JSON::Any
          body["tools"] = JSON::Any.new(tool_defs) unless tool_defs.empty?
          turns = turns(messages.reject(&.role.system?), endpoint)
          body["messages"] = JSON::Any.new(turns)

          unless caching.off?
            # Tools render before the system prompt, so one marker on the
            # last of them caches both.
            (system.last? || tool_defs.last?).try { |b| b.as_h["cache_control"] = marker(ttl) }
            if caching.automatic?
              body["cache_control"] = marker(ttl)
            else
              mark_cache(turns, marker(ttl))
            end
          end

          extra_body.try &.each { |k, v| body[k] = v unless body.has_key?(k) }
          body
        end

        private def tool(t : Tool, eager : Bool) : JSON::Any
          schema = t.function.parameters || {
            "type"       => JSON::Any.new("object"),
            "properties" => JSON::Any.new({} of String => JSON::Any),
          }
          h = {
            "name"         => JSON::Any.new(t.function.name),
            "description"  => JSON::Any.new(t.function.description.to_s),
            "input_schema" => JSON::Any.new(schema),
          }
          h["eager_input_streaming"] = JSON::Any.new(true) if eager
          JSON::Any.new(h)
        end

        # The flat list into the API's turns. Everything between two assistant
        # replies is one user turn: tool results (parallel calls are answered
        # in one message) and then any user messages, such as a background
        # task's result followed by what the person typed. The API would
        # merge consecutive user turns itself; other backends refuse them.
        private def turns(messages : Array(Message), endpoint : String?) : Array(JSON::Any)
          turns = [] of JSON::Any
          pending = [] of JSON::Any
          flush = -> {
            unless pending.empty?
              turns << turn("user", pending.dup)
              pending.clear
            end
          }
          messages.each do |m|
            case m.role
            in .tool?
              pending << tool_result(m)
            in .user?
              pending.concat(user_blocks(m))
            in .assistant?
              flush.call
              turns << turn("assistant", assistant_blocks(m, endpoint))
            in .system?
              # in `system` (see #build)
            end
          end
          flush.call
          turns
        end

        # A marker on the last block of the last user turn caches the
        # transcript so far, so each call of a tool loop re-reads it rather
        # than re-sends it. Moving a marker doesn't count as editing the
        # history.
        #
        # The previous request's marker sat on the user turn before it. When
        # more than LOOKBACK positions came since, the new marker can't find
        # that entry, so markers go every MARKER_STRIDE positions back from
        # the end, as many as the API allows.
        private def mark_cache(turns : Array(JSON::Any), mark : JSON::Any) : Nil
          users = turns.each_index.select { |i| turns[i]["role"] == "user" }.to_a
          last = users.last? || return
          positions = positions(turns, (users[-2]? || -1) + 1, last)
          return if positions.empty?

          marks = [positions.last]
          if users.size > 1 && positions.size > LOOKBACK
            i = positions.size - 1 - MARKER_STRIDE
            # One marker is the system prompt's.
            while i >= 0 && marks.size < MAX_MARKERS - 1
              marks << positions[i]
              i -= MARKER_STRIDE
            end
          end
          marks.each { |(t, b)| set_marker(turns, t, b, mark) }
        end

        # Blocks a marker can go on from turn `first` through `last`, one per
        # position: a run of tool_use blocks, or of tool_result blocks, is
        # one position (its last block). Thinking blocks take no marker.
        private def positions(turns : Array(JSON::Any), first : Int32, last : Int32) : Array({Int32, Int32})
          out = [] of {Int32, Int32}
          (first..last).each do |t|
            blocks = turns[t]["content"].as_a
            blocks.each_with_index do |b, i|
              type = b["type"]?.try(&.as_s?)
              next if type.in?({"thinking", "redacted_thinking"})
              next if type.in?({"tool_use", "tool_result"}) && blocks[i + 1]?.try(&.["type"]?.try(&.as_s?)) == type
              out << {t, i}
            end
          end
          out
        end

        # A copy of the block gets the marker: an assistant turn's blocks
        # are the reply's stored ones, sent back unchanged.
        private def set_marker(turns : Array(JSON::Any), t : Int32, b : Int32, mark : JSON::Any) : Nil
          blocks = turns[t]["content"].as_a.dup
          block = blocks[b].as_h.dup
          block["cache_control"] = mark
          blocks[b] = JSON::Any.new(block)
          turns[t].as_h["content"] = JSON::Any.new(blocks)
        end

        private def turn(role : String, blocks : Array(JSON::Any)) : JSON::Any
          JSON::Any.new({"role" => JSON::Any.new(role), "content" => JSON::Any.new(blocks)})
        end

        private def text_block(text : String) : JSON::Any
          JSON::Any.new({"type" => JSON::Any.new("text"), "text" => JSON::Any.new(text)})
        end

        # User content always goes as blocks, so a turn reads the same
        # whether or not it carries the cache breakpoint.
        private def user_blocks(m : Message) : Array(JSON::Any)
          blocks = m.content_parts.try(&.compact_map { |p| content_part(p) }) || [] of JSON::Any
          # Text alone, or every part was audio: the API refuses an empty turn.
          blocks << text_block(m.content.presence || "(no content)") if blocks.empty?
          blocks
        end

        private def content_part(p : ContentPart) : JSON::Any?
          case p.type
          in .text?
            p.text.try { |t| text_block(t) }
          in .image_url?
            source(p).try { |s| JSON::Any.new({"type" => JSON::Any.new("image"), "source" => s}) }
          in .file?
            source(p).try { |s| JSON::Any.new({"type" => JSON::Any.new("document"), "source" => s}) }
          in .input_audio?
            nil # the Messages API takes no audio
          end
        end

        # A data: URI or base64 data becomes a base64 source; anything else
        # is fetched by the API from its URL.
        private def source(p : ContentPart) : JSON::Any?
          if (url = p.url) && (m = url.match(/\Adata:([^;,]+);base64,(.*)\z/m))
            return base64_source(m[1], m[2])
          end
          if (data = p.data) && (mime = p.mime_type)
            return base64_source(mime, data)
          end
          p.url.try { |u| JSON::Any.new({"type" => JSON::Any.new("url"), "url" => JSON::Any.new(u)}) }
        end

        private def base64_source(media_type : String, data : String) : JSON::Any
          JSON::Any.new({
            "type"       => JSON::Any.new("base64"),
            "media_type" => JSON::Any.new(media_type),
            "data"       => JSON::Any.new(data),
          })
        end

        private def tool_result(m : Message) : JSON::Any
          block = {
            "type"        => JSON::Any.new("tool_result"),
            "tool_use_id" => JSON::Any.new(m.tool_call_id.to_s),
            "content"     => JSON::Any.new(m.content.to_s),
          }
          block["is_error"] = JSON::Any.new(true) if m.error?
          JSON::Any.new(block)
        end

        # The reply's own blocks when it came from this endpoint, so signed
        # thinking goes back byte for byte; otherwise rebuilt from its text
        # and tool calls (another provider's reply, or a cancelled one).
        private def assistant_blocks(m : Message, endpoint : String?) : Array(JSON::Any)
          if raw = Anthropic.blocks(m, endpoint)
            return raw
          end

          blocks = [] of JSON::Any
          if (text = m.content) && !text.empty?
            blocks << text_block(text)
          end
          m.tool_calls.try &.each do |tc|
            blocks << JSON::Any.new({
              "type"  => JSON::Any.new("tool_use"),
              "id"    => JSON::Any.new(tc.id),
              "name"  => JSON::Any.new(tc.name),
              "input" => parse_input(tc.arguments),
            })
          end
          # The API refuses an empty assistant turn.
          blocks << text_block("(no content)") if blocks.empty?
          blocks
        end

        def parse_input(arguments : String) : JSON::Any
          parsed = JSON.parse(arguments.presence || "{}")
          parsed.as_h? ? parsed : JSON::Any.new({} of String => JSON::Any)
        rescue JSON::ParseException
          JSON::Any.new({} of String => JSON::Any)
        end
      end
    end
  end
end

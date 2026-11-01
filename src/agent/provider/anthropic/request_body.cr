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

        EPHEMERAL = JSON::Any.new({"type" => JSON::Any.new("ephemeral")})

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
          system = messages.select(&.role.system?)
          unless system.empty?
            blocks = system.map { |m| text_block(m.content.to_s) }
            # One breakpoint on the last system block caches the tools and
            # the prompt.
            blocks.last.as_h["cache_control"] = EPHEMERAL
            body["system"] = JSON::Any.new(blocks)
          end

          if (ts = tools) && !ts.empty?
            body["tools"] = JSON::Any.new(ts.map { |t| tool(t, eager_input_streaming) })
          end
          body["messages"] = JSON::Any.new(turns(messages.reject(&.role.system?), endpoint))

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

        # The flat list into the API's turns. Consecutive tool results become
        # one user turn of tool_result blocks, since parallel calls are
        # answered in one message.
        private def turns(messages : Array(Message), endpoint : String?) : Array(JSON::Any)
          turns = [] of JSON::Any
          results = [] of JSON::Any
          flush = -> {
            unless results.empty?
              turns << turn("user", results.dup)
              results.clear
            end
          }
          messages.each do |m|
            case m.role
            in .tool?
              results << tool_result(m)
            in .user?
              flush.call
              turns << turn("user", user_blocks(m))
            in .assistant?
              flush.call
              turns << turn("assistant", assistant_blocks(m, endpoint))
            in .system?
              # in `system` (see #build)
            end
          end
          flush.call
          mark_cache(turns)
          turns
        end

        # A breakpoint on the last block of the last user turn caches the
        # transcript so far, so each call of a tool loop re-reads it rather
        # than re-sends it. Moving a breakpoint doesn't count as editing the
        # history.
        private def mark_cache(turns : Array(JSON::Any)) : Nil
          last = turns.reverse_each.find { |t| t["role"] == "user" } || return
          last["content"].as_a.last?.try(&.as_h.[]=("cache_control", EPHEMERAL))
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

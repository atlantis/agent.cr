class Agent
  module Provider
    class OpenAI
      # Builds the JSON request body for the OpenAI chat completions API.
      module RequestBody
        extend self

        def build(
          messages : Array(Message),
          tools : Array(Tool)?,
          model : String,
          max_tokens : Int32?,
          temperature : Float64?,
          cache_key : String?,
          include_usage : Bool = true,
          reasoning_effort : ReasoningEffort? = nil,
          extra_body : Hash(String, JSON::Any)? = nil,
          *,
          openrouter : Bool = false,
          endpoint : String? = nil,
          replay : Bool = true,
          caching : {PromptCaching, CacheTTL}? = nil,
        ) : Hash(String, JSON::Any)
          wire = messages.map { |m| message_to_request_body(m, replay ? {endpoint, model} : nil) }
          body = {
            "model"    => JSON::Any.new(model),
            "messages" => JSON::Any.new(wire.map { |h| JSON::Any.new(h) }),
            "stream"   => JSON::Any.new(true),
          }
          caching.try { |(mode, ttl)| mark_cache(body, wire, mode, ttl) }

          if include_usage
            body["stream_options"] = JSON::Any.new({"include_usage" => JSON::Any.new(true)})
          end

          # Only send prompt_cache_key if it was explicitly configured.
          # Non-OpenAI providers may reject unknown fields.
          if cache_key
            body["prompt_cache_key"] = JSON::Any.new(cache_key)
          end

          if mt = max_tokens
            body["max_tokens"] = JSON::Any.new(mt.to_i64)
          end

          if t = temperature
            body["temperature"] = JSON::Any.new(t)
          end

          if effort = reasoning_effort
            if openrouter
              body["reasoning"] = JSON::Any.new({"effort" => JSON::Any.new(OPENROUTER_EFFORT[effort])})
            else
              body["reasoning_effort"] = JSON::Any.new(OPENAI_EFFORT[effort])
            end
          end

          if ts = tools
            body["tools"] = JSON::Any.new(ts.map do |tool_def|
              fd = tool_def.function
              JSON::Any.new({
                "type"     => JSON::Any.new("function"),
                "function" => JSON::Any.new({
                  "name"        => JSON::Any.new(fd.name),
                  "description" => JSON::Any.new(fd.description || ""),
                  "parameters"  => JSON::Any.new(fd.parameters || {} of String => JSON::Any),
                }),
              })
            end)
          end

          extra_body.try &.each { |k, v| body[k] = v unless body.has_key?(k) }

          body
        end

        # Cache markers as OpenRouter takes them for Claude models, on content
        # parts: the last system message (OpenRouter puts the tools ahead of
        # it), and the latest user or tool message, or the API's own marker
        # for the conversation (`cache_control` at the top level).
        private def mark_cache(body : Hash(String, JSON::Any), wire : Array(Hash(String, JSON::Any)), mode : PromptCaching, ttl : CacheTTL) : Nil
          mark = {"type" => JSON::Any.new("ephemeral")}
          mark["ttl"] = JSON::Any.new("1h") if ttl.one_hour?
          mark = JSON::Any.new(mark)
          wire.reverse_each.find { |m| m["role"] == "system" }.try { |m| set_marker(m, mark) }
          if mode.automatic?
            body["cache_control"] = mark
          else
            wire.reverse_each.find(&.["role"].in?("user", "tool")).try { |m| set_marker(m, mark) }
          end
        end

        # Text content becomes one text part, so it can carry the marker.
        private def set_marker(message : Hash(String, JSON::Any), mark : JSON::Any) : Nil
          parts = message["content"]?.try(&.as_a?).try(&.dup)
          unless parts
            # The API refuses a marker on an empty text block.
            text = message["content"]?.try(&.as_s?).presence || return
            parts = [JSON::Any.new({"type" => JSON::Any.new("text"), "text" => JSON::Any.new(text)})]
          end
          return if parts.empty?
          parts[-1] = JSON::Any.new(parts[-1].as_h.merge({"cache_control" => mark}))
          message["content"] = JSON::Any.new(parts)
        end

        # OpenAI has no "max"; its highest is "xhigh". "none" turns reasoning
        # off on models that allow it.
        OPENAI_EFFORT = {
          ReasoningEffort::Off     => "none",
          ReasoningEffort::Minimal => "minimal",
          ReasoningEffort::Low     => "low",
          ReasoningEffort::Medium  => "medium",
          ReasoningEffort::High    => "high",
          ReasoningEffort::Xhigh   => "xhigh",
          ReasoningEffort::Max     => "xhigh",
        }

        OPENROUTER_EFFORT = {
          ReasoningEffort::Off     => "none",
          ReasoningEffort::Minimal => "minimal",
          ReasoningEffort::Low     => "low",
          ReasoningEffort::Medium  => "medium",
          ReasoningEffort::High    => "high",
          ReasoningEffort::Xhigh   => "xhigh",
          ReasoningEffort::Max     => "max",
        }

        # Sends back the reasoning an assistant message came with, in the
        # shape its endpoint returned it, so the model keeps its reasoning
        # across a tool loop:
        # - `reasoning_details` blocks (OpenRouter), encrypted and signed
        #   ones included, unchanged and in order;
        # - `reasoning_content` text (DeepSeek, Kimi, vLLM);
        # - Mistral's thinking chunk ahead of the answer in `content`.
        private def replay_reasoning(body : Hash(String, JSON::Any), msg : Message, ours : Hash(String, JSON::Any)) : Nil
          if details = ours["reasoning_details"]?
            body["reasoning_details"] = details
          end
          text = msg.reasoning.presence || return

          case ours["reasoning_field"]?.try(&.as_s?)
          when "reasoning_content"
            body["reasoning_content"] = JSON::Any.new(text)
          when "content_thinking"
            thinking = {"type" => JSON::Any.new("text"), "text" => JSON::Any.new(text)}
            parts = [JSON::Any.new({"type" => JSON::Any.new("thinking"), "thinking" => JSON::Any.new([JSON::Any.new(thinking)])})]
            if answer = msg.content.presence
              parts << JSON::Any.new({"type" => JSON::Any.new("text"), "text" => JSON::Any.new(answer)})
            end
            body["content"] = JSON::Any.new(parts)
          end
        end

        # Build the OpenAI wire-format hash for a single Message.
        # `scope` is the {endpoint, model} whose reasoning is sent back, or
        # nil to send none.
        private def message_to_request_body(msg : Message, scope : {String?, String}?) : Hash(String, JSON::Any)
          body = {"role" => JSON::Any.new(msg.role.to_s)}

          # Build content
          if parts = msg.content_parts
            body["content"] = JSON::Any.new(parts.map { |p| content_part_to_json_body(p) }.map { |h| JSON::Any.new(h) })
          elsif text = msg.content
            body["content"] = JSON::Any.new(text)
          elsif msg.role == Role::Tool
            body["content"] = JSON::Any.new("")
          else
            body["content"] = JSON::Any.new(nil)
          end

          body["tool_call_id"] = JSON::Any.new(msg.tool_call_id) if msg.tool_call_id
          body["name"] = JSON::Any.new(msg.name) if msg.name

          if tcs = msg.tool_calls
            body["tool_calls"] = JSON::Any.new(tcs.map do |tc|
              call = {
                "id"       => JSON::Any.new(tc.id),
                "type"     => JSON::Any.new("function"),
                "function" => JSON::Any.new({
                  "name"      => JSON::Any.new(tc.name),
                  "arguments" => JSON::Any.new(tc.arguments),
                }),
              }
              if extra = scope.try { |(e, m)| OpenAI.replay_entries(tc.provider_data, e, m) }.try(&.["extra_content"]?)
                call["extra_content"] = extra
              end
              JSON::Any.new(call)
            end)
          end

          if msg.role == Role::Assistant
            if ours = scope.try { |(e, m)| OpenAI.replay_entries(msg.provider_data, e, m) }
              replay_reasoning(body, msg, ours)
            end
          end

          body
        end

        # Build the OpenAI wire-format hash for a single ContentPart.
        # Delegates to ContentPart#to_json_body so wire format is centralised.
        private def content_part_to_json_body(part : ContentPart) : Hash(String, JSON::Any)
          part.to_json_body
        end
      end
    end
  end
end

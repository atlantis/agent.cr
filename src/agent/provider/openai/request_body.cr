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
        ) : Hash(String, JSON::Any)
          body = {
            "model"    => JSON::Any.new(model),
            "messages" => JSON::Any.new(messages.map { |m| message_to_request_body(m) }.map { |h| JSON::Any.new(h) }),
            "stream"   => JSON::Any.new(true),
          }

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

          body
        end

        # Build the OpenAI wire-format hash for a single Message.
        private def message_to_request_body(msg : Message) : Hash(String, JSON::Any)
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
              JSON::Any.new({
                "id"       => JSON::Any.new(tc.id),
                "type"     => JSON::Any.new("function"),
                "function" => JSON::Any.new({
                  "name"      => JSON::Any.new(tc.name),
                  "arguments" => JSON::Any.new(tc.arguments),
                }),
              })
            end)
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

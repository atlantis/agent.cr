require "./spec_helper"

# Agent integration tests using a local test server that mimics the OpenAI API.
# This avoids external HTTP calls and makes tests deterministic.

# A minimal mock server that speaks the OpenAI streaming format.
# the loop's tool execution is private; the spec reaches it through this seam
class Agent
  def execute_registered_tools_for_spec(calls, response)
    execute_registered_tools(calls, response)
  end
end

def with_mock_server(&)
  server = HTTP::Server.new do |ctx|
    if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
      body = ctx.request.body.try(&.gets_to_end) || "{}"
      parsed = JSON.parse(body)

      ctx.response.content_type = "text/event-stream"
      ctx.response.status_code = 200

      messages = parsed["messages"].as_a
      last_user_msg = messages.reverse.find { |m| m["role"].as_s == "user" }
      reply = last_user_msg.try(&.["content"].as_s?) || "Hello"

      # Send reasoning content first (simulating Qwen/DeepSeek models)
      reasoning_text = "Reasoning..."
      reasoning_text.each_char do |ch|
        data = {
          choices: [{
            delta: {reasoning_content: ch.to_s},
            index: 0,
          }],
        }
        ctx.response.puts "data: #{data.to_json}"
        ctx.response.flush
      end

      reply.each_char do |ch|
        data = {
          choices: [{
            delta: {content: ch.to_s},
            index: 0,
          }],
        }
        ctx.response.puts "data: #{data.to_json}"
        ctx.response.flush
      end

      final = {
        choices: [{
          delta:         {} of String => JSON::Any,
          index:         0,
          finish_reason: "stop",
        }],
        usage: {
          prompt_tokens:     10,
          completion_tokens: reply.size,
          total_tokens:      10 + reply.size,
        },
      }
      ctx.response.puts "data: #{final.to_json}"
      ctx.response.puts "data: [DONE]"
      ctx.response.flush
      ctx.response.close
    else
      ctx.response.status_code = 404
      ctx.response.puts "Not Found"
    end
  end

  address = server.bind_tcp(0)
  port = address.port
  ready = Channel(Nil).new
  spawn do
    ready.send(nil)
    server.listen
  end
  ready.receive

  begin
    yield port
  ensure
    server.close
  end
end

describe Agent do
  it "asks a question and gets a streamed response" do
    with_mock_server do |port|
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        model: "gpt-4o",
      )

      agent = Agent.new(config)
      resp = agent.ask("What is the capital of France?")

      chunks = [] of Agent::Response::Chunk
      resp.stream { |chunk| chunks << chunk }

      # Content should only include the actual content (reasoning is separate)
      resp.message.content.should eq("What is the capital of France?")
      resp.message.reasoning.should eq("Reasoning...")
      resp.metadata.total_tokens.should eq(10 + "What is the capital of France?".size)
      resp.finished?.should be_true
    end
  end

  it "tags reasoning chunks correctly" do
    with_mock_server do |port|
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
      )

      agent = Agent.new(config)
      resp = agent.ask("Hello")

      reasoning_chunks = [] of String
      content_chunks = [] of String
      resp.stream do |chunk|
        if chunk.reasoning?
          reasoning_chunks << chunk.text
        else
          content_chunks << chunk.text
        end
      end

      reasoning_chunks.join.should eq("Reasoning...")
      content_chunks.join.should eq("Hello")
    end
  end

  it "maintains conversation history" do
    with_mock_server do |port|
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
      )

      agent = Agent.new(config)
      resp1 = agent.ask("First")
      resp1.join

      resp2 = agent.ask("Second")
      resp2.join

      # History should contain both user messages and their responses
      agent.history.size.should eq(4)
      agent.history[0].role.should eq(Agent::Role::User)
      agent.history[0].content.should eq("First")
      agent.history[2].role.should eq(Agent::Role::User)
      agent.history[2].content.should eq("Second")
    end
  end

  it "resets history" do
    with_mock_server do |port|
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        system_prompt: "Be nice.",
      )

      agent = Agent.new(config)
      resp = agent.ask("Hello")
      resp.join

      agent.history.size.should eq(2) # user + assistant

      agent.reset
      agent.history.should be_empty
    end
  end

  it "supports multimodal input" do
    with_mock_server do |port|
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
      )

      agent = Agent.new(config)
      resp = agent.ask(
        "What's in this image?",
        attachments: ["https://example.com/photo.jpg"]
      )
      resp.join

      # The generated message should have content_parts
      user_msg = agent.history[0]
      user_msg.content_parts.should_not be_nil
      user_msg.content_parts.try do |parts|
        parts.size.should eq(2)
        parts[0].text.should eq("What's in this image?")
        parts[1].url.should eq("https://example.com/photo.jpg")
        parts[1].type.should eq(Agent::ContentPart::PartType::ImageUrl)
      end
    end
  end

  it "supports tool calls in the request" do
    with_mock_server do |port|
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
      )

      agent = Agent.new(config)

      tool = Agent::Tool.new(Agent::Tool::FunctionDef.new(
        name: "get_weather",
        description: "Get the weather",
        parameters: {
          "type"       => JSON::Any.new("object"),
          "properties" => JSON::Any.new({
            "city" => JSON::Any.new({"type" => JSON::Any.new("string")}),
          }),
        }
      ))

      resp = agent.ask("What's the weather in Paris?", tools: [tool])
      resp.join
      resp.message.content.should eq("What's the weather in Paris?")
      resp.message.reasoning.should eq("Reasoning...")
    end
  end

  it "handles API errors gracefully" do
    config = Agent::Config.new(
      api_key: "bad-key",
      api_endpoint: "http://localhost:1",
    )

    agent = Agent.new(config)
    resp = agent.ask("Hello")
    resp.join
    content = resp.message.content
    content.should_not be_nil
    content.to_s.should contain("error")
    resp.error?.should be_true
    resp.error.should be_a(Agent::ConnectionError)
  end

  it "reports finish_reason from the API response" do
    with_mock_server do |port|
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
      )

      agent = Agent.new(config)
      resp = agent.ask("Hello")
      resp.join
      resp.finish_reason.should eq("stop")
    end
  end

  it "streams tool call argument chunks" do
    # Build JSON for tool call streaming test
    # arguments value must have its inner quotes escaped for valid JSON
    args_json = %({"city":"Paris"})
    delta1 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "id" => "call_abc", "type" => "function", "function" => {"name" => "get_weather", "arguments" => ""}}]}, "index" => 0}]}.to_json
    delta2 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "function" => {"arguments" => args_json}}]}, "index" => 0}]}.to_json
    delta3 = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "tool_calls"}], "usage" => {"prompt_tokens" => 5, "completion_tokens" => 10, "total_tokens" => 15}}.to_json

    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        ctx.request.body.try(&.gets_to_end)
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        ctx.response.puts "data: #{delta1}"
        ctx.response.flush

        ctx.response.puts "data: #{delta2}"
        ctx.response.flush

        ctx.response.puts "data: #{delta3}"
        ctx.response.puts "data: [DONE]"
        ctx.response.flush
        ctx.response.close
      else
        ctx.response.status_code = 404
        ctx.response.puts "Not Found"
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
      )

      agent = Agent.new(config)
      resp = agent.ask("What's the weather?")

      tool_chunks = [] of String
      resp.stream do |chunk|
        if chunk.kind == Agent::Response::ChunkKind::ToolCallArgs
          tool_chunks << chunk.text
        end
      end

      tool_chunks.join.should eq(%({"city":"Paris"}))
      tcs = resp.message.tool_calls
      tcs.should_not be_nil
      # ameba:disable Lint/NotNil
      tcs = tcs.not_nil!
      tcs.size.should eq(1)
      tcs[0].name.should eq("get_weather")
      tcs[0].id.should eq("call_abc")
      tcs[0].arguments.should eq(%({"city":"Paris"}))
      resp.finish_reason.should eq("tool_calls")
    ensure
      server.close
    end
  end

  it "auto-resolves registered tools" do
    # A mock server that:
    #  1st request -> returns a tool call (finish_reason: tool_calls)
    #  2nd request -> returns a normal text response (finish_reason: stop)
    call_count = 0
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        # Drain the request body so keep-alive framing stays valid for the
        # next request on the same pooled connection.
        ctx.request.body.try(&.gets_to_end)
        call_count += 1
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        if call_count == 1
          # Return tool call
          delta1 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "id" => "call_001", "type" => "function", "function" => {"name" => "test_tool", "arguments" => ""}}]}, "index" => 0}]}.to_json
          delta2 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "function" => {"arguments" => %({"input":"world"})}}]}, "index" => 0}]}.to_json
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "tool_calls"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}}.to_json
          ctx.response.puts "data: #{delta1}"
          ctx.response.flush
          ctx.response.puts "data: #{delta2}"
          ctx.response.flush
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        else
          # Return normal text (tool result was included in the second request)
          reply = "Tool result received: processed"
          reply.each_char do |ch|
            data = {"choices" => [{"delta" => {"content" => ch.to_s}, "index" => 0}]}.to_json
            ctx.response.puts "data: #{data}"
            ctx.response.flush
          end
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 20, "completion_tokens" => reply.size, "total_tokens" => 20 + reply.size}}.to_json
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        end
        ctx.response.close
      else
        ctx.response.status_code = 404
        ctx.response.puts "Not Found"
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        auto_execute_tools: true,
      )

      agent = Agent.new(config)

      # Register the tool that should be auto-resolved
      tool_result = ""
      agent.register_tool("test_tool", "A test tool",
        parameters: Agent::JSONConverter.from({
          type:       "object",
          properties: {
            input: {type: "string"},
          },
          required: ["input"],
        })
      ) do |args|
        tool_result = args["input"]?.try(&.as_s) || ""
        "Processed: #{tool_result}"
      end

      resp = agent.ask("Run the tool")

      # Stream the response (should be the final text after tool resolution)
      content = [] of String
      resp.stream { |chunk| content << chunk.text if chunk.content? }

      # Verify the auto-resolve loop ran
      resp.message.content.should eq("Tool result received: processed")
      tool_result.should eq("world")

      # Verify the history includes: user, assistant(tool_calls), tool, assistant(final)
      agent.history.size.should eq(4)
      agent.history[0].role.should eq(Agent::Role::User)
      agent.history[1].role.should eq(Agent::Role::Assistant)
      agent.history[1].has_tool_calls?.should be_true
      agent.history[2].role.should eq(Agent::Role::Tool)
      agent.history[3].role.should eq(Agent::Role::Assistant)
      agent.history[3].content.should eq("Tool result received: processed")
    ensure
      server.close
    end
  end

  it "accumulates usage across the auto-resolve tool loop" do
    # A mock server that bills every iteration of the tool loop:
    #   1st request -> tool call, usage 10/5/15
    #   2nd request -> normal text, usage 20/30/50
    # Response#metadata must report the sum (30/35/65) — the whole turn's
    # provider calls — not only the finishing iteration's counts.
    call_count = 0
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        ctx.request.body.try(&.gets_to_end)
        call_count += 1
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        if call_count == 1
          delta1 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "id" => "call_001", "type" => "function", "function" => {"name" => "test_tool", "arguments" => ""}}]}, "index" => 0}]}.to_json
          delta2 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "function" => {"arguments" => %({"input":"world"})}}]}, "index" => 0}]}.to_json
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "tool_calls"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}}.to_json
          ctx.response.puts "data: #{delta1}"
          ctx.response.flush
          ctx.response.puts "data: #{delta2}"
          ctx.response.flush
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        else
          reply = "Tool result received: processed"
          reply.each_char do |ch|
            data = {"choices" => [{"delta" => {"content" => ch.to_s}, "index" => 0}]}.to_json
            ctx.response.puts "data: #{data}"
            ctx.response.flush
          end
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 20, "completion_tokens" => 30, "total_tokens" => 50}}.to_json
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        end
        ctx.response.close
      else
        ctx.response.status_code = 404
        ctx.response.puts "Not Found"
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        auto_execute_tools: true,
      )

      agent = Agent.new(config)
      agent.register_tool("test_tool", "A test tool",
        parameters: Agent::JSONConverter.from({
          type:       "object",
          properties: {
            input: {type: "string"},
          },
          required: ["input"],
        })
      ) do |args|
        "Processed: #{args["input"]?.try(&.as_s) || ""}"
      end

      resp = agent.ask("Run the tool")
      resp.join

      resp.finish_reason.should eq("stop")
      resp.metadata.prompt_tokens.should eq(30)
      resp.metadata.completion_tokens.should eq(35)
      resp.metadata.total_tokens.should eq(65)
    ensure
      server.close
    end
  end

  it "reports error when auto-resolve tool callback raises" do
    call_count = 0
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        ctx.request.body.try(&.gets_to_end)
        call_count += 1
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        if call_count == 1
          delta1 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "id" => "call_001", "type" => "function", "function" => {"name" => "failing_tool", "arguments" => ""}}]}, "index" => 0}]}.to_json
          delta2 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "function" => {"arguments" => %({})}}]}, "index" => 0}]}.to_json
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "tool_calls"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}}.to_json
          ctx.response.puts "data: #{delta1}"
          ctx.response.flush
          ctx.response.puts "data: #{delta2}"
          ctx.response.flush
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        else
          # Second request: verify the tool error message is present in history
          reply = "error noted"
          reply.each_char do |ch|
            data = {"choices" => [{"delta" => {"content" => ch.to_s}, "index" => 0}]}.to_json
            ctx.response.puts "data: #{data}"
            ctx.response.flush
          end
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 20, "completion_tokens" => reply.size, "total_tokens" => 20 + reply.size}}.to_json
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        end
        ctx.response.close
      else
        ctx.response.status_code = 404
        ctx.response.puts "Not Found"
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        auto_execute_tools: true,
      )

      agent = Agent.new(config)

      # Register a tool that raises
      agent.register_tool("failing_tool", "A tool that always fails",
        parameters: Agent::JSONConverter.from({
          type:       "object",
          properties: {} of String => String,
          required:   [] of String,
        })
      ) do |_args|
        raise "Intentional failure"
      end

      resp = agent.ask("Test failing tool")
      resp.join

      # The tool error should be captured as a tool result message with error text
      tool_msg = agent.history[2]
      tool_msg.role.should eq(Agent::Role::Tool)
      tool_msg.content.should_not be_nil
      tool_msg.content.to_s.should contain("Error executing tool 'failing_tool'")
      tool_msg.content.to_s.should contain("Intentional failure")
    ensure
      server.close
    end
  end

  it "handles register_tool with no auto_execute" do
    with_mock_server do |port|
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        auto_execute_tools: false,
      )

      agent = Agent.new(config)

      called = false
      agent.register_tool("echo", "Echo input",
        parameters: Agent::JSONConverter.from({
          type:       "object",
          properties: {
            text: {type: "string"},
          },
          required: ["text"],
        })
      ) do |args|
        called = true
        args["text"]?.try(&.as_s) || ""
      end

      # When auto_execute_tools is false, the tool should be included in the request
      # but the callback won't be called automatically.
      resp = agent.ask("Hello")
      resp.join
      called.should be_false
    end
  end

  it "rejects null for a typed tool argument and reports it to the model" do
    # 1st request -> tool call with `"ids": null` against an array-typed property
    # 2nd request -> carries the validation error back to the model as the tool result
    call_count = 0
    request_bodies = [] of String
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        request_bodies << (ctx.request.body.try(&.gets_to_end) || "{}")
        call_count += 1
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        if call_count == 1
          delta1 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "id" => "call_001", "type" => "function", "function" => {"name" => "show_documents", "arguments" => ""}}]}, "index" => 0}]}.to_json
          delta2 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "function" => {"arguments" => %({"ids": null})}}]}, "index" => 0}]}.to_json
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "tool_calls"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}}.to_json
          ctx.response.puts "data: #{delta1}"
          ctx.response.flush
          ctx.response.puts "data: #{delta2}"
          ctx.response.flush
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        else
          reply = "Recovered"
          final = {"choices" => [{"delta" => {"content" => reply}, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 20, "completion_tokens" => reply.size, "total_tokens" => 20 + reply.size}}.to_json
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        end
        ctx.response.close
      else
        ctx.response.status_code = 404
        ctx.response.puts "Not Found"
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        auto_execute_tools: true,
      )

      agent = Agent.new(config)

      callback_called = false
      agent.register_tool("show_documents", "Show documents by id",
        parameters: Agent::JSONConverter.from({
          type:       "object",
          properties: {
            ids: {type: "array", items: {type: "string"}},
          },
          required: ["ids"],
        })
      ) do |_args|
        callback_called = true
        "docs"
      end

      resp = agent.ask("Show the documents")
      resp.join

      # The callback must never see the null argument.
      callback_called.should be_false
      resp.message.content.should eq("Recovered")

      # The tool result fed back to the model names both the expectation and
      # the received type, so a retrying model has a structured signal.
      agent.history[2].role.should eq(Agent::Role::Tool)
      agent.history[2].content.should eq(
        "Error validating arguments for tool 'show_documents': Field 'ids' expected type 'array', got 'null'"
      )
      request_bodies[1].should contain("expected type 'array', got 'null'")
    ensure
      server.close
    end
  end

  it "validates array item types when the schema declares items" do
    # 1st request -> tool call with a mixed-type array `{"ids": ["a", 2]}`
    # 2nd request -> tool call with a valid array (after self-correction)
    # 3rd request -> normal text response
    call_count = 0
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        ctx.request.body.try(&.gets_to_end)
        call_count += 1
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        if call_count == 1
          args = %({"ids": ["a", 2]})
        elsif call_count == 2
          args = %({"ids": ["ok"]})
        else
          args = nil
        end

        if args
          delta1 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "id" => "call_00#{call_count}", "type" => "function", "function" => {"name" => "show_documents", "arguments" => ""}}]}, "index" => 0}]}.to_json
          delta2 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "function" => {"arguments" => args}}]}, "index" => 0}]}.to_json
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "tool_calls"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}}.to_json
          ctx.response.puts "data: #{delta1}"
          ctx.response.flush
          ctx.response.puts "data: #{delta2}"
          ctx.response.flush
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        else
          reply = "Done"
          final = {"choices" => [{"delta" => {"content" => reply}, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 20, "completion_tokens" => reply.size, "total_tokens" => 20 + reply.size}}.to_json
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        end
        ctx.response.close
      else
        ctx.response.status_code = 404
        ctx.response.puts "Not Found"
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        auto_execute_tools: true,
      )

      agent = Agent.new(config)

      callback_calls = 0
      received_ids : Array(JSON::Any)? = nil
      agent.register_tool("show_documents", "Show documents by id",
        parameters: Agent::JSONConverter.from({
          type:       "object",
          properties: {
            ids: {type: "array", items: {type: "string"}},
          },
          required: ["ids"],
        })
      ) do |args|
        callback_calls += 1
        received_ids = args["ids"]?.try(&.as_a)
        "docs"
      end

      resp = agent.ask("Show the documents")
      resp.join

      resp.message.content.should eq("Done")

      # The mixed-type call was rejected with a precise message...
      agent.history[2].role.should eq(Agent::Role::Tool)
      agent.history[2].content.should eq(
        "Error validating arguments for tool 'show_documents': Field 'ids' expected array items of type 'string', got 'number' at index 1"
      )

      # ...and the valid retry reached the callback exactly once.
      callback_calls.should eq(1)
      received_ids.try(&.map(&.as_s)).should eq(["ok"])

      # History: user, assistant(tool_calls), tool(error),
      # assistant(tool_calls), tool(result), assistant(final)
      agent.history.size.should eq(6)
    ensure
      server.close
    end
  end

  it "does not reject null when the property schema declares no type" do
    # Compat valve: a property without a declared `type` (or an unknown
    # argument) accepts null — only declared types reject it.
    call_count = 0
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        ctx.request.body.try(&.gets_to_end)
        call_count += 1
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        if call_count == 1
          delta1 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "id" => "call_001", "type" => "function", "function" => {"name" => "store", "arguments" => ""}}]}, "index" => 0}]}.to_json
          delta2 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "function" => {"arguments" => %({"payload": null})}}]}, "index" => 0}]}.to_json
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "tool_calls"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}}.to_json
          ctx.response.puts "data: #{delta1}"
          ctx.response.flush
          ctx.response.puts "data: #{delta2}"
          ctx.response.flush
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        else
          reply = "Stored"
          final = {"choices" => [{"delta" => {"content" => reply}, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 20, "completion_tokens" => reply.size, "total_tokens" => 20 + reply.size}}.to_json
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        end
        ctx.response.close
      else
        ctx.response.status_code = 404
        ctx.response.puts "Not Found"
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        auto_execute_tools: true,
      )

      agent = Agent.new(config)

      callback_called = false
      received_payload : JSON::Any? = nil
      agent.register_tool("store", "Store a payload",
        parameters: Agent::JSONConverter.from({
          type:       "object",
          properties: {
            payload: {description: "anything"},
          },
        })
      ) do |args|
        callback_called = true
        received_payload = args["payload"]?
        "ok"
      end

      resp = agent.ask("Store it")
      resp.join

      callback_called.should be_true
      resp.message.content.should eq("Stored")
      received_payload.try(&.raw).should be_nil
    ensure
      server.close
    end
  end

  it "trims history with max_history" do
    call_count = 0
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        ctx.request.body.try(&.gets_to_end)
        call_count += 1
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        reply = "Response #{call_count}"
        reply.each_char do |ch|
          data = {"choices" => [{"delta" => {"content" => ch.to_s}, "index" => 0}]}.to_json
          ctx.response.puts "data: #{data}"
          ctx.response.flush
        end
        final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => reply.size, "total_tokens" => 10 + reply.size}}.to_json
        ctx.response.puts "data: #{final}"
        ctx.response.puts "data: [DONE]"
        ctx.response.flush
        ctx.response.close
      else
        ctx.response.status_code = 404
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        max_history: 1, # Keep at most 1 user+assistant pair
      )

      agent = Agent.new(config)
      agent.ask("First").join  # history: user, assistant (2 msgs)
      agent.ask("Second").join # history: user, assistant, user, assistant (4 msgs) -> trimmed to 2
      agent.ask("Third").join  # trimmed again

      # With max_history=1, only the last user+assistant pair should survive
      agent.history.size.should eq(2)
      agent.history[0].role.should eq(Agent::Role::User)
      agent.history[0].content.should eq("Third")
      agent.history[1].role.should eq(Agent::Role::Assistant)
      agent.history[1].content.should eq("Response 3")
    ensure
      server.close
    end
  end

  it "produces structured error on API non-2xx" do
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        ctx.response.status_code = 401
        ctx.response.puts "Unauthorized"
        ctx.response.close
      else
        ctx.response.status_code = 404
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "bad-key",
        api_endpoint: "http://localhost:#{port}",
      )

      agent = Agent.new(config)
      resp = agent.ask("Hello")
      resp.join
      resp.error?.should be_true
      resp.error.should be_a(Agent::ApiError)
      resp.error.as(Agent::ApiError).status_code.should eq(401)
    ensure
      server.close
    end
  end

  it "marks the tool results it builds for failed calls as errors" do
    agent = Agent.new(Agent::Config.new(api_key: "k"))
    agent.register_tool("boom", "fails", parameters: Agent::JSONConverter.from({type: "object", properties: {n: {type: "integer"}}, required: ["n"]})) do |_args|
      raise "no"
    end
    response = Agent::Response.new
    calls = [
      Agent::ToolCall.new("c1", "boom", %({"n":1})),
      Agent::ToolCall.new("c2", "boom", %({"n":"x"})),
      Agent::ToolCall.new("c3", "boom", "{not json"),
    ]
    results = agent.execute_registered_tools_for_spec(calls, response)
    results.map(&.error?).should eq([true, true, true])
    results[0].content.to_s.should contain("Error executing tool")
    results[1].content.to_s.should contain("Error validating arguments")
    results[2].content.to_s.should contain("invalid JSON")
  end

  describe "#trim_history!" do
    it "trims history without orphaning tool messages" do
      # A more complex mock server that:
      #   1st ask -> returns tool calls
      #   2nd ask -> returns plain text (tool results fed back)
      #   3rd ask -> triggers trimming (more than max_history turns)
      call_count = 0
      server = HTTP::Server.new do |ctx|
        if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
          ctx.request.body.try(&.gets_to_end)
          call_count += 1
          ctx.response.content_type = "text/event-stream"
          ctx.response.status_code = 200

          if call_count <= 2
            # First two calls: return tool calls
            delta1 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "id" => "call_001", "type" => "function", "function" => {"name" => "test_tool", "arguments" => ""}}]}, "index" => 0}]}.to_json
            delta2 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "function" => {"arguments" => %({"input":"test"})}}]}, "index" => 0}]}.to_json
            final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "tool_calls"}], "usage" => {"prompt_tokens" => 5, "completion_tokens" => 5, "total_tokens" => 10}}.to_json
            ctx.response.puts "data: #{delta1}"
            ctx.response.flush
            ctx.response.puts "data: #{delta2}"
            ctx.response.flush
            ctx.response.puts "data: #{final}"
            ctx.response.puts "data: [DONE]"
            ctx.response.flush
          else
            # Third call: return plain text (triggers trim after this)
            reply = "final"
            reply.each_char do |ch|
              data = {"choices" => [{"delta" => {"content" => ch.to_s}, "index" => 0}]}.to_json
              ctx.response.puts "data: #{data}"
              ctx.response.flush
            end
            final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => reply.size, "total_tokens" => 10 + reply.size}}.to_json
            ctx.response.puts "data: #{final}"
            ctx.response.puts "data: [DONE]"
            ctx.response.flush
          end
          ctx.response.close
        else
          ctx.response.status_code = 404
        end
      end

      address = server.bind_tcp(0)
      port = address.port
      ready = Channel(Nil).new
      spawn do
        ready.send(nil)
        server.listen
      end
      ready.receive

      begin
        config = Agent::Config.new(
          api_key: "test-key",
          api_endpoint: "http://localhost:#{port}",
          max_history: 1, # keep only 1 user+assistant pair
          auto_execute_tools: true,
        )

        agent = Agent.new(config)

        # Register the tool so auto-resolve kicks in
        agent.register_tool("test_tool", "A test tool",
          parameters: Agent::JSONConverter.from({
            type:       "object",
            properties: {
              input: {type: "string"},
            },
            required: ["input"],
          })
        ) do |args|
          "processed: #{args["input"]?.try(&.as_s)}"
        end

        # Round 1: user->tool_call->tool_result->final
        resp1 = agent.ask("First")
        resp1.join
        # History: user, assistant(tool_calls), tool, assistant(final) = 4

        # Round 2: another tool-call turn
        resp2 = agent.ask("Second")
        resp2.join
        # History grows: old 4 + new 4 = 8, then trimmed to max_history*2=2
        # The trim must not orphan the tool messages from the surviving turn

        # With max_history=1, only 2 user+assistant should survive
        # The survivor must be a complete turn with no orphaned tool messages
        agent.history.size.should eq(2)
        agent.history[0].role.should eq(Agent::Role::User)
        agent.history[0].content.should eq("Second")
        agent.history[1].role.should eq(Agent::Role::Assistant)
        agent.history[1].content.should eq("final")
        agent.history[1].has_tool_calls?.should be_false
      ensure
        server.close
      end
    end
  end

  it "drops whole turns, so a kept history starts at a user message" do
    with_mock_server do |port|
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        max_history: 2,
      )
      agent = Agent.new(config)
      tc = Agent::ToolCall.new("call_1", "lookup", "{}")
      # three turns, the first two with tool calls: 2 + 4 + 3 = 9 messages,
      # 6 of them user or assistant
      agent.load_history([
        Agent::Message.user("one"),
        Agent::Message.assistant("1"),
        Agent::Message.user("two"),
        Agent::Message.assistant(nil, tool_calls: [tc]),
        Agent::Message.tool_result(tc, "found"),
        Agent::Message.assistant("2"),
        Agent::Message.user("three"),
        Agent::Message.assistant(nil, tool_calls: [tc]),
        Agent::Message.tool_result(tc, "found again"),
      ])

      agent.ask("four").join

      # the oldest two turns are gone; the third, tool calls and all, stays with the new one
      agent.history.map(&.role).should eq([
        Agent::Role::User, Agent::Role::Assistant, Agent::Role::Tool,
        Agent::Role::User, Agent::Role::Assistant,
      ])
      agent.history[0].content.should eq("three")
      agent.history[3].content.should eq("four")
    end
  end

  it "handles concurrent #ask from multiple fibers" do
    with_mock_server do |port|
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
      )

      agent = Agent.new(config)
      responses = [] of Agent::Response
      fibers = 5
      channel = Channel(Nil).new(fibers)

      fibers.times do |i|
        spawn do
          resp = agent.ask("Concurrent #{i}")
          resp.join
          responses << resp
          channel.send(nil)
        end
      end

      # Wait for all fibers to complete
      fibers.times { channel.receive }

      responses.size.should eq(fibers)
      responses.each do |resp|
        resp.message.content.should_not be_nil
        resp.finished?.should be_true
      end

      # History should have 2*fibers entries (user + assistant per request)
      agent.history.size.should eq(fibers * 2)
    end
  end

  it "cancels an in-flight response" do
    with_mock_server do |port|
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
      )

      agent = Agent.new(config)
      resp = agent.ask("Hello")

      # Cancel immediately — the SSE stream should stop early
      resp.cancel

      # The response should complete with an error (CancelledError)
      resp.join
      resp.finished?.should be_true
      resp.error?.should be_true
      resp.error.should be_a(Agent::CancelledError)
    end
  end

  it "preserves the canceled turn in history after cancellation" do
    # A mock server that streams content slowly enough to be canceled mid-flight.
    call_count = 0
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        ctx.request.body.try(&.gets_to_end)
        call_count += 1
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        reply = call_count == 1 ? "Partial reply" : "Second reply"
        reply.each_char do |ch|
          data = {"choices" => [{"delta" => {"content" => ch.to_s}, "index" => 0}]}.to_json
          ctx.response.puts "data: #{data}"
          ctx.response.flush
          # Yield between chunks so the caller can cancel mid-stream.
          Fiber.yield
        end
        final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => reply.size, "total_tokens" => 10 + reply.size}}.to_json
        ctx.response.puts "data: #{final}"
        ctx.response.puts "data: [DONE]"
        ctx.response.flush
        ctx.response.close
      else
        ctx.response.status_code = 404
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
      )

      agent = Agent.new(config)

      # --- Turn 1: ask, stream a few chunks, then cancel mid-flight ---
      resp1 = agent.ask("Question 1")
      chunks = [] of String
      resp1.stream do |chunk|
        chunks << chunk.text if chunk.content?
        # After receiving a couple of chunks, cancel the in-flight stream.
        if chunks.size >= 2
          resp1.cancel
        end
      end
      resp1.join
      resp1.error?.should be_true
      resp1.error.should be_a(Agent::CancelledError)
      chunks.should_not be_empty

      # --- Turn 2: ask a follow-up question that completes normally ---
      resp2 = agent.ask("Question 2")
      resp2.stream { |_| } # drain
      resp2.join
      resp2.finished?.should be_true
      resp2.error?.should be_false, "second request failed after cancel: #{resp2.error.try(&.message)}"

      # --- Expectation: the canceled turn must NOT be lost from history ---
      # History should contain, at minimum:
      #   user("Question 1")           <- the canceled user message
      #   assistant(partial content)   <- whatever was streamed before cancel
      #   user("Question 2")
      #   assistant("Second reply")
      history = agent.history
      history.size.should be >= 4
      history[0].role.should eq(Agent::Role::User)
      history[0].content.should eq("Question 1")
      history[1].role.should eq(Agent::Role::Assistant)
      history[2].role.should eq(Agent::Role::User)
      history[2].content.should eq("Question 2")
      history[3].role.should eq(Agent::Role::Assistant)
      history[3].content.should eq("Second reply")
    ensure
      server.close
    end
  end

  describe "#close" do
    it "prevents further #ask calls" do
      with_mock_server do |port|
        config = Agent::Config.new(
          api_key: "test-key",
          api_endpoint: "http://localhost:#{port}",
        )

        agent = Agent.new(config)
        resp = agent.ask("Hello")
        resp.join

        agent.close
        expect_raises(Agent::ClosedError) do
          agent.ask("Another")
        end
      end
    end

    it "prevents #reset" do
      with_mock_server do |port|
        config = Agent::Config.new(
          api_key: "test-key",
          api_endpoint: "http://localhost:#{port}",
        )

        agent = Agent.new(config)
        agent.close
        expect_raises(Agent::ClosedError) do
          agent.reset
        end
      end
    end

    it "is safe to call multiple times" do
      agent = Agent.new(Agent::Config.new)
      agent.close
      agent.close # should not raise
    end

    it "finishes an in-flight request normally before closing" do
      with_mock_server do |port|
        config = Agent::Config.new(
          api_key: "test-key",
          api_endpoint: "http://localhost:#{port}",
        )

        agent = Agent.new(config)
        resp = agent.ask("Hello")
        # The fiber is already processing this request.
        # Close only prevents *future* requests.
        agent.close

        # The in-flight response should still complete.
        resp.join
        resp.message.content.should_not be_nil
        resp.finished?.should be_true
      end
    end
  end

  it "dump/load round-trip preserves session_id, cache_key, and history" do
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        call_count = (ctx.request.body.try(&.gets_to_end) || "").size # just to force reading
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        reply = "Response #{call_count}"
        reply.each_char do |ch|
          data = {"choices" => [{"delta" => {"content" => ch.to_s}, "index" => 0}]}.to_json
          ctx.response.puts "data: #{data}"
          ctx.response.flush
        end
        final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => reply.size, "total_tokens" => 10 + reply.size}}.to_json
        ctx.response.puts "data: #{final}"
        ctx.response.puts "data: [DONE]"
        ctx.response.flush
        ctx.response.close
      else
        ctx.response.status_code = 404
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
      )

      original = Agent.new(config)
      resp1 = original.ask("First message")
      resp1.join

      session_id = original.session_id
      cache_key = original.cache_key
      history_size = original.history.size

      # Dump to JSON string
      dump_str = original.dump
      dump_str.should be_a(String)

      # Load into a new agent with the same config
      restored = Agent.load(config, dump_str)

      # Verify restored fields
      restored.session_id.should eq(session_id)
      restored.cache_key.should eq(cache_key)
      restored.history.size.should eq(history_size)

      # Verify the loaded agent can continue the conversation
      resp2 = restored.ask("Continue")
      resp2.join
      resp2.message.content.should_not be_nil
      resp2.finished?.should be_true
    ensure
      server.close
    end
  end

  it "trims history with tool calls and max_history leaving no orphaned tool messages" do
    call_count = 0
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        ctx.request.body.try(&.gets_to_end)
        call_count += 1
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        if call_count <= 2
          # First two calls return tool calls
          delta1 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "id" => "call_#{call_count}", "type" => "function", "function" => {"name" => "my_tool", "arguments" => ""}}]}, "index" => 0}]}.to_json
          delta2 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "function" => {"arguments" => %({"input":"test"})}}]}, "index" => 0}]}.to_json
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "tool_calls"}], "usage" => {"prompt_tokens" => 5, "completion_tokens" => 5, "total_tokens" => 10}}.to_json
          ctx.response.puts "data: #{delta1}"
          ctx.response.flush
          ctx.response.puts "data: #{delta2}"
          ctx.response.flush
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        else
          # Third call returns plain text (triggers trim)
          reply = "final result"
          reply.each_char do |ch|
            data = {"choices" => [{"delta" => {"content" => ch.to_s}, "index" => 0}]}.to_json
            ctx.response.puts "data: #{data}"
            ctx.response.flush
          end
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => reply.size, "total_tokens" => 10 + reply.size}}.to_json
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        end
        ctx.response.close
      else
        ctx.response.status_code = 404
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        max_history: 1,
        auto_execute_tools: true,
      )

      agent = Agent.new(config)

      agent.register_tool("my_tool", "A test tool",
        parameters: Agent::JSONConverter.from({
          type:       "object",
          properties: {
            input: {type: "string"},
          },
          required: ["input"],
        })
      ) do |args|
        "processed: #{args["input"]?.try(&.as_s)}"
      end

      # First turn: user -> tool_calls -> tool_result -> assistant(final) = 4 messages
      resp1 = agent.ask("First")
      resp1.join

      # Second turn: same pattern, triggers trim after auto-resolve
      resp2 = agent.ask("Second")
      resp2.join

      # With max_history=1, only the last complete turn should survive (2 messages)
      agent.history.size.should eq(2)
      agent.history[0].role.should eq(Agent::Role::User)
      agent.history[0].content.should eq("Second")
      agent.history[1].role.should eq(Agent::Role::Assistant)
      agent.history[1].content.should eq("final result")
      agent.history[1].has_tool_calls?.should be_false
    ensure
      server.close
    end
  end

  it "register_tool from inside a callback works for in-fiber registration" do
    call_count = 0
    second_tool_called = false
    server = HTTP::Server.new do |ctx|
      if ctx.request.method == "POST" && ctx.request.path.includes?("/chat/completions")
        ctx.request.body.try(&.gets_to_end)
        call_count += 1
        ctx.response.content_type = "text/event-stream"
        ctx.response.status_code = 200

        if call_count == 1
          # First request: return tool call for first_tool
          delta1 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "id" => "call_001", "type" => "function", "function" => {"name" => "first_tool", "arguments" => ""}}]}, "index" => 0}]}.to_json
          delta2 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "function" => {"arguments" => %({"input":"hello"})}}]}, "index" => 0}]}.to_json
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "tool_calls"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}}.to_json
          ctx.response.puts "data: #{delta1}"
          ctx.response.flush
          ctx.response.puts "data: #{delta2}"
          ctx.response.flush
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        elsif call_count == 2
          # Second request: return tool call for second_tool
          delta1 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "id" => "call_002", "type" => "function", "function" => {"name" => "second_tool", "arguments" => ""}}]}, "index" => 0}]}.to_json
          delta2 = {"choices" => [{"delta" => {"tool_calls" => [{"index" => 0, "function" => {"arguments" => %({"input":"world"})}}]}, "index" => 0}]}.to_json
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "tool_calls"}], "usage" => {"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}}.to_json
          ctx.response.puts "data: #{delta1}"
          ctx.response.flush
          ctx.response.puts "data: #{delta2}"
          ctx.response.flush
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        else
          # Third request: return final text
          reply = "Both tools executed successfully"
          reply.each_char do |ch|
            data = {"choices" => [{"delta" => {"content" => ch.to_s}, "index" => 0}]}.to_json
            ctx.response.puts "data: #{data}"
            ctx.response.flush
          end
          final = {"choices" => [{"delta" => {} of String => JSON::Any, "index" => 0, "finish_reason" => "stop"}], "usage" => {"prompt_tokens" => 20, "completion_tokens" => reply.size, "total_tokens" => 20 + reply.size}}.to_json
          ctx.response.puts "data: #{final}"
          ctx.response.puts "data: [DONE]"
          ctx.response.flush
        end
        ctx.response.close
      else
        ctx.response.status_code = 404
      end
    end

    address = server.bind_tcp(0)
    port = address.port
    ready = Channel(Nil).new
    spawn do
      ready.send(nil)
      server.listen
    end
    ready.receive

    begin
      config = Agent::Config.new(
        api_key: "test-key",
        api_endpoint: "http://localhost:#{port}",
        auto_execute_tools: true,
      )

      agent = Agent.new(config)

      # Register first_tool — its callback will register second_tool
      agent.register_tool("first_tool", "First tool",
        parameters: Agent::JSONConverter.from({
          type:       "object",
          properties: {
            input: {type: "string"},
          },
          required: ["input"],
        })
      ) do |args|
        # Register second_tool from inside the callback (in-fiber registration path)
        agent.register_tool("second_tool", "Second tool",
          parameters: Agent::JSONConverter.from({
            type:       "object",
            properties: {
              input: {type: "string"},
            },
            required: ["input"],
          })
        ) do |args2|
          second_tool_called = true
          "second_result: #{args2["input"]?.try(&.as_s)}"
        end
        "first_result: #{args["input"]?.try(&.as_s)}"
      end

      resp = agent.ask("Run both tools")
      resp.join

      resp.message.content.should eq("Both tools executed successfully")
      second_tool_called.should be_true

      # History should have: user, assistant(tool_calls), tool, assistant(tool_calls), tool, assistant(final) = 6
      agent.history.size.should eq(6)
      agent.history[0].role.should eq(Agent::Role::User)
      agent.history[1].role.should eq(Agent::Role::Assistant)
      agent.history[1].has_tool_calls?.should be_true
      agent.history[2].role.should eq(Agent::Role::Tool)
      agent.history[2].content.to_s.should contain("first_result")
      agent.history[3].role.should eq(Agent::Role::Assistant)
      agent.history[3].has_tool_calls?.should be_true
      agent.history[4].role.should eq(Agent::Role::Tool)
      agent.history[4].content.to_s.should contain("second_result")
      agent.history[5].role.should eq(Agent::Role::Assistant)
      agent.history[5].content.should eq("Both tools executed successfully")
    ensure
      server.close
    end
  end

  describe "Agent.load error handling" do
    config = Agent::Config.new(api_key: "test-key")

    it "raises SessionLoadError for non-JSON input" do
      expect_raises(Agent::SessionLoadError, "not a valid JSON object") do
        Agent.load(config, "not json at all")
      end
    end

    it "raises SessionLoadError for JSON non-object (array)" do
      expect_raises(Agent::SessionLoadError, "not a valid JSON object") do
        Agent.load(config, "[]")
      end
    end

    it "raises SessionLoadError for JSON literal (number)" do
      expect_raises(Agent::SessionLoadError, "not a valid JSON object") do
        Agent.load(config, "42")
      end
    end

    it "raises SessionLoadError when session_id is missing" do
      bad = %({"cache_key":"x","history":[]})
      expect_raises(Agent::SessionLoadError, "session_id") do
        Agent.load(config, bad)
      end
    end

    it "raises SessionLoadError when session_id is not a string" do
      bad = %({"session_id":123,"cache_key":"x","history":[]})
      expect_raises(Agent::SessionLoadError, "session_id") do
        Agent.load(config, bad)
      end
    end

    it "raises SessionLoadError when cache_key is missing" do
      bad = %({"session_id":"s1","history":[]})
      expect_raises(Agent::SessionLoadError, "cache_key") do
        Agent.load(config, bad)
      end
    end

    it "raises SessionLoadError when history is missing" do
      bad = %({"session_id":"s1","cache_key":"k1"})
      expect_raises(Agent::SessionLoadError, "history") do
        Agent.load(config, bad)
      end
    end

    it "raises SessionLoadError when history is not an array" do
      bad = %({"session_id":"s1","cache_key":"k1","history":"oops"})
      expect_raises(Agent::SessionLoadError, "history") do
        Agent.load(config, bad)
      end
    end

    it "raises SessionLoadError when a message has no role" do
      bad = %({"session_id":"s1","cache_key":"k1","history":[{"content":"hi"}]})
      expect_raises(Agent::SessionLoadError, "history") do
        Agent.load(config, bad)
      end
    end

    it "raises SessionLoadError when a message has an invalid role" do
      bad = %({"session_id":"s1","cache_key":"k1","history":[{"role":"superuser","content":"hi"}]})
      expect_raises(Agent::SessionLoadError, "history") do
        Agent.load(config, bad)
      end
    end

    it "raises SessionLoadError when enabled_tools is not an array" do
      bad = %({"session_id":"s1","cache_key":"k1","history":[],"enabled_tools":"string"})
      expect_raises(Agent::SessionLoadError, "enabled_tools") do
        Agent.load(config, bad)
      end
    end

    it "raises SessionLoadError when enabled_tools has non-string elements" do
      bad = %({"session_id":"s1","cache_key":"k1","history":[],"enabled_tools":[123]})
      expect_raises(Agent::SessionLoadError, "enabled_tools") do
        Agent.load(config, bad)
      end
    end

    it "accepts valid JSON::Any input" do
      raw = %({"session_id":"s1","cache_key":"k1","history":[]})
      any = JSON.parse(raw)
      agent = Agent.load(config, any)
      agent.session_id.should eq("s1")
      agent.cache_key.should eq("k1")
      agent.history.should be_empty
    ensure
      agent.try(&.close)
    end

    it "still raises SessionLoadError when data is corrupt after creating the agent" do
      # The agent is created first (fiber spawned), then data is parsed.
      expect_raises(Agent::SessionLoadError) do
        Agent.load(config, "invalid json")
      end
    end
  end
end

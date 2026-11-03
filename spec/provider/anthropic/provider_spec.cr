require "../../spec_helper"
require "../../support/fake_model_server"

private ENDPOINT = "https://api.anthropic.com"

private def anthropic(endpoint = ENDPOINT, effort : Agent::ReasoningEffort? = nil, **opts)
  config = Agent::Config.new(api_key: "sk-test", api_endpoint: endpoint, model: "claude-opus-5-5", reasoning_effort: effort)
  Agent::Provider::Anthropic.new(config, **opts)
end

private def body_of(provider, messages, tools = nil, options = Agent::RequestOptions.new)
  JSON.parse(provider.build_request(messages, tools, options)[:body])
end

private def finding_schema
  {
    "type"       => JSON::Any.new("object"),
    "properties" => JSON::Any.new({"id" => JSON::Any.new({"type" => JSON::Any.new("string")})}),
    "required"   => JSON::Any.new([JSON::Any.new("id")]),
  }
end

private def register_finding_tool(agent : Agent, calls : Array(Hash(String, JSON::Any)))
  agent.register_tool("get_finding", "the finding by id", finding_schema) do |args|
    calls << args
    {id: args["id"].as_s, rule: "shell_in_container"}.to_json
  end
end

describe Agent::Provider::Anthropic do
  describe "#build_request" do
    it "posts to the messages path with the required headers" do
      req = anthropic.build_request([Agent::Message.user("hi")], nil)
      req[:path].should eq "/v1/messages"
      req[:headers]["x-api-key"].should eq "sk-test"
      req[:headers]["anthropic-version"].should eq Agent::Provider::Anthropic::API_VERSION
      req[:headers]["anthropic-beta"].should eq Agent::Provider::Anthropic::BINDING_BETA
    end

    it "keeps a gateway's path prefix, and takes an endpoint ending in /v1 as the API root" do
      anthropic("https://proxy.example/anthropic/").path.should eq "/anthropic/v1/messages"
      anthropic("https://api.anthropic.com/v1").path.should eq "/v1/messages"
    end

    it "merges anthropic-beta from extra_headers with its own" do
      config = Agent::Config.new(api_endpoint: ENDPOINT, model: "m", extra_headers: {"anthropic-beta" => "a-1, b-2", "x-trace" => "t"})
      headers = Agent::Provider::Anthropic.new(config).build_request([Agent::Message.user("hi")], nil)[:headers]
      headers["anthropic-beta"].should eq "a-1,b-2,#{Agent::Provider::Anthropic::BINDING_BETA}"
      headers["x-trace"].should eq "t"
    end

    it "turns the shard's flat history into system, turns and tool blocks" do
      call = Agent::ToolCall.new(id: "toolu_1", name: "get_finding", arguments: %({"id":"f1"}))
      call2 = Agent::ToolCall.new(id: "toolu_2", name: "list_events", arguments: %({"finding":"f1"}))
      messages = [
        Agent::Message.system("You review findings."),
        Agent::Message.user("Review f1"),
        Agent::Message.assistant("I'll look.", tool_calls: [call, call2]),
        Agent::Message.tool_result(call, %({"id":"f1"})),
        Agent::Message.tool_result(call2, "the tool failed", is_error: true),
      ]
      tool = Agent::Tool.new(Agent::Tool::FunctionDef.new("get_finding", "the finding", {"type" => JSON::Any.new("object")}))
      body = body_of(anthropic(effort: :high), messages, [tool])

      body["model"].should eq "claude-opus-5-5"
      body["stream"].as_bool.should be_true
      body["max_tokens"].should eq Agent::Provider::Anthropic::DEFAULT_MAX_TOKENS
      body["output_config"]["effort"].should eq "high"
      body.as_h.has_key?("temperature").should be_false
      body["system"].as_a.size.should eq 1
      body["system"][0]["text"].should eq "You review findings."
      body["system"][0]["cache_control"]["type"].should eq "ephemeral"
      body["tools"][0]["name"].should eq "get_finding"
      body["tools"][0]["input_schema"]["type"].should eq "object"
      body["tools"][0]["eager_input_streaming"].as_bool.should be_true

      turns = body["messages"].as_a
      turns.size.should eq 3
      turns[0]["role"].should eq "user"
      turns[0]["content"][0]["text"].should eq "Review f1"
      turns[0]["content"][0]["cache_control"]?.should be_nil
      turns[1]["role"].should eq "assistant"
      turns[1]["content"][0]["text"].should eq "I'll look."
      turns[1]["content"][1]["type"].should eq "tool_use"
      turns[1]["content"][1]["input"]["id"].should eq "f1"
      # both results in one user turn, the failed one flagged
      turns[2]["role"].should eq "user"
      turns[2]["content"].as_a.size.should eq 2
      turns[2]["content"][0]["tool_use_id"].should eq "toolu_1"
      turns[2]["content"][0]["is_error"]?.should be_nil
      turns[2]["content"][1]["is_error"].as_bool.should be_true
      # the transcript's breakpoint sits on the last block of the last user turn only
      turns[2]["content"][1]["cache_control"]["type"].should eq "ephemeral"
      turns[2]["content"][0]["cache_control"]?.should be_nil
    end

    it "sends a user message as a text block carrying the cache breakpoint" do
      block = body_of(anthropic, [Agent::Message.user("Review f1")])["messages"][0]["content"][0]
      block["type"].should eq "text"
      block["text"].should eq "Review f1"
      block["cache_control"]["type"].should eq "ephemeral"
    end

    it "joins tool results, a task's result and the next user message into one turn" do
      call = Agent::ToolCall.new(id: "toolu_1", name: "wait", arguments: "{}")
      messages = [
        Agent::Message.user("wait for it"),
        Agent::Message.assistant(nil, tool_calls: [call]),
        Agent::Message.tool_result(call, "Started background task t-1"),
        Agent::Message.user("<task_result id=\"t-1\" status=\"done\">it happened</task_result>", source: :task),
        Agent::Message.user("great, continue"),
      ]
      turns = body_of(anthropic, messages)["messages"].as_a
      turns.map(&.["role"].as_s).should eq %w[user assistant user]
      blocks = turns[2]["content"].as_a
      blocks.map(&.["type"].as_s).should eq %w[tool_result text text]
      blocks[1]["text"].as_s.should contain "it happened"
      blocks[2]["text"].should eq "great, continue"
      blocks[2]["cache_control"]["type"].should eq "ephemeral"
      blocks[1]["cache_control"]?.should be_nil
    end

    it "moves a later system message into the system prompt" do
      messages = [
        Agent::Message.system("Base."),
        Agent::Message.user("one"),
        Agent::Message.assistant("ok"),
        Agent::Message.user("two"),
        Agent::Message.system("Now be brief."),
      ]
      body = body_of(anthropic, messages)
      body["system"].as_a.map(&.["text"]).should eq ["Base.", "Now be brief."]
      body["system"][0]["cache_control"]?.should be_nil
      body["system"][1]["cache_control"]["type"].should eq "ephemeral"
      body["messages"].as_a.map(&.["role"].as_s).should eq ["user", "assistant", "user"]
      body["messages"][2]["content"][0]["cache_control"]["type"].should eq "ephemeral"
    end

    it "asks for adaptive thinking with a readable summary, and drops blocks a trim invalidated" do
      thinking = body_of(anthropic, [Agent::Message.user("x")])["thinking"]
      thinking["type"].should eq "adaptive"
      thinking["display"].should eq "summarized"
      thinking["block_binding"]["prefix_mismatch_behavior"].should eq "drop_block"

      plain = anthropic(thinking: false)
      plain.build_request([Agent::Message.user("x")], nil)[:headers]["anthropic-beta"]?.should be_nil
      body_of(plain, [Agent::Message.user("x")]).as_h.has_key?("thinking").should be_false

      bare = body_of(anthropic(display: nil, prefix_mismatch: nil), [Agent::Message.user("x")])["thinking"]
      bare.should eq JSON.parse(%({"type":"adaptive"}))
    end

    it "maps the shard's effort levels, and lets the request override the config" do
      {Agent::ReasoningEffort::Off => "low", Agent::ReasoningEffort::Minimal => "low",
       Agent::ReasoningEffort::Medium => "medium", Agent::ReasoningEffort::Max => "max"}.each do |effort, wire|
        body_of(anthropic(effort: effort), [Agent::Message.user("x")])["output_config"]["effort"].should eq wire
      end
      body_of(anthropic, [Agent::Message.user("x")]).as_h.has_key?("output_config").should be_false
      body = body_of(anthropic(effort: :low), [Agent::Message.user("x")], options: Agent::RequestOptions.new(reasoning_effort: :xhigh))
      body["output_config"]["effort"].should eq "xhigh"
    end

    it "replays an assistant reply's raw blocks only to the endpoint they came from" do
      blocks = JSON.parse([
        {type: "thinking", thinking: "hmm", signature: "sig-xyz"},
        {type: "tool_use", id: "t1", name: "get_finding", input: {id: "f1"}},
      ].to_json).as_a
      data = Agent::Provider::Anthropic.replay_data(ENDPOINT, blocks)
      kept = Agent::Message.assistant(nil, tool_calls: [Agent::ToolCall.new("t1", "get_finding", %({"id":"f1"}))], reasoning: "hmm", provider_data: data)

      sent = body_of(anthropic, [Agent::Message.user("go"), kept])["messages"][1]["content"].as_a
      sent.should eq blocks

      other = body_of(anthropic("https://gateway.example"), [Agent::Message.user("go"), kept])["messages"][1]["content"].as_a
      other.map(&.["type"].as_s).should eq ["tool_use"]
    end

    it "sends only the text of a reply a cancel cut short" do
      cut = Agent::Message.assistant("partial answer", reasoning: "half a thought")
      blocks = body_of(anthropic, [Agent::Message.user("go"), cut])["messages"][1]["content"].as_a
      blocks.should eq [JSON.parse(%({"type":"text","text":"partial answer"}))]
    end

    it "sends images and PDFs, as base64 or by url" do
      # (the last block also carries the cache breakpoint)
      parts = [
        Agent::ContentPart.new(text: "Look"),
        Agent::ContentPart.from_path("data:image/png;base64,iVBORw0K"),
        Agent::ContentPart.from_path("https://example.com/a.png"),
        Agent::ContentPart.from_path("data:application/pdf;base64,JVBERi0"),
      ]
      message = Agent::Message.new(role: Agent::Role::User, content_parts: parts)
      content = body_of(anthropic, [message])["messages"][0]["content"].as_a
      content.map(&.["type"].as_s).should eq ["text", "image", "image", "document"]
      content[3]["source"]["media_type"].should eq "application/pdf"
      content[1]["source"].should eq JSON.parse(%({"type":"base64","media_type":"image/png","data":"iVBORw0K"}))
      content[2]["source"]["url"].should eq "https://example.com/a.png"
    end

    it "never sends an empty user turn" do
      audio = Agent::ContentPart.new(type: Agent::ContentPart::PartType::InputAudio, data: "AAA=", mime_type: "audio/wav")
      message = Agent::Message.new(role: Agent::Role::User, content_parts: [audio])
      body_of(anthropic, [message])["messages"][0]["content"][0]["text"].should eq "(no content)"
    end

    it "gives an empty tool schema a shape the api accepts" do
      tool = Agent::Tool.new(Agent::Tool::FunctionDef.new("ping"))
      body_of(anthropic, [Agent::Message.user("x")], [tool])["tools"][0]["input_schema"]["type"].should eq "object"
    end

    it "refuses a bad endpoint, OpenAI's default endpoint, or a bad option" do
      expect_raises(ArgumentError) { anthropic("nonsense") }
      expect_raises(ArgumentError, /OpenAI/) { Agent::Provider::Anthropic.new(Agent::Config.new(model: "m")) }
      expect_raises(ArgumentError) { anthropic(display: "verbose") }
      expect_raises(ArgumentError) { anthropic(prefix_mismatch: "ignore") }
    end
  end

  describe "#parse_stream" do
    it "assembles text, a tool call split over deltas, the thinking block and usage" do
      response = Agent::Response.new
      message, usage, stop = anthropic.parse_stream(IO::Memory.new(anthropic_tool_call_stream), response, -> { false })
      response.finish(message, usage, stop)
      chunks = [] of Agent::Response::Chunk
      response.stream { |c| chunks << c }

      stop.should eq "tool_use"
      message.content.should eq "Looking at the finding."
      calls = message.tool_calls.as(Array(Agent::ToolCall))
      calls.size.should eq 1
      calls[0].id.should eq "toolu_1"
      calls[0].name.should eq "get_finding"
      JSON.parse(calls[0].arguments)["id"].should eq "f1"
      # cache reads count as prompt tokens, and are kept beside them
      usage.prompt_tokens.should eq 1240
      usage.completion_tokens.should eq 57
      usage.cached_tokens.should eq 1200

      blocks = Agent::Provider::Anthropic.blocks(message).as(Array(JSON::Any))
      blocks[0]["type"].should eq "thinking"
      blocks[0]["signature"].should eq "sig-abc"
      blocks[2]["type"].should eq "tool_use"
      blocks[2]["input"]["id"].should eq "f1"

      chunks.map(&.kind).should contain(Agent::Response::ChunkKind::ToolCallName)
      chunks.select(&.content?).join(&.text).should eq "Looking at the finding."
      chunks.select(&.tool_call_args?).join(&.text).should eq %({"id":"f1"})
    end

    it "parses a recorded stream: summarized thinking, then two parallel tool calls" do
      # Recorded from claude-opus-5-5 at effort high (Nov 2026).
      stream = File.read("#{__DIR__}/../../fixtures/anthropic/thinking_parallel_tools.sse")
      message, usage, stop = anthropic.parse_stream(IO::Memory.new(stream), Agent::Response.new, -> { false })

      stop.should eq "tool_use"
      message.reasoning.to_s.should_not be_empty
      message.content.to_s.should_not be_empty
      calls = message.tool_calls.as(Array(Agent::ToolCall))
      calls.map(&.name).should eq ["get_time", "get_time"]
      calls.map { |c| JSON.parse(c.arguments)["city"].as_s }.should eq ["Tokyo", "Lima"]
      calls.map(&.id).uniq!.size.should eq 2
      usage.prompt_tokens.should eq 395
      usage.completion_tokens.as(Int32).should be > 0

      blocks = Agent::Provider::Anthropic.blocks(message).as(Array(JSON::Any))
      blocks.map(&.["type"].as_s).should eq ["thinking", "text", "tool_use", "tool_use"]
      blocks[0]["thinking"].should eq message.reasoning
      blocks[0]["signature"].as_s.should_not be_empty

      # the next request sends the blocks back exactly as they came
      replayed = body_of(anthropic, [Agent::Message.user("go"), message])["messages"][1]["content"].as_a
      replayed.should eq blocks
    end

    it "puts summarized thinking in reasoning and keeps redacted and server-tool blocks whole" do
      stream = sse([
        {type: "message_start", message: {usage: {input_tokens: 5}}},
        {type: "content_block_start", index: 0, content_block: {type: "thinking", thinking: ""}},
        {type: "content_block_delta", index: 0, delta: {type: "thinking_delta", thinking: "Check the "}},
        {type: "content_block_delta", index: 0, delta: {type: "thinking_delta", thinking: "docs."}},
        {type: "content_block_delta", index: 0, delta: {type: "signature_delta", signature: "s1"}},
        {type: "content_block_start", index: 1, content_block: {type: "redacted_thinking", data: "ENC"}},
        {type: "content_block_start", index: 2, content_block: {type: "server_tool_use", id: "srv_1", name: "web_search", input: {} of String => String}},
        {type: "content_block_delta", index: 2, delta: {type: "input_json_delta", partial_json: %({"query":"x"})}},
        {type: "content_block_start", index: 3, content_block: {type: "web_search_tool_result", tool_use_id: "srv_1", content: [{type: "web_search_result", url: "https://e.x"}]}},
        {type: "content_block_start", index: 4, content_block: {type: "text", text: ""}},
        {type: "content_block_delta", index: 4, delta: {type: "text_delta", text: "Found it."}},
        {type: "message_delta", delta: {stop_reason: "end_turn"}, usage: {output_tokens: 9}},
        {type: "message_stop"},
      ])
      message, _, _ = anthropic.parse_stream(IO::Memory.new(stream), Agent::Response.new, -> { false })
      message.reasoning.should eq "Check the docs."
      message.content.should eq "Found it."
      message.tool_calls.should be_nil
      blocks = Agent::Provider::Anthropic.blocks(message).as(Array(JSON::Any))
      blocks.map(&.["type"].as_s).should eq ["thinking", "redacted_thinking", "server_tool_use", "web_search_tool_result", "text"]
      blocks[1]["data"].should eq "ENC"
      blocks[2]["input"]["query"].should eq "x"
      blocks[3]["content"][0]["url"].should eq "https://e.x"
    end

    it "raises a retryable stream error on an overloaded error event" do
      stream = sse([{type: "error", error: {type: "overloaded_error", message: "Overloaded"}}])
      ex = expect_raises(Agent::StreamError) do
        anthropic.parse_stream(IO::Memory.new(stream), Agent::Response.new, -> { false })
      end
      ex.code.should eq "overloaded_error"
      ex.retryable?.should be_true
    end

    it "refuses a stream cut off before message_stop, but not a cancelled one" do
      stream = sse([
        {type: "message_start", message: {usage: {input_tokens: 5}}},
        {type: "content_block_start", index: 0, content_block: {type: "text", text: ""}},
        {type: "content_block_delta", index: 0, delta: {type: "text_delta", text: "Half a rep"}},
      ])
      ex = expect_raises(Agent::StreamError) do
        anthropic.parse_stream(IO::Memory.new(stream), Agent::Response.new, -> { false })
      end
      ex.retryable?.should be_true
      message, _, _ = anthropic.parse_stream(IO::Memory.new(stream), Agent::Response.new, -> { true })
      message.role.should eq Agent::Role::Assistant
    end

    it "leaves content nil when the turn is only a tool call" do
      stream = sse([
        {type: "message_start", message: {usage: {input_tokens: 5}}},
        {type: "content_block_start", index: 0, content_block: {type: "tool_use", id: "toolu_9", name: "conclude", input: {} of String => String}},
        {type: "content_block_stop", index: 0},
        {type: "message_delta", delta: {stop_reason: "tool_use"}, usage: {output_tokens: 3}},
        {type: "message_stop"},
      ])
      message, _, _ = anthropic.parse_stream(IO::Memory.new(stream), Agent::Response.new, -> { false })
      message.content.should be_nil
      message.tool_calls.as(Array(Agent::ToolCall))[0].arguments.should eq "{}"
    end
  end

  describe "the agent loop" do
    it "runs a tool call round trip, replaying the signed thinking block" do
      fake = FakeModelServer.new([anthropic_tool_call_stream, anthropic_final_stream("It is a shell in a container.")])
      config = Agent::Config.new(api_key: "sk-test", api_endpoint: fake.url, model: "claude-opus-5-5", system_prompt: "You review findings.")
      agent = Agent.new(config, provider: Agent::Provider::Anthropic.new(config))
      calls = [] of Hash(String, JSON::Any)
      register_finding_tool(agent, calls)
      begin
        response = agent.ask("Review f1")
        response.join
        response.error.should be_nil
        response.message.content.should eq "It is a shell in a container."
        response.metadata.prompt_tokens.should eq 1240 + 90
        calls.size.should eq 1
        calls[0]["id"].should eq "f1"

        fake.requests.size.should eq 2
        second = fake.requests[1][:body]
        second["system"][0]["text"].should eq "You review findings."
        turns = second["messages"].as_a
        # user, assistant (signed thinking replayed), tool result
        turns.size.should eq 3
        turns[1]["content"][0]["type"].should eq "thinking"
        turns[1]["content"][0]["signature"].should eq "sig-abc"
        turns[2]["content"][0]["type"].should eq "tool_result"
        turns[2]["content"][0]["tool_use_id"].should eq "toolu_1"
        JSON.parse(turns[2]["content"][0]["content"].as_s)["rule"].should eq "shell_in_container"
      ensure
        agent.close
        fake.close
      end
    end

    it "does not run the tool calls of a reply cut off by max_tokens" do
      cut = sse([
        {type: "message_start", message: {usage: {input_tokens: 5}}},
        {type: "content_block_start", index: 0, content_block: {type: "tool_use", id: "toolu_1", name: "get_finding", input: {} of String => String}},
        {type: "content_block_delta", index: 0, delta: {type: "input_json_delta", partial_json: %({"id":"f)}},
        {type: "message_delta", delta: {stop_reason: "max_tokens"}, usage: {output_tokens: 3}},
        {type: "message_stop"},
      ])
      fake = FakeModelServer.new([cut])
      config = Agent::Config.new(api_key: "sk-test", api_endpoint: fake.url, model: "claude-opus-5-5", max_retries: 0)
      agent = Agent.new(config, provider: Agent::Provider::Anthropic.new(config))
      calls = [] of Hash(String, JSON::Any)
      register_finding_tool(agent, calls)
      begin
        response = agent.ask("Review f1")
        response.join
        error = response.error.as(Agent::StreamError)
        error.code.should eq "max_tokens"
        calls.should be_empty
        agent.history.should be_empty
      ensure
        agent.close
        fake.close
      end
    end

    it "survives a dump and load with the raw blocks intact" do
      fake = FakeModelServer.new([anthropic_tool_call_stream, anthropic_final_stream("done")])
      config = Agent::Config.new(api_key: "sk-test", api_endpoint: fake.url, model: "claude-opus-5-5")
      agent = Agent.new(config, provider: Agent::Provider::Anthropic.new(config))
      register_finding_tool(agent, [] of Hash(String, JSON::Any))
      begin
        agent.ask("Review f1").join
        restored = Agent.load(config, agent.dump, provider: Agent::Provider::Anthropic.new(config))
        assistant = restored.history.find!(&.role.assistant?)
        Agent::Provider::Anthropic.blocks(assistant).as(Array(JSON::Any))[0]["signature"].should eq "sig-abc"
        restored.close
      ensure
        agent.close
        fake.close
      end
    end
  end
end

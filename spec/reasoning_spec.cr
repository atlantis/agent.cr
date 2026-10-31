require "./spec_helper"
require "http/server"

private def parse_stream(lines : Array(String), endpoint = "https://openrouter.ai/api/v1") : {Agent::Message, Agent::Usage}
  body = lines.map { |l| "data: #{l}\n" }.join + "data: [DONE]\n"
  msg, usage, _ = Agent::Provider::OpenAI::StreamParser.parse(IO::Memory.new(body), Agent::Response.new, -> { false }, endpoint: endpoint, model: "m")
  {msg, usage}
end

private def build_body(messages, endpoint = "https://openrouter.ai/api/v1", **opts)
  Agent::Provider::OpenAI::RequestBody.build(messages, nil, "m", nil, nil, nil, **opts, endpoint: endpoint)
end

private def details_delta(*items) : String
  {choices: [{index: 0, delta: {reasoning_details: items.to_a}}]}.to_json
end

class Agent::Spec::HighEffort < Agent::Handler
  def handle(ctx : Agent::TurnContext, next_proc : Agent::TurnContext -> {Agent::Message, Agent::Usage, String?}) : {Agent::Message, Agent::Usage, String?}
    ctx.options = ctx.options.copy_with(reasoning_effort: Agent::ReasoningEffort::High)
    next_proc.call(ctx)
  end
end

# One-shot SSE server: writes `lines`, then `trailer` heartbeats 50 ms apart.
private def with_sse(lines : Array(String), trailer = 0, &)
  requests = [] of JSON::Any
  server = HTTP::Server.new do |ctx|
    requests << JSON.parse(ctx.request.body.try(&.gets_to_end) || "{}")
    ctx.response.content_type = "text/event-stream"
    lines.each { |l| ctx.response.puts "data: #{l}"; ctx.response.flush }
    trailer.times { sleep 50.milliseconds; ctx.response.puts ": keepalive"; ctx.response.flush }
    ctx.response.puts "data: [DONE]"
    ctx.response.close
  end
  port = server.bind_tcp(0).port
  spawn { server.listen }
  Fiber.yield
  begin
    yield "http://127.0.0.1:#{port}/v1", requests
  ensure
    server.close
  end
end

private def details(msg : Agent::Message) : Array(JSON::Any)
  msg.provider_data.try(&.["openai"]?).try(&.["reasoning_details"]?).try(&.as_a?) || fail("no reasoning_details")
end

describe "Reasoning" do
  describe "reasoning_details in the stream" do
    it "joins a block's fragments and keeps its signature" do
      msg, _ = parse_stream([
        details_delta({type: "reasoning.text", text: "Let me ", index: 0, format: "anthropic-claude-v1"}),
        details_delta({type: "reasoning.text", text: "think.", index: 0}),
        details_delta({type: "reasoning.text", text: "", signature: "sig-abc", index: 0}),
      ])
      blocks = details(msg)
      blocks.size.should eq(1)
      blocks[0]["text"].should eq("Let me think.")
      blocks[0]["signature"].should eq("sig-abc")
      blocks[0]["format"].should eq("anthropic-claude-v1")
      msg.reasoning.should eq("Let me think.")
    end

    it "keeps distinct blocks in the order they came, encrypted ones included" do
      msg, _ = parse_stream([
        details_delta({type: "reasoning.text", text: "a", index: 0}),
        details_delta({type: "reasoning.encrypted", data: "ENC1", index: 1}),
        details_delta({type: "reasoning.text", text: "b", index: 2}),
      ])
      details(msg).map(&.["type"].as_s).should eq(["reasoning.text", "reasoning.encrypted", "reasoning.text"])
      details(msg)[1]["data"].should eq("ENC1")
    end

    it "separates blocks with different ids even at the same index" do
      msg, _ = parse_stream([
        details_delta({type: "reasoning.summary", summary: "one", id: "r1", index: 0}),
        details_delta({type: "reasoning.summary", summary: "two", id: "r2", index: 0}),
      ])
      details(msg).map(&.["summary"].as_s).should eq(["one", "two"])
    end

    it "leaves provider_data empty when the stream has no reasoning_details or replayed text" do
      msg, _ = parse_stream([{choices: [{index: 0, delta: {reasoning: "hmm", content: "hi"}}]}.to_json])
      msg.reasoning.should eq("hmm")
      msg.provider_data.should be_nil
    end
  end

  it "reads reasoning_tokens from completion_tokens_details" do
    _, usage = parse_stream([{
      choices: [{index: 0, delta: {content: "hi"}, finish_reason: "stop"}],
      usage:   {prompt_tokens: 10, completion_tokens: 50, total_tokens: 60, completion_tokens_details: {reasoning_tokens: 42}},
    }.to_json])
    usage.reasoning_tokens.should eq(42)
    (usage + usage).reasoning_tokens.should eq(84)
  end

  describe "request body" do
    it "maps reasoning_effort to OpenAI's field, and to OpenRouter's reasoning object" do
      openai = build_body([Agent::Message.user("hi")], reasoning_effort: Agent::ReasoningEffort::Max)
      openai["reasoning_effort"].should eq("xhigh")
      openai.has_key?("reasoning").should be_false

      openrouter = build_body([Agent::Message.user("hi")], reasoning_effort: Agent::ReasoningEffort::Max, openrouter: true)
      openrouter["reasoning"]["effort"].should eq("max")
      openrouter.has_key?("reasoning_effort").should be_false

      build_body([Agent::Message.user("hi")], reasoning_effort: Agent::ReasoningEffort::Off)["reasoning_effort"].should eq("none")
    end

    it "adds extra_body fields, never replacing its own" do
      body = build_body([Agent::Message.user("hi")], extra_body: {
        "chat_template_kwargs" => JSON::Any.new({"enable_thinking" => JSON::Any.new(true)}),
        "model"                => JSON::Any.new("other"),
      })
      body["chat_template_kwargs"]["enable_thinking"].should be_true
      body["model"].should eq("m")
    end

    it "sends no reasoning fields by default" do
      body = build_body([Agent::Message.user("hi")])
      body.has_key?("reasoning_effort").should be_false
      body.has_key?("reasoning").should be_false
    end

    it "replays reasoning_details on assistant messages, only to the endpoint that produced them" do
      blocks = JSON.parse(%([{"type":"reasoning.encrypted","data":"ENC1","index":0}]))
      data = Agent::Provider::OpenAI.replay_data("https://openrouter.ai/api/v1", "m", {"reasoning_details" => blocks})
      history = [
        Agent::Message.user("hi"),
        Agent::Message.assistant("ok", provider_data: data),
        Agent::Message.new(role: Agent::Role::User, content: "again", provider_data: data),
      ]

      messages = build_body(history)["messages"].as_a
      messages[1]["reasoning_details"].should eq(blocks)
      messages[1].as_h.has_key?("provider_data").should be_false
      messages[2].as_h.has_key?("reasoning_details").should be_false

      other = build_body(history, endpoint: "https://api.openai.com/v1")["messages"].as_a
      other[1].as_h.has_key?("reasoning_details").should be_false
    end

    it "ignores provider_data written by another provider" do
      data = JSON.parse(%({"anthropic":{"blocks":[{"type":"thinking"}]}}))
      body = build_body([Agent::Message.user("hi"), Agent::Message.assistant("ok", provider_data: data)])
      body["messages"].as_a[1].as_h.has_key?("reasoning_details").should be_false
    end
  end

  it "keeps a tool call's extra_content (Gemini's thought signature) and sends it back" do
    gemini = "https://generativelanguage.googleapis.com/v1beta/openai"
    extra = {google: {thought_signature: "ts-1"}}
    msg, _ = parse_stream([
      {choices: [{index: 0, delta: {tool_calls: [{index: 0, id: "c1", type: "function", extra_content: extra, function: {name: "now", arguments: "{}"}}]}}]}.to_json,
    ], endpoint: gemini)
    call = msg.tool_calls.try(&.first) || fail("no tool call")

    sent = build_body([Agent::Message.user("hi"), msg], endpoint: gemini)["messages"][1]["tool_calls"][0]
    sent["extra_content"]["google"]["thought_signature"].should eq("ts-1")
    Agent::ToolCall.from_json(call.to_json).should eq(call)
  end

  it "takes a per-request effort over the configured one" do
    provider = Agent::Provider::OpenAI.new(Agent::Config.new(reasoning_effort: :low), "k")
    JSON.parse(provider.build_request([Agent::Message.user("hi")], nil)[:body])["reasoning_effort"].should eq("low")
    options = Agent::RequestOptions.new(reasoning_effort: :high)
    JSON.parse(provider.build_request([Agent::Message.user("hi")], nil, options)[:body])["reasoning_effort"].should eq("high")
  end

  it "detects OpenRouter from the endpoint" do
    provider = Agent::Provider::OpenAI.new(Agent::Config.new(api_endpoint: "https://openrouter.ai/api/v1", reasoning_effort: :medium), "k")
    JSON.parse(provider.build_request([Agent::Message.user("hi")], nil)[:body])["reasoning"]["effort"].should eq("medium")
  end

  describe "dump version" do
    it "refuses a dump newer than it supports" do
      agent = Agent.new(Agent::Config.new)
      dump = JSON.parse(agent.dump).as_h
      dump["version"].should eq(Agent::DUMP_VERSION)
      dump["version"] = JSON::Any.new(Agent::DUMP_VERSION.to_i64 + 1)
      expect_raises(Agent::SessionLoadError, "newer") { agent.load(dump.to_json) }
      agent.close
    end

    it "loads a dump from before versioning" do
      agent = Agent.new(Agent::Config.new)
      agent.load(%({"session_id":"s","cache_key":"k","history":[]}))
      agent.session_id.should eq("s")
      agent.close
    end
  end

  it "replays the model's reasoning blocks through a tool loop and a saved session" do
    requests = [] of JSON::Any
    server = HTTP::Server.new do |ctx|
      requests << JSON.parse(ctx.request.body.try(&.gets_to_end) || "{}")
      ctx.response.content_type = "text/event-stream"
      lines = if requests.size == 1
                [
                  details_delta({type: "reasoning.text", text: "Need the time.", signature: "sig-1", index: 0}),
                  {choices: [{index: 0, delta: {tool_calls: [{index: 0, id: "c1", type: "function", function: {name: "now", arguments: "{}"}}]}}]}.to_json,
                ]
              else
                [{choices: [{index: 0, delta: {content: "It is noon."}}]}.to_json]
              end
      lines.each { |l| ctx.response.puts "data: #{l}" }
      ctx.response.puts "data: [DONE]"
      ctx.response.close
    end
    port = server.bind_tcp(0).port
    spawn { server.listen }
    Fiber.yield

    config = Agent::Config.new(api_endpoint: "http://127.0.0.1:#{port}/v1", reasoning_effort: :medium)
    agent = Agent.new(config)
    agent.register_tool("now", "time") { |_| "12:00" }
    begin
      agent.ask("What time is it?").message.content.should eq("It is noon.")
      requests[0]["reasoning_effort"].should eq("medium")
      replayed = requests[1]["messages"].as_a.find! { |m| m["role"] == "assistant" }
      replayed["reasoning_details"].as_a.first["signature"].should eq("sig-1")

      restored = Agent.load(config, agent.dump)
      details(restored.history[1]).first["text"].should eq("Need the time.")
      restored.close
    ensure
      agent.close
      server.close
    end
  end

  it "drops replay data from a cancelled reply, keeping its text" do
    lines = [
      details_delta({type: "reasoning.text", text: "Half a thought", index: 0}),
      {choices: [{index: 0, delta: {content: "Partial"}}]}.to_json,
    ]
    with_sse(lines, trailer: 20) do |url, _|
      agent = Agent.new(Agent::Config.new(api_endpoint: url))
      resp = agent.ask("x")
      spawn { sleep 150.milliseconds; resp.cancel }
      resp.join
      resp.error.should be_a(Agent::CancelledError)
      kept = agent.history.last
      kept.content.should eq("Partial")
      kept.provider_data.should be_nil
      agent.close
    end
  end

  it "lets a handler change the effort for one provider call" do
    with_sse([{choices: [{index: 0, delta: {content: "ok"}}]}.to_json]) do |url, requests|
      agent = Agent.new(Agent::Config.new(api_endpoint: url, reasoning_effort: :low))
      agent.use(Agent::Spec::HighEffort.new)
      agent.ask("x").join
      requests.first["reasoning_effort"].should eq("high")
      agent.close
    end
  end

  describe "reasoning text sent back" do
    it "sends reasoning_content back in the field it came in (DeepSeek, Kimi)" do
      deepseek = "https://api.deepseek.com/v1"
      msg, _ = parse_stream([
        {choices: [{index: 0, delta: {reasoning_content: "Need the "}}]}.to_json,
        {choices: [{index: 0, delta: {reasoning_content: "time."}}]}.to_json,
        {choices: [{index: 0, delta: {tool_calls: [{index: 0, id: "c1", type: "function", function: {name: "now", arguments: "{}"}}]}, finish_reason: "tool_calls"}]}.to_json,
      ], endpoint: deepseek)
      msg.reasoning.should eq("Need the time.")

      sent = build_body([Agent::Message.user("hi"), msg], endpoint: deepseek)["messages"][1]
      sent["reasoning_content"].should eq("Need the time.")
      build_body([Agent::Message.user("hi"), msg], endpoint: deepseek, replay: false)["messages"][1].as_h.has_key?("reasoning_content").should be_false
    end

    it "does not send back reasoning that came in a display-only field" do
      msg, _ = parse_stream([
        {choices: [{index: 0, delta: {reasoning: "Thinking"}, finish_reason: "stop"}]}.to_json,
      ], endpoint: "https://api.groq.com/openai/v1")
      msg.reasoning.should eq("Thinking")
      msg.provider_data.should be_nil
    end

    it "reads Mistral's thinking chunks and sends them back ahead of the answer" do
      mistral = "https://api.mistral.ai/v1"
      thinking = ->(t : String) { {type: "thinking", thinking: [{type: "text", text: t}]} }
      msg, _ = parse_stream([
        {choices: [{index: 0, delta: {content: [thinking.call("Two plus ")]}}]}.to_json,
        {choices: [{index: 0, delta: {content: [thinking.call("two."), {type: "text", text: "It's "}]}}]}.to_json,
        {choices: [{index: 0, delta: {content: "4."}, finish_reason: "stop"}]}.to_json,
      ], endpoint: mistral)
      msg.reasoning.should eq("Two plus two.")
      msg.content.should eq("It's 4.")

      content = build_body([Agent::Message.user("2+2?"), msg], endpoint: mistral)["messages"][1]["content"].as_a
      content[0]["type"].should eq("thinking")
      content[0]["thinking"][0]["text"].should eq("Two plus two.")
      content[1].should eq(JSON.parse(%({"type":"text","text":"It's 4."})))
    end

    it "sends nothing back to a different model on the same endpoint" do
      blocks = JSON.parse(%([{"type":"reasoning.text","text":"x","signature":"s"}]))
      data = Agent::Provider::OpenAI.replay_data("https://openrouter.ai/api/v1", "anthropic/claude", {"reasoning_details" => blocks})
      history = [Agent::Message.user("hi"), Agent::Message.assistant("ok", provider_data: data)]
      build_body(history)["messages"][1].as_h.has_key?("reasoning_details").should be_false
    end
  end

  describe "at scale" do
    it "assembles a long reasoning stream in linear time" do
      fragment = "x" * 8
      lines = Array.new(40_000) { details_delta({type: "reasoning.text", text: fragment, index: 0}) }
      started = Time.instant
      msg, _ = parse_stream(lines)
      (Time.instant - started).should be < 3.seconds
      details(msg).first["text"].as_s.size.should eq(320_000)
    end

    it "joins chunks of a kind while they wait for #stream, keeping tool names apart" do
      response = Agent::Response.new
      %w[a b].each { |t| response.push_chunk(Agent::Response::Chunk.new(t, Agent::Response::ChunkKind::Reasoning)) }
      %w[c d].each { |t| response.push_chunk(Agent::Response::Chunk.new(t)) }
      %w[get_a get_b].each { |t| response.push_chunk(Agent::Response::Chunk.new(t, Agent::Response::ChunkKind::ToolCallName)) }
      response.finish(Agent::Message.assistant("cd"), Agent::Usage.new)

      seen = [] of {String, Agent::Response::ChunkKind}
      response.stream { |c| seen << {c.text, c.kind} }
      seen.should eq([
        {"ab", Agent::Response::ChunkKind::Reasoning},
        {"cd", Agent::Response::ChunkKind::Content},
        {"get_a", Agent::Response::ChunkKind::ToolCallName},
        {"get_b", Agent::Response::ChunkKind::ToolCallName},
      ])
    end
  end
end

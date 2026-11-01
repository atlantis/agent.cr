require "./spec_helper"

private ANTHROPIC  = "https://api.anthropic.com"
private OPENROUTER = "https://openrouter.ai/api/v1"

private def anthropic_body(messages, tools = nil, prompt_caching : Agent::PromptCaching = :explicit, cache_ttl : Agent::CacheTTL = :five_minutes)
  c = Agent::Config.new(api_key: "k", api_endpoint: ANTHROPIC, model: "claude-opus-5-5", prompt_caching: prompt_caching, cache_ttl: cache_ttl)
  JSON.parse(Agent::Provider::Anthropic.new(c).build_request(messages, tools)[:body])
end

private def openai_body(messages, endpoint = OPENROUTER, model = "anthropic/claude-opus-5.5", prompt_caching : Agent::PromptCaching = :explicit, cache_ttl : Agent::CacheTTL = :five_minutes)
  c = Agent::Config.new(api_key: "k", api_endpoint: endpoint, model: model, prompt_caching: prompt_caching, cache_ttl: cache_ttl)
  JSON.parse(Agent::Provider::OpenAI.new(c, "key").build_request(messages, nil)[:body])
end

private def tool(name)
  Agent::Tool.new(Agent::Tool::FunctionDef.new(name, "t", {"type" => JSON::Any.new("object")}))
end

# Every cache_control in a request body, as {json path, marker}.
private def markers(any : JSON::Any, path = "") : Array({String, JSON::Any})
  found = [] of {String, JSON::Any}
  if h = any.as_h?
    h.each do |k, v|
      if k == "cache_control"
        found << {path, v}
      else
        found.concat(markers(v, "#{path}/#{k}"))
      end
    end
  elsif a = any.as_a?
    a.each_with_index { |v, i| found.concat(markers(v, "#{path}/#{i}")) }
  end
  found
end

private def conversation
  call = Agent::ToolCall.new("t1", "probe", "{}")
  [
    Agent::Message.system("Base."),
    Agent::Message.user("one"),
    Agent::Message.assistant(nil, tool_calls: [call]),
    Agent::Message.tool_result(call, "ok"),
  ]
end

describe "prompt caching" do
  describe Agent::Provider::Anthropic do
    it "marks the system prompt and the latest turn, 5 minutes by default" do
      found = markers(anthropic_body(conversation))
      found.map(&.[0]).should eq ["/system/0", "/messages/2/content/0"]
      found.map(&.[1]).uniq!.should eq [JSON.parse(%({"type":"ephemeral"}))]
    end

    it "asks for an hour on every marker" do
      found = markers(anthropic_body(conversation, cache_ttl: :one_hour))
      found.size.should eq 2
      found.each(&.[1]["ttl"].should(eq("1h")))
    end

    it "leaves the conversation's marker to the API in automatic mode" do
      body = anthropic_body(conversation, prompt_caching: :automatic, cache_ttl: :one_hour)
      markers(body).map(&.[0]).sort!.should eq ["", "/system/0"]
      body["cache_control"].should eq JSON.parse(%({"type":"ephemeral","ttl":"1h"}))
    end

    it "sends no marker when caching is off" do
      markers(anthropic_body(conversation, prompt_caching: :off)).should be_empty
    end

    it "marks the last tool when there is no system prompt" do
      body = anthropic_body([Agent::Message.user("x")], [tool("a"), tool("b")])
      markers(body).map(&.[0]).should eq ["/tools/1", "/messages/0/content/0"]
    end

    it "adds markers inside a turn too long for the cache lookback, at most 4 in all" do
      parts = (1..60).map { |i| Agent::ContentPart.new(text: "part #{i}") }
      messages = [
        Agent::Message.system("Base."),
        Agent::Message.user("first"),
        Agent::Message.assistant("ok"),
        Agent::Message.new(role: Agent::Role::User, content_parts: parts),
      ]
      found = markers(anthropic_body(messages)).map(&.[0])
      found.should eq ["/system/0", "/messages/2/content/29", "/messages/2/content/44", "/messages/2/content/59"]
    end

    it "counts a run of parallel tool calls or results as one position" do
      calls = (1..30).map { |i| Agent::ToolCall.new("t#{i}", "probe", "{}") }
      messages = [Agent::Message.user("go"), Agent::Message.assistant(nil, tool_calls: calls)]
      messages.concat(calls.map { |c| Agent::Message.tool_result(c, "ok") })
      markers(anthropic_body(messages)).map(&.[0]).should eq ["/messages/2/content/29"]
    end

    it "marks a copy of a stored reply's block, never thinking, and leaves the reply as it was" do
      blocks = [JSON.parse(%({"type":"thinking","thinking":"t","signature":"s"}))]
      blocks.concat((1..30).map { |i| JSON.parse(%({"type":"text","text":"line #{i}"})) })
      reply = Agent::Message.assistant("x", provider_data: Agent::Provider::Anthropic.replay_data(ANTHROPIC, blocks))
      before = reply.provider_data.to_json
      messages = [Agent::Message.user("one"), reply, Agent::Message.user("two")]
      found = markers(anthropic_body(messages)).map(&.[0])
      found.should eq ["/messages/1/content/1", "/messages/1/content/16", "/messages/2/content/0"]
      reply.provider_data.to_json.should eq before
    end
  end

  describe Agent::Provider::OpenAI do
    it "marks a Claude model's system prompt and latest turn on OpenRouter" do
      body = openai_body(conversation)
      markers(body).map(&.[0]).should eq ["/messages/0/content/0", "/messages/3/content/0"]
      body["messages"][0]["content"][0]["text"].should eq "Base."
      body["messages"][3]["content"][0]["text"].should eq "ok"
    end

    it "uses OpenRouter's automatic marker and the hour-long lifetime" do
      body = openai_body(conversation, prompt_caching: :automatic, cache_ttl: :one_hour)
      markers(body).map(&.[0]).sort!.should eq ["", "/messages/0/content/0"]
      body["cache_control"]["ttl"].should eq "1h"
    end

    it "marks nothing where the endpoint caches on its own, or caching is off" do
      markers(openai_body(conversation, model: "openai/gpt-5.5")).should be_empty
      markers(openai_body(conversation, endpoint: "https://api.openai.com/v1", model: "gpt-5.5")).should be_empty
      markers(openai_body(conversation, prompt_caching: :off)).should be_empty
    end

    it "reads OpenRouter's count of tokens written to the cache" do
      usage = {usage: {prompt_tokens: 100, completion_tokens: 5, total_tokens: 105,
                       prompt_tokens_details: {cached_tokens: 80, cache_write_tokens: 20}}}
      body = "data: #{{choices: [{index: 0, delta: {content: "hi"}, finish_reason: "stop"}]}.to_json}\n\ndata: #{usage.to_json}\n\ndata: [DONE]\n\n"
      _, u, _ = Agent::Provider::OpenAI::StreamParser.parse(IO::Memory.new(body), Agent::Response.new, -> { false }, endpoint: OPENROUTER, model: "m")
      u.cached_tokens.should eq 80
      u.cache_creation_tokens.should eq 20
    end
  end
end

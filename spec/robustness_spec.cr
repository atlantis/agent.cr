require "./spec_helper"
require "http/server"

# Failures that used to leave an Agent wedged: a hung fiber, or a history the
# API rejects on every later request. Each test drives a scripted mock server
# whose POSTs are answered by the handlers in order.

private alias Script = Array(HTTP::Server::Context -> Nil)

private def with_script(handlers : Script, retries = 0, &)
  queue = handlers.dup
  requests = [] of JSON::Any
  server = HTTP::Server.new do |ctx|
    requests << JSON.parse(ctx.request.body.try(&.gets_to_end) || "{}")
    if handler = queue.shift?
      handler.call(ctx)
    else
      ctx.response.status_code = 500
    end
  end
  port = server.bind_tcp(0).port
  spawn { server.listen }
  Fiber.yield
  agent = Agent.new(Agent::Config.new(
    api_endpoint: "http://127.0.0.1:#{port}/v1",
    idle_byte_timeout: 5, first_byte_timeout_min: 5, first_byte_timeout_max: 5,
    max_retries: retries, retry_base_delay: 10.milliseconds,
  ))
  begin
    yield agent, requests
  ensure
    agent.close
    server.close
  end
end

private def sse(ctx, chunks : Array(String), usage = {prompt_tokens: 10, completion_tokens: 5, total_tokens: 15})
  ctx.response.content_type = "text/event-stream"
  chunks.each { |c| ctx.response.puts "data: #{c}"; ctx.response.flush }
  ctx.response.puts "data: #{{choices: [{delta: {} of String => String, index: 0, finish_reason: "stop"}], usage: usage}.to_json}"
  ctx.response.puts "data: [DONE]"
  ctx.response.close
end

private def text_reply(text : String) : HTTP::Server::Context -> Nil
  ->(ctx : HTTP::Server::Context) {
    sse(ctx, text.chars.map { |ch| {choices: [{delta: {content: ch.to_s}, index: 0}]}.to_json })
    nil
  }
end

private def tool_reply(calls : Array({String, String, String})) : HTTP::Server::Context -> Nil
  ->(ctx : HTTP::Server::Context) {
    chunks = calls.map_with_index do |(id, name, args), i|
      {choices: [{delta: {tool_calls: [{index: i, id: id, type: "function", function: {name: name, arguments: args}}]}, index: 0}]}.to_json
    end
    sse(ctx, chunks)
    nil
  }
end

private def status_reply(code : Int32, body : String = "") : HTTP::Server::Context -> Nil
  ->(ctx : HTTP::Server::Context) {
    ctx.response.status_code = code
    ctx.response.content_type = "application/json"
    ctx.response.print body
    nil
  }
end

# Answers each connection with the next raw HTTP bytes, then closes the
# socket: a way to cut a chunked stream off mid-way.
private def with_raw_replies(replies : Array(String), retries = 0, &)
  server = TCPServer.new("127.0.0.1", 0)
  connections = 0
  spawn do
    while client = server.accept?
      client.gets("\r\n\r\n") # request head
      reply = replies[connections]? || replies.last
      connections += 1
      client.print reply
      client.flush
      client.close
    end
  end
  agent = Agent.new(Agent::Config.new(
    api_endpoint: "http://127.0.0.1:#{server.local_address.port}/v1",
    idle_byte_timeout: 5, first_byte_timeout_min: 5, first_byte_timeout_max: 5,
    max_retries: retries, retry_base_delay: 10.milliseconds,
  ))
  begin
    yield agent, -> { connections }
  ensure
    agent.close
    server.close
  end
end

# A 200 event stream with these chunks of content, cut off without the
# terminating chunk.
private def truncated_stream(contents : Array(String)) : String
  head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
  head + contents.join do |c|
    event = "data: #{{choices: [{index: 0, delta: {content: c}}]}.to_json}\n\n"
    "#{event.bytesize.to_s(16)}\r\n#{event}\r\n"
  end
end

# Runs the block in a fiber and fails the spec instead of hanging the suite.
private def within(seconds, &block : -> T) : T forall T
  done = Channel(T).new(1)
  spawn { done.send(block.call) }
  select
  when value = done.receive then value
  when timeout(seconds.seconds) then fail "timed out after #{seconds}s"
  end
end

private def roles(request : JSON::Any) : Array(String)
  request["messages"].as_a.map(&.["role"].as_s)
end

class Agent::Spec::FailingTurn < Agent::Handler
  def handle(ctx : Agent::TurnContext, next_proc : Agent::TurnContext -> {Agent::Message, Agent::Usage, String?}) : {Agent::Message, Agent::Usage, String?}
    raise Agent::Error.new("budget exhausted")
  end
end

class Agent::Spec::QuotaError < Agent::ToolError
  getter tenant : String

  def initialize(message, @tenant)
    super(message)
  end
end

class Agent::Spec::ToolErrorRecorder < Agent::Handler
  getter errors = [] of Exception

  def handle(ctx : Agent::ToolCallContext, next_proc : Agent::ToolCallContext -> Agent::Message) : Agent::Message
    result = next_proc.call(ctx)
    return result unless err = ctx.error
    @errors << err
    return result unless err.is_a?(Agent::Spec::QuotaError)
    Agent::Message.tool_result(ctx.tool_call, "Quota reached for this account; tell the user.", is_error: true)
  end
end

describe "Agent robustness" do
  it "recovers the connection after an API error with a body" do
    with_script([status_reply(429, %({"error":{"message":"rate limited"}})), text_reply("ok")]) do |agent, _|
      first = agent.ask("one")
      first.join
      first.error.should be_a(Agent::ApiError)

      second = agent.ask("two")
      second.join
      second.error.should be_nil
      second.message.content.should eq("ok")
    end
  end

  it "keeps working when a stream consumer leaves early" do
    with_script([text_reply("x" * 600), text_reply("next")]) do |agent, _|
      resp = agent.ask("long")
      expect_raises(Exception, "client went away") do
        resp.stream { |_| raise "client went away" }
      end
      within(3) { resp.join }
      within(3) { agent.ask("again").message.content }.should eq("next")
    end
  end

  it "answers a call to an unknown tool instead of discarding the registered one's result" do
    calls = [{"c1", "charge", %({"amount":5})}, {"c2", "made_up", "{}"}]
    with_script([tool_reply(calls), text_reply("done")]) do |agent, requests|
      runs = 0
      agent.register_tool("charge", "charge") { |_| runs += 1; "charged" }

      agent.ask("pay").message.content.should eq("done")
      runs.should eq(1)
      tool_results = requests[1]["messages"].as_a.select { |m| m["role"] == "tool" }
      tool_results.map(&.["content"].as_s).should eq(["charged", "Error: tool 'made_up' is not available"])
    end
  end

  it "runs no registered tool when the message also calls a per-request tool" do
    calls = [{"c1", "charge", "{}"}, {"c2", "approve", "{}"}]
    with_script([tool_reply(calls), text_reply("after")]) do |agent, requests|
      runs = 0
      agent.register_tool("charge", "charge") { |_| runs += 1; "charged" }
      manual = Agent::Tool.new(Agent::Tool::FunctionDef.new(name: "approve"))

      agent.ask("pay", tools: [manual]).message.tool_calls.try(&.size).should eq(2)
      runs.should eq(0)

      # Skipping the tool-result follow-up no longer poisons the history.
      agent.ask("never mind").join
      roles(requests.last).should eq(["user", "assistant", "tool", "tool", "user"])
    end
  end

  it "keeps a valid history and reports usage when a later call of a tool loop fails" do
    with_script([tool_reply([{"c1", "t", "{}"}]), status_reply(500), text_reply("ok")]) do |agent, requests|
      agent.register_tool("t", "t") { |_| "ok" }
      resp = agent.ask("go")
      resp.join
      resp.error.should be_a(Agent::ApiError)
      resp.metadata.total_tokens.should eq(15)
      agent.history.map(&.role.to_s).should eq(["user", "assistant", "tool"])

      agent.ask("retry").join
      roles(requests.last).should eq(["user", "assistant", "tool", "user"])
    end
  end

  it "rolls back a turn whose first call fails" do
    with_script([status_reply(500)]) do |agent, _|
      agent.ask("go").join
      agent.history.should be_empty
    end
  end

  it "drops unexecuted tool calls when a turn is cancelled" do
    slow = ->(ctx : HTTP::Server::Context) {
      ctx.response.content_type = "text/event-stream"
      ctx.response.puts "data: #{{choices: [{delta: {tool_calls: [{index: 0, id: "c1", type: "function", function: {name: "t", arguments: "{\"a\":"}}]}, index: 0}]}.to_json}"
      ctx.response.flush
      40.times { sleep 50.milliseconds; ctx.response.puts ": keepalive"; ctx.response.flush }
      ctx.response.close
      nil
    }
    with_script([slow, text_reply("after")]) do |agent, requests|
      agent.register_tool("t", "t") { |_| "ok" }
      resp = agent.ask("go")
      spawn { sleep 100.milliseconds; resp.cancel }
      within(1) { resp.join }
      resp.error.should be_a(Agent::CancelledError)

      agent.ask("next").join
      roles(requests.last).should eq(["user", "user"])
    end
  end

  it "accepts usage fields that are null" do
    reply = ->(ctx : HTTP::Server::Context) {
      sse(ctx, [{choices: [{delta: {content: "hi"}, index: 0}]}.to_json], usage: {prompt_tokens: 10, completion_tokens: nil, total_tokens: 10})
      nil
    }
    with_script([reply]) do |agent, _|
      resp = agent.ask("x")
      resp.message.content.should eq("hi")
      resp.metadata.prompt_tokens.should eq(10)
      resp.metadata.completion_tokens.should be_nil
    end
  end

  it "finishes the response and stays usable when a handler raises" do
    with_script([text_reply("ok")]) do |agent, _|
      agent.use(Agent::Spec::FailingTurn.new)
      resp = within(2) { agent.ask("x").tap(&.join) }
      resp.error.try(&.message).should eq("budget exhausted")
      agent.history.should be_empty
      within(2) { agent.ask("y").tap(&.join) }.error.should_not be_nil
    end
  end

  it "keeps a tool's internal error out of the model's context" do
    with_script([tool_reply([{"c1", "db", "{}"}]), text_reply("sorry")]) do |agent, requests|
      agent.register_tool("db", "db") { |_| raise "PG::Error password=hunter2 host=10.0.0.5" }
      agent.ask("q").join
      result = requests[1]["messages"].as_a.find! { |m| m["role"] == "tool" }
      result["content"].as_s.should eq("Error executing tool 'db': the tool failed")
    end
  end

  it "shows handlers the exception a tool raised so they can choose what the model sees" do
    calls = [{"c1", "export", "{}"}, {"c2", "db", "{}"}]
    with_script([tool_reply(calls), text_reply("ok")]) do |agent, requests|
      recorder = Agent::Spec::ToolErrorRecorder.new
      agent.use(recorder)
      agent.register_tool("export", "export") { |_| raise Agent::Spec::QuotaError.new("tenant t1 over quota", "t1") }
      agent.register_tool("db", "db") { |_| raise "connection refused" }
      agent.ask("go").join

      recorder.errors.map(&.class).should eq([Agent::Spec::QuotaError, Exception])
      recorder.errors[0].as(Agent::Spec::QuotaError).tenant.should eq("t1")
      results = requests[1]["messages"].as_a.select { |m| m["role"] == "tool" }.map(&.["content"].as_s)
      results.should eq(["Quota reached for this account; tell the user.", "Error executing tool 'db': the tool failed"])
    end
  end

  it "lets the call in flight finish when the agent is closed" do
    slow = ->(ctx : HTTP::Server::Context) {
      chunks = "abc".chars.map { |ch| {choices: [{delta: {content: ch.to_s}, index: 0}]}.to_json }
      ctx.response.content_type = "text/event-stream"
      chunks.each { |c| sleep 100.milliseconds; ctx.response.puts "data: #{c}"; ctx.response.flush }
      ctx.response.puts "data: [DONE]"
      ctx.response.close
      nil
    }
    with_script([slow]) do |agent, _|
      resp = agent.ask("x")
      spawn { sleep 50.milliseconds; agent.close }
      within(2) { resp.message.content }.should eq("abc")
      expect_raises(Agent::ClosedError) { agent.ask("y") }
    end
  end

  describe "provider errors" do
    it "retries a 429 or 5xx before the stream starts, honouring Retry-After" do
      limited = ->(ctx : HTTP::Server::Context) {
        ctx.response.headers["Retry-After"] = "0.05"
        status_reply(429, %({"error":{"message":"slow down"}})).call(ctx)
      }
      with_script([limited, status_reply(503), text_reply("ok")], retries: 2) do |agent, requests|
        resp = agent.ask("x")
        resp.message.content.should eq("ok")
        requests.size.should eq(3)
      end
    end

    it "gives up after max_retries with the last error, its body and headers" do
      err = ->(ctx : HTTP::Server::Context) {
        ctx.response.headers["x-request-id"] = "req_123"
        status_reply(500, %({"error":{"message":"upstream exploded"}})).call(ctx)
      }
      with_script([err, err], retries: 1) do |agent, requests|
        resp = agent.ask("x")
        resp.join
        error = resp.error.as(Agent::ApiError)
        error.status_code.should eq(500)
        error.message.should eq("500 Internal Server Error: upstream exploded")
        error.request_id.should eq("req_123")
        error.body.try(&.includes?("upstream exploded")).should be_true
        requests.size.should eq(2)
      end
    end

    it "does not retry a client error" do
      with_script([status_reply(400, %({"error":{"message":"bad schema"}})), text_reply("ok")], retries: 2) do |agent, requests|
        resp = agent.ask("x")
        resp.join
        resp.error.as(Agent::ApiError).message.should eq("400 Bad Request: bad schema")
        requests.size.should eq(1)
      end
    end

    it "does not wait out a Retry-After longer than retry_max_delay" do
      long = ->(ctx : HTTP::Server::Context) {
        ctx.response.headers["Retry-After"] = "3600"
        status_reply(429).call(ctx)
      }
      with_script([long, text_reply("ok")], retries: 2) do |agent, requests|
        resp = within(2) { agent.ask("x").tap(&.join) }
        resp.error.as(Agent::ApiError).retry_after.should eq(3600.seconds)
        requests.size.should eq(1)
      end
    end

    it "fails on an error event inside the stream" do
      broken = ->(ctx : HTTP::Server::Context) {
        ctx.response.content_type = "text/event-stream"
        ctx.response.puts "data: #{{choices: [{index: 0, delta: {content: "Hal"}}]}.to_json}"
        ctx.response.puts "data: #{{error: {code: 502, message: "Upstream provider error"}, choices: [{index: 0, delta: {content: ""}, finish_reason: "error"}]}.to_json}"
        ctx.response.puts "data: [DONE]"
        ctx.response.close
        nil
      }
      with_script([broken, text_reply("never")], retries: 2) do |agent, requests|
        resp = agent.ask("x")
        resp.join
        error = resp.error.as(Agent::StreamError)
        error.code.should eq("502")
        resp.message.content.should eq("Agent error: provider error in stream: Upstream provider error")
        requests.size.should eq(1) # tokens had streamed: not retried
        agent.history.should be_empty
      end
    end

    it "fails on a stream that stops before the model finished" do
      cut = ->(ctx : HTTP::Server::Context) {
        ctx.response.content_type = "text/event-stream"
        ctx.response.puts "data: #{{choices: [{index: 0, delta: {content: "Half an ans"}}]}.to_json}"
        ctx.response.close
        nil
      }
      with_script([cut]) do |agent, _|
        resp = agent.ask("x")
        resp.join
        resp.error.should be_a(Agent::StreamError)
        resp.error.try(&.message).should eq("stream ended before the model finished")
      end
    end
  end

  it "does not run the tool calls of a reply cut off by its token limit" do
    cut = ->(ctx : HTTP::Server::Context) {
      ctx.response.content_type = "text/event-stream"
      call = {index: 0, id: "c1", type: "function", function: {name: "probe", arguments: %({"path":"/et)}}
      ctx.response.puts "data: #{{choices: [{index: 0, delta: {tool_calls: [call]}}]}.to_json}"
      ctx.response.puts "data: #{{choices: [{index: 0, delta: {} of String => String, finish_reason: "length"}]}.to_json}"
      ctx.response.puts "data: [DONE]"
      ctx.response.close
      nil
    }
    with_script([cut]) do |agent, _|
      ran = [] of String
      agent.register_tool("probe", "probe") { |_| ran << "probe"; "ok" }
      resp = agent.ask("go")
      resp.join
      resp.error.as(Agent::StreamError).code.should eq("length")
      ran.should be_empty
      agent.history.should be_empty
    end
  end

  it "skips the remaining tool calls once the response is cancelled" do
    calls = [{"c1", "first", "{}"}, {"c2", "second", "{}"}]
    with_script([tool_reply(calls)]) do |agent, _|
      ran = [] of String
      current = [] of Agent::Response
      agent.register_tool("first", "first") { |_| ran << "first"; current.each(&.cancel); "done" }
      agent.register_tool("second", "second") { |_| ran << "second"; "done" }
      resp = agent.ask("go")
      current << resp
      resp.join
      ran.should eq(["first"])
      resp.error.should be_a(Agent::CancelledError)
      agent.history.last.content.should eq("Error: cancelled before this tool ran")
    end
  end

  it "delivers chunks that arrived before #stream was called" do
    with_script([text_reply("hello")]) do |agent, _|
      resp = agent.ask("x")
      resp.join
      text = String.build { |io| resp.stream { |c| io << c.text } }
      text.should eq("hello")
    end
  end

  describe "where a failure lands" do
    it "retries an error event that arrives before any output" do
      first = ->(ctx : HTTP::Server::Context) {
        ctx.response.content_type = "text/event-stream"
        ctx.response.puts "data: #{{error: {code: 503, message: "No provider available"}}.to_json}"
        ctx.response.close
        nil
      }
      with_script([first, text_reply("ok")], retries: 1) do |agent, requests|
        resp = agent.ask("x")
        resp.message.content.should eq("ok")
        requests.size.should eq(2)
      end
    end

    it "retries a later call of a tool loop although an earlier one streamed" do
      calls = [{"c1", "probe", "{}"}]
      with_script([tool_reply(calls), status_reply(503), text_reply("done")], retries: 1) do |agent, requests|
        agent.register_tool("probe", "probe") { |_| "ok" }
        resp = agent.ask("x")
        resp.message.content.should eq("done")
        requests.size.should eq(3)
      end
    end

    it "does not retry an error event with a client error code" do
      first = ->(ctx : HTTP::Server::Context) {
        ctx.response.content_type = "text/event-stream"
        ctx.response.puts "data: #{{error: {code: 402, message: "Insufficient credits"}}.to_json}"
        ctx.response.close
        nil
      }
      with_script([first, text_reply("ok")], retries: 2) do |agent, requests|
        resp = agent.ask("x")
        resp.join
        resp.error.as(Agent::StreamError).retryable?.should be_false
        resp.partial?.should be_false
        requests.size.should eq(1)
      end
    end

    it "reports a connection lost mid-stream as a partial StreamError" do
      with_raw_replies([truncated_stream(["Hel", "lo"])], retries: 2) do |agent, connections|
        resp = agent.ask("x")
        text = String.build { |io| resp.stream { |c| io << c.text } }
        text.should eq("Hello")
        error = resp.error.as(Agent::StreamError)
        error.message.try(&.starts_with?("connection lost during the stream")).should be_true
        error.cause.should be_a(IO::Error)
        resp.partial?.should be_true
        connections.call.should eq(1) # output had streamed: not retried
        agent.history.should be_empty
      end
    end

    it "retries a stream cut off before any output" do
      with_raw_replies([truncated_stream([] of String), truncated_stream(["ok"])], retries: 1) do |agent, connections|
        resp = agent.ask("x")
        resp.join
        connections.call.should eq(2)
        resp.partial?.should be_true # the second one was cut off too, after output
      end
    end

    it "fails a 200 with a JSON body instead of a stream" do
      json = ->(ctx : HTTP::Server::Context) {
        ctx.response.content_type = "application/json"
        ctx.response.print %({"error":{"message":"model not found"}})
        nil
      }
      with_script([json, text_reply("ok")], retries: 2) do |agent, requests|
        resp = agent.ask("x")
        resp.join
        resp.error.as(Agent::StreamError).message.should eq("provider replied with JSON instead of a stream: model not found")
        requests.size.should eq(1)
      end
    end

    it "passes an Agent::Error raised by a chunk handler through unchanged" do
      with_script([text_reply("hi")]) do |agent, _|
        agent.use(Agent::Spec::ChunkRefuser.new)
        resp = agent.ask("x")
        resp.join
        resp.error.should be_a(Agent::Spec::QuotaError)
      end
    end
  end
end

class Agent::Spec::ChunkRefuser < Agent::Handler
  def handle(ctx : Agent::ChunkContext, next_proc : Agent::ChunkContext -> Agent::Response::Chunk) : Agent::Response::Chunk
    raise Agent::Spec::QuotaError.new("out of tokens", "t1")
  end
end

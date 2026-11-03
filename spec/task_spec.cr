require "./spec_helper"
require "http/server"

# Background tasks: a tool starts work and answers at once, the turn ends,
# and the outcome reaches the model as a `<task_result>` message with the
# next model call. Each test drives a scripted mock server whose POSTs are
# answered by the handlers in order.

private alias Script = Array(HTTP::Server::Context -> Nil)

private def with_script(handlers : Script, max_background_tasks = 4, max_history = nil, task_id_generator : (-> String)? = nil, max_task_result_chars : Int32? = 30_000, mid_turn = false, &)
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
    max_retries: 0, max_background_tasks: max_background_tasks, max_history: max_history, task_id_generator: task_id_generator, max_task_result_chars: max_task_result_chars, deliver_tasks_mid_turn: mid_turn,
  ))
  begin
    yield agent, requests
  ensure
    agent.close
    server.close
  end
end

private def sse(ctx, chunks : Array(String))
  ctx.response.content_type = "text/event-stream"
  chunks.each { |c| ctx.response.puts "data: #{c}"; ctx.response.flush }
  ctx.response.puts "data: #{{choices: [{delta: {} of String => String, index: 0, finish_reason: "stop"}], usage: {prompt_tokens: 10, completion_tokens: 5, total_tokens: 15}}.to_json}"
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

private def status_reply(code : Int32) : HTTP::Server::Context -> Nil
  ->(ctx : HTTP::Server::Context) {
    ctx.response.status_code = code
    ctx.response.content_type = "application/json"
    ctx.response.print %({"error":{"message":"boom"}})
    nil
  }
end

private def empty_schema
  Agent::JSONConverter.from({type: "object", properties: {} of String => String, required: [] of String})
end

# Wait for the task to end and leave the running list, or fail the spec
# instead of hanging the suite.
private def await(agent : Agent, task : Agent::Task, seconds = 2) : Agent::Task
  deadline = Time.instant + seconds.seconds
  while task.running? || agent.running_tasks.includes?(task)
    fail "task #{task.id} still running after #{seconds}s" if Time.instant > deadline
    sleep 5.milliseconds
  end
  task
end

# Tells the model why a task failed, from the exception's class, and
# shortens a long output.
class Agent::Spec::TaskExplainer < Agent::Handler
  def handle(ctx : Agent::TaskContext, next_proc : Agent::TaskContext -> Agent::Message) : Agent::Message
    if err = ctx.error
      ctx.result = "#{err.class.name.split("::").last}: #{ctx.result}" unless err.is_a?(Agent::ToolError)
    elsif (text = ctx.result) && text.size > 20
      ctx.result = "#{text[0, 20]}... (#{text.size} chars; call read_task_output)"
    end
    next_proc.call(ctx)
  end
end

class Agent::Spec::BrokenTaskHandler < Agent::Handler
  def handle(ctx : Agent::TaskContext, next_proc : Agent::TaskContext -> Agent::Message) : Agent::Message
    raise "handler bug"
  end
end

private def roles(request : JSON::Any) : Array(String)
  request["messages"].as_a.map(&.["role"].as_s)
end

private def text_of(message : JSON::Any) : String
  content = message["content"]
  content.as_s? || content.as_a.join { |p| p["text"]?.try(&.as_s) || "" }
end

describe Agent::Task do
  it "a tool starts a task, the turn ends, and resume delivers the outcome" do
    with_script([tool_reply([{"c1", "wait", %({"for":"x"})}]), text_reply("Waiting."), text_reply("Done!")]) do |agent, requests|
      started : Agent::Task? = nil
      ended = [] of Agent::Task
      agent.on_task_finished { |t| ended << t }
      agent.register_tool("wait", "wait for something", parameters: empty_schema) do |args|
        task = agent.background("#{args["for"]} to happen", timeout: 1.second) { "x happened" }
        started = task
        task.accepted
      end

      resp = agent.ask("wait for x")
      resp.message.content.should eq "Waiting."
      tool_result = requests[1]["messages"].as_a.find! { |m| m["role"] == "tool" }
      text_of(tool_result).should start_with "Started background task t-"
      text_of(tool_result).should contain "(x to happen). It times out after 1s. Its result will arrive in a later message as <task_result id=\"t-"

      task = await(agent, started.as(Agent::Task))
      ended.should eq [task]
      task.status.done?.should be_true
      task.tool.should eq "wait"
      task.result.should eq "x happened"
      agent.finished_tasks.should eq [task]
      agent.awaiting_tasks?.should be_true

      agent.resume.message.content.should eq "Done!"
      roles(requests[2]).should eq %w[user assistant tool assistant user]
      delivered = text_of(requests[2]["messages"][-1])
      delivered.should start_with %(<task_result id="#{task.id}" tool="wait" description="x to happen" status="done" elapsed=")
      delivered.should end_with ">\nx happened\n</task_result>"
      requests[2]["messages"][-1].as_h.has_key?("source").should be_false

      agent.history[-2].task_result?.should be_true
      agent.history[-2].source.task?.should be_true
      agent.history[-1].content.should eq "Done!"
      agent.awaiting_tasks?.should be_false
    end
  end

  it "delivers outcomes ahead of the next user message, as one message on the wire" do
    with_script([text_reply("hi"), text_reply("yes")]) do |agent, requests|
      agent.ask("hello").message
      task = agent.background { "the deploy finished" }
      task.tool.should be_nil
      await(agent, task)

      agent.ask("and now?").message.content.should eq "yes"
      roles(requests[1]).should eq %w[user assistant user]
      text = requests[1]["messages"][-1]["content"].as_s
      text.should contain %(status="done")
      text.should contain "the deploy finished"
      text.should end_with "</task_result>\n\nand now?"

      history = agent.history
      history.map(&.role.to_s).should eq %w[user assistant user user assistant]
      history[2].task_result?.should be_true
      history[3].source.human?.should be_true
    end
  end

  it "holds outcomes that arrive mid-turn until the next user message, by default" do
    with_script([tool_reply([{"c1", "start", "{}"}]), tool_reply([{"c2", "noop", "{}"}]), text_reply("ok"), text_reply("seen")]) do |agent, requests|
      agent.register_tool("start", parameters: empty_schema) { agent.background { "finished" }.accepted }
      agent.register_tool("noop", parameters: empty_schema) { "nothing" }

      agent.ask("go").message.content.should eq "ok"
      requests.size.should eq 3
      requests.each { |r| r["messages"].as_a.none? { |m| m["role"] == "user" && text_of(m).includes?("<task_result") }.should be_true }
      agent.finished_tasks.size.should eq 1

      agent.ask("next").message.content.should eq "seen"
      roles(requests[3]).should eq %w[user assistant tool assistant tool assistant user]
      text_of(requests[3]["messages"][-1]).should start_with "<task_result"
    end
  end

  it "reports outcomes that arrive mid-turn after the tool results when asked to" do
    with_script([tool_reply([{"c1", "start", "{}"}]), tool_reply([{"c2", "noop", "{}"}]), text_reply("ok")], mid_turn: true) do |agent, requests|
      agent.register_tool("start", parameters: empty_schema) { agent.background { "finished" }.accepted }
      agent.register_tool("noop", parameters: empty_schema) { "nothing" }

      agent.ask("go").message.content.should eq "ok"
      messages = requests[2]["messages"].as_a
      idx = messages.index! { |m| m["role"] == "user" && text_of(m).includes?("<task_result") }
      messages[idx - 1]["role"].should eq "tool"
      messages[idx - 1]["tool_call_id"].should eq "c2"
      messages[idx + 1]?.should be_nil
      agent.awaiting_tasks?.should be_false
    end
  end

  it "reports a task that overruns its timeout as timed out, and tells it to stop" do
    with_script([text_reply("hi")]) do |agent, _requests|
      stopped = Channel(Bool).new(1)
      task = agent.background("a slow thing", timeout: 50.milliseconds) do |t|
        slept = t.sleep(5.seconds)
        stopped.send(!slept)
        slept ? "slept" : "stopped"
      end

      await(agent, task)
      task.status.timed_out?.should be_true
      task.result.should eq "timed out after 50ms"
      task.cancelled?.should be_true
      stopped.receive.should be_true
      # the block's late outcome is dropped
      Fiber.yield
      task.status.timed_out?.should be_true
      task.result.should eq "timed out after 50ms"
      task.to_message.content.to_s.should contain %(status="timed_out")
      task.to_message.content.to_s.should contain "timed out after 50ms"
    end
  end

  it "reports a ToolError's message and hides any other exception, keeping it on the task" do
    with_script([text_reply("hi")]) do |agent, _requests|
      first = agent.background { raise Agent::ToolError.new("no such alarm") }
      second = agent.background { raise "secret host 10.0.0.1 unreachable" }
      results = [await(agent, first), await(agent, second)]
      results.map(&.status.failed?).should eq [true, true]
      results.map(&.result).should eq ["no such alarm", "the task failed"]
      results[0].error.should be_a Agent::ToolError
      results[1].error.to_s.should contain "10.0.0.1"
      results[0].to_message.content.to_s.should contain %(status="failed")
      results[0].to_message.content.to_s.should contain ">\nno such alarm\n<"
      agent.background { "fine" }.tap { |t| await(agent, t) }.error.should be_nil
    end
  end

  it "lets a handler rewrite what the model reads, and loses nothing when one raises" do
    with_script([text_reply("ok"), text_reply("ok")]) do |agent, requests|
      agent.use(Agent::Spec::TaskExplainer.new)
      failed = agent.background { raise IO::Error.new("connection refused") }
      long = agent.background { "x" * 50 }
      await(agent, failed)
      await(agent, long)

      agent.ask("so?").message
      text = text_of(requests[0]["messages"][0])
      text.should contain %(status="failed")
      text.should contain ">\nError: the task failed\n<"
      text.should contain "xxxxxxxxxxxxxxxxxxxx... (50 chars; call read_task_output)"
      failed.result.should eq "the task failed"
      long.result.should eq "x" * 50

      # a raising handler is logged and the outcome delivered as it is
      agent.use(Agent::Spec::BrokenTaskHandler.new)
      task = agent.background { "y" }
      await(agent, task)
      agent.ask("and?").message.content.should eq "ok"
      agent.finished_tasks.should be_empty
      text_of(requests[1]["messages"][-1]).should contain ">\ny\n</task_result>"
      agent.history.map(&.role.to_s).should eq %w[user user user assistant user user assistant]
    end
  end

  it "turns the task limit into an error result the model can act on" do
    script = [tool_reply([{"c1", "wait", "{}"}, {"c2", "wait", "{}"}]), text_reply("ok")]
    with_script(script, max_background_tasks: 1) do |agent, requests|
      agent.register_tool("wait", parameters: empty_schema) do
        agent.background { |t| t.sleep(5.seconds); "late" }.accepted
      end

      agent.ask("go").message.content.should eq "ok"
      results = requests[1]["messages"].as_a.select { |m| m["role"] == "tool" }
      text_of(results[0]).should start_with "Started background task"
      text_of(results[1]).should eq "Error executing tool 'wait': background task limit (1) reached; wait for a task to finish"
      agent.running_tasks.size.should eq 1

      expect_raises(Agent::TaskLimitError, "limit (1)") { agent.background { "x" } }
    end
  end

  it "refuses a bad timeout and a user message marked as a task result" do
    with_script([] of HTTP::Server::Context -> Nil) do |agent, _requests|
      expect_raises(ArgumentError, "timeout must be positive") { agent.background(timeout: 0.seconds) { "x" } }
      expect_raises(ArgumentError, "reserved") { agent.ask("x", source: :task) }
    end
  end

  it "answers a resume with nothing to deliver from history, never with a request" do
    with_script([text_reply("hi"), tool_reply([{"c1", "x", "{}"}])]) do |agent, requests|
      empty = agent.resume
      empty.join
      empty.error.to_s.should contain "nothing to deliver"

      agent.ask("hello").message
      resp = agent.resume
      resp.message.content.should eq "hi"
      resp.from_history?.should be_true
      resp.metadata.prompt_tokens.should be_nil
      requests.size.should eq 1

      # a manual dispatch still pending is left alone
      pending = agent.ask("do x").message
      pending.has_tool_calls?.should be_true
      again = agent.resume
      again.message.should eq pending
      again.from_history?.should be_true
      requests.size.should eq 2
      agent.history.last.should eq pending
    end
  end

  it "refuses ask and resume from its own fiber instead of hanging" do
    with_script([tool_reply([{"c1", "peek", "{}"}]), text_reply("ok")]) do |agent, requests|
      agent.register_tool("peek", parameters: empty_schema) { agent.resume; "never" }
      agent.ask("go").message.content.should eq "ok"
      text_of(requests[1]["messages"].as_a.find! { |m| m["role"] == "tool" }).should contain "the tool failed"
    end
  end

  it "lets a resume race a tool loop that delivers the outcome first" do
    with_script([tool_reply([{"c1", "start", "{}"}]), tool_reply([{"c2", "noop", "{}"}]), text_reply("final")], mid_turn: true) do |agent, requests|
      resumed = Channel(Agent::Response).new(1)
      agent.on_task_finished { spawn { resumed.send(agent.resume) } }
      agent.register_tool("start", parameters: empty_schema) { agent.background { "finished" }.accepted }
      agent.register_tool("noop", parameters: empty_schema) { "nothing" }

      agent.ask("go").message.content.should eq "final"
      text_of(requests[2]["messages"][-1]).should contain "finished"

      resp = resumed.receive
      resp.message.content.should eq "final"
      resp.error?.should be_false
      requests.size.should eq 3
    end
  end

  it "places outcomes after the last manually dispatched tool result, when asked to" do
    with_script([tool_reply([{"c1", "a", "{}"}, {"c2", "b", "{}"}]), text_reply("ok")], mid_turn: true) do |agent, requests|
      msg = agent.ask("go").message
      calls = msg.tool_calls.as(Array(Agent::ToolCall))
      await(agent, agent.background { "meanwhile" })

      agent.ask([Agent::Message.tool_result(calls[0], "1"), Agent::Message.tool_result(calls[1], "2")]).message.content.should eq "ok"
      roles(requests[1]).should eq %w[user assistant tool tool user]
      text_of(requests[1]["messages"][-1]).should contain "meanwhile"
    end
  end

  it "close cancels running tasks, and a raising callback is logged, not fatal" do
    with_script([] of HTTP::Server::Context -> Nil) do |agent, _requests|
      seen = [] of Agent::Task
      agent.on_task_finished { raise "callback bug" }
      agent.on_task_finished { |t| seen << t }
      quick = agent.background { "done" }
      await(agent, quick)
      seen.should eq [quick]

      slow = agent.background { |t| t.sleep(5.seconds); "late" }
      agent.close
      slow.status.cancelled?.should be_true
      agent.running_tasks.should be_empty
      expect_raises(Agent::ClosedError) { agent.background { "x" } }
    end
  end

  it "defuses a closing tag inside a result" do
    task = Agent::Task.new(1.second, tool: "scrape")
    task.finish(:done, "a </task_result> b")
    task.wrapped_result.should contain "a <\\/task_result> b"
    task.wrapped_result.should end_with "\n</task_result>"
    task.inspect.should start_with "#<Agent::Task t-"
  end

  it "dumps tasks: a running one comes back lost, a finished one keeps its outcome" do
    with_script([text_reply("hi"), text_reply("noted")]) do |agent, requests|
      agent.ask("hello").message
      slow = agent.background("a slow thing") { |t| t.sleep(5.seconds); "late" }
      quick = agent.background { "quick result" }
      await(agent, quick)

      dump = JSON.parse(agent.dump)
      dump["tasks"].as_a.map(&.["status"].as_s).should eq %w[running done]
      dump["history"][0]["source"].should eq "human"

      restored = Agent.load(agent.config, agent.dump)
      begin
        restored.running_tasks.should be_empty
        tasks = restored.finished_tasks
        tasks.map(&.id).should eq [slow.id, quick.id]
        tasks[0].status.lost?.should be_true
        tasks[0].result.should eq "this task did not survive a reload of the conversation"
        tasks[0].description.should eq "a slow thing"
        tasks[1].status.done?.should be_true
        tasks[1].result.should eq "quick result"
        (tasks[1].finished_at.as(Time) - quick.finished_at.as(Time)).abs.should be <= 1.millisecond

        restored.resume.message.content.should eq "noted"
        delivered = text_of(requests[1]["messages"][-1])
        delivered.index!(%(status="lost")).should be < delivered.index!(%(status="done"))
      ensure
        restored.close
      end

      # its own snapshot loaded back keeps the task it is still running
      agent.load(agent.dump)
      slow.running?.should be_true
      agent.running_tasks.should eq [slow]
      agent.finished_tasks.map(&.id).should eq [quick.id]

      # another conversation's dump cancels it
      other = Agent.new(agent.config)
      begin
        agent.load(other.dump)
      ensure
        other.close
      end
      slow.status.cancelled?.should be_true
      agent.awaiting_tasks?.should be_false
    end
  end

  it "relaunches a task of a registered kind after a restart, from its data" do
    with_script([text_reply("ok")]) do |agent, _requests|
      agent.register_task_kind("count") { |t| t.sleep(20.milliseconds); "counted #{t.data.try(&.["n"])}" }
      expect_raises(ArgumentError, "no task kind 'nope'") { agent.background(kind: "nope") }
      expect_raises(ArgumentError, "must not be empty") { agent.register_task_kind("") { "x" } }

      task = agent.background("counting", kind: "count", data: JSON::Any.new({"n" => JSON::Any.new(3_i64)}))
      task.kind.should eq "count"
      dump = agent.dump
      JSON.parse(dump)["tasks"][0]["data"]["n"].should eq 3
      JSON.parse(dump)["tasks"][0]["status"].should eq "running"

      restarted = Agent.new(agent.config)
      begin
        restarted.register_task_kind("count") { |t| "counted again #{t.data.try(&.["n"])}" }
        restarted.load(dump)
        relaunched = restarted.running_tasks.first
        relaunched.id.should eq task.id
        relaunched.description.should eq "counting"
        await(restarted, relaunched)
        relaunched.result.should eq "counted again 3"
        restarted.finished_tasks.should eq [relaunched]
        relaunched.to_message.content.to_s.should_not contain "\"n\""
      ensure
        restarted.close
      end

      # without the kind registered it is lost, as before
      unaware = Agent.load(agent.config, dump)
      begin
        unaware.running_tasks.should be_empty
        unaware.finished_tasks.first.status.lost?.should be_true
      ensure
        unaware.close
      end
    end
  end

  it "lets the application finish a task it started without a block" do
    with_script([tool_reply([{"c1", "settle", "{}"}]), text_reply("ok"), text_reply("noted")]) do |agent, requests|
      agent.register_tool("settle", parameters: empty_schema) { agent.background("deploy 42 to settle", timeout: 1.minute).accepted }
      agent.ask("go").message.content.should eq "ok"
      task = agent.running_tasks.first
      task.external?.should be_true
      task.tool.should eq "settle"
      agent.awaiting_tasks?.should be_true

      agent.complete_task("nope", "x").should be_false
      agent.complete_task(task.id, "deploy 42 is serving on 3 nodes").should be_true
      # queued before complete returns, not when the watcher fiber gets to run
      agent.finished_tasks.should eq [task]
      agent.running_tasks.should be_empty
      agent.complete_task(task.id, "again").should be_false
      await(agent, task)
      task.status.done?.should be_true
      agent.finished_tasks.should eq [task]

      agent.ask("and?").message.content.should eq "noted"
      text = text_of(requests[2]["messages"][-1])
      text.should contain %(<task_result id="#{task.id}" tool="settle" description="deploy 42 to settle" status="done")
      text.should contain "deploy 42 is serving on 3 nodes"

      failed = agent.background("a check")
      agent.fail_task(failed.id, "the rollback did not take: 2 containers still Down").should be_true
      agent.finished_tasks.should eq [failed]
      await(agent, failed)
      failed.status.failed?.should be_true
      failed.result.should eq "the rollback did not take: 2 containers still Down"
      failed.fail("twice").should be_false

      slow = agent.background("never reported", timeout: 20.milliseconds)
      await(agent, slow)
      slow.status.timed_out?.should be_true
    end
  end

  it "queues an outcome the application produced on its own" do
    with_script([text_reply("ok")]) do |agent, requests|
      started = Time.utc - 3.minutes
      task = Agent::Task.new(45.minutes, tool: "rollback", description: "rollback of 42 to settle", id: "fu-9", started_at: started)
      expect_raises(ArgumentError, "still running") { agent.queue_task(task) }
      task.fail("the rollback did not take: 2 containers still Down")
      agent.queue_task(task)
      expect_raises(Agent::TaskIdError, "fu-9 is already in use") { agent.queue_task(task) }
      agent.awaiting_tasks?.should be_true

      cancelled = Agent::Task.new(1.minute, id: "fu-10")
      cancelled.cancel
      expect_raises(ArgumentError, "cancelled") { agent.queue_task(cancelled) }

      JSON.parse(agent.dump)["tasks"].as_a.map(&.["id"].as_s).should eq ["fu-9"]

      agent.resume.message.content.should eq "ok"
      text = text_of(requests[0]["messages"][0])
      text.should contain %(<task_result id="fu-9" tool="rollback" description="rollback of 42 to settle" status="failed" elapsed="3m">)
      text.should contain "the rollback did not take"
      agent.history.first.source.task?.should be_true
      agent.awaiting_tasks?.should be_false
    end
  end

  it "completing a block's task from outside ends its sleep" do
    with_script([] of HTTP::Server::Context -> Nil) do |agent, _requests|
      woke = Channel(Bool).new(1)
      task = agent.background { |t| woke.send(t.sleep(5.seconds)); "late" }
      Fiber.yield
      task.complete("early").should be_true
      woke.receive.should be_false
      await(agent, task)
      task.result.should eq "early"
    end
  end

  it "keeps an externally finished task waiting across a reload, with the time it had left" do
    with_script([] of HTTP::Server::Context -> Nil) do |agent, _requests|
      task = agent.background("a long wait", timeout: 1.minute)
      restored = Agent.load(agent.config, agent.dump)
      begin
        again = restored.running_tasks.first
        again.id.should eq task.id
        again.external?.should be_true
        restored.complete_task(task.id, "done elsewhere").should be_true
        await(restored, again)
        restored.finished_tasks.first.result.should eq "done elsewhere"
      ensure
        restored.close
      end

      # one whose time ran out while the process was down times out at once
      stale = JSON.parse(agent.dump)
      stale["tasks"][0].as_h["started_at"] = JSON::Any.new((Time.utc - 1.hour).to_rfc3339)
      late = Agent.load(agent.config, stale.to_json)
      begin
        timed = late.running_tasks.first? || late.finished_tasks.first
        await(late, timed)
        timed.status.timed_out?.should be_true
      ensure
        late.close
      end
    end
  end

  it "drops cancelled and duplicate tasks from a dump, and refuses broken ones" do
    with_script([] of HTTP::Server::Context -> Nil) do |agent, _requests|
      task = {id: "t-1", timeout_ms: 1000, started_at: Time.utc.to_rfc3339, status: "done", result: "r"}
      dump = {version: 2, session_id: "s", cache_key: "c", history: [] of String,
              tasks: [task, task.merge({id: "t-2", status: "cancelled"}), task.merge({result: "again"})]}
      agent.load(dump.to_json)
      agent.finished_tasks.map { |t| {t.id, t.result} }.should eq [{"t-1", "r"}]

      expect_raises(Agent::SessionLoadError, "'tasks' field is malformed") do
        agent.load(dump.merge({tasks: [task.merge({timeout_ms: 0})]}).to_json)
      end
      expect_raises(Agent::SessionLoadError, "'tasks' field is malformed") do
        agent.load(dump.merge({tasks: [task.merge({id: ""})]}).to_json)
      end
    end
  end

  it "discards an undelivered outcome, and unregisters a callback" do
    with_script([text_reply("ok")]) do |agent, requests|
      seen = [] of String
      callback = agent.on_task_finished { |t| seen << t.id }
      task = agent.background { "x" }
      await(agent, task)
      seen.should eq [task.id]
      agent.discard_task("nope").should be_false
      agent.discard_task(task.id).should be_true
      agent.awaiting_tasks?.should be_false

      agent.off_task_finished(callback).should be_true
      agent.off_task_finished(callback).should be_false
      await(agent, agent.background { "y" })
      seen.should eq [task.id]

      agent.ask("hi").message
      text_of(requests[0]["messages"][0]).should_not contain "\nx\n"
    end
  end

  it "reset cancels running tasks and drops undelivered outcomes" do
    with_script([] of HTTP::Server::Context -> Nil) do |agent, _requests|
      ended = [] of Agent::Task
      agent.on_task_finished { |t| ended << t }
      slow = agent.background { |t| t.sleep(5.seconds); "late" }
      quick = agent.background { "done" }
      await(agent, quick)

      agent.reset
      agent.awaiting_tasks?.should be_false
      slow.status.cancelled?.should be_true
      slow.cancelled?.should be_true
      Fiber.yield
      ended.should eq [quick]
    end
  end

  it "queues the outcome again when the first call of a turn is rolled back" do
    with_script([text_reply("hi"), status_reply(500), text_reply("ok")]) do |agent, requests|
      agent.ask("hello").message
      task = agent.background { "result" }
      await(agent, task)

      failed = agent.ask("next")
      failed.join
      failed.error.should be_a Agent::ApiError
      agent.history.size.should eq 2
      agent.finished_tasks.should eq [task]

      agent.ask("again").message.content.should eq "ok"
      text_of(requests[2]["messages"][-1]).should contain "result"
      agent.finished_tasks.should be_empty
    end
  end

  it "counts a delivered outcome with the turn it arrived in when trimming" do
    with_script([text_reply("a"), text_reply("b")], max_history: 1) do |agent, _requests|
      agent.ask("one").message
      await(agent, agent.background { "result" })
      agent.ask("two").message
      agent.history.map(&.role.to_s).should eq %w[user user assistant]
      agent.history[0].task_result?.should be_true
    end
  end

  it "leaves an outcome delivered inside a loop with that turn when the next message starts a new one" do
    with_script([text_reply("b")], max_history: 1) do |agent, _requests|
      tc = Agent::ToolCall.new("c1", "lookup", "{}")
      agent.load_history([
        Agent::Message.user("one"),
        Agent::Message.assistant(nil, tool_calls: [tc]),
        Agent::Message.tool_result(tc, "found"),
        Agent::Message.user("<task_result>x</task_result>", source: :task),
      ])
      agent.ask("two").message
      agent.history.map(&.content).should eq ["two", "b"]
    end
  end

  it "shortens a long outcome to its start and end, flags it, and keeps the whole on the task" do
    with_script([text_reply("ok")], max_task_result_chars: 80) do |agent, requests|
      long = agent.background { (1..40).join("\n") { |i| "line #{i}" } }
      short = agent.background { "all good" }
      await(agent, long)
      await(agent, short)
      agent.ask("so?").message

      texts = requests[0]["messages"][0]["content"].as_s.split("\n\n")
      cut = texts.find!(&.includes?(long.id))
      cut.should contain %(status="done" elapsed=)
      cut.should contain %( warning="truncated">)
      body = cut.lines[1..-2].join("\n")
      body.size.should be <= 80
      body.should start_with "line 1\nline 2"
      body.should contain "characters omitted ...]"
      body.should end_with "line 39\nline 40"
      long.result.to_s.lines.size.should eq 40

      texts.find!(&.includes?(short.id)).should_not contain "warning="
    end
  end

  it "runs on_cancel hooks on cancel and timeout, so a block can kill its process" do
    with_script([] of HTTP::Server::Context -> Nil) do |agent, _requests|
      process = Process.new("sleep", ["30"])
      exited = Channel(Process::Status).new(1)
      task = agent.background("a sleeping process") do |t|
        t.on_cancel { process.terminate }
        exited.send(process.wait)
        "slept"
      end
      Fiber.yield
      task.wait(10.milliseconds).should be_false
      agent.cancel_task("nope").should be_false
      agent.cancel_task(task.id).should be_true
      agent.running_tasks.should be_empty
      task.wait(1.second).should be_true
      select
      when status = exited.receive then status.success?.should be_false
      when timeout(2.seconds) then fail "the process was not terminated"
      end
      task.status.cancelled?.should be_true

      calls = [] of Symbol
      late = agent.background(timeout: 20.milliseconds) do |t|
        t.on_cancel { raise "hook bug" }
        t.on_cancel { calls << :first }
        t.sleep(5.seconds)
        "x"
      end
      await(agent, late)
      late.status.timed_out?.should be_true
      # hooks run in their own fiber
      deadline = Time.instant + 1.second
      while calls.empty? && Time.instant < deadline
        Fiber.yield
      end
      calls.should eq [:first]
      late.on_cancel { calls << :late }
      calls.should eq [:first, :late]

      done = agent.background { "fine" }
      await(agent, done)
      done.on_cancel { calls << :never }
      calls.should eq [:first, :late]
    end
  end

  it "takes ids from the config's generator or the call, and refuses a duplicate" do
    counter = 0
    with_script([text_reply("ok")], task_id_generator: -> { "job-#{counter += 1}" }) do |agent, requests|
      first = agent.background { |t| t.sleep(5.seconds); "late" }
      first.id.should eq "job-1"
      agent.background(id: "deploy-7") { "shipped" }.id.should eq "deploy-7"
      agent.background { |t| t.sleep(5.seconds); "late" }.id.should eq "job-2"
      expect_raises(Agent::TaskIdError, "job-1 is already in use") { agent.background(id: "job-1") { "x" } }
      expect_raises(Agent::TaskIdError, "must not be empty") { agent.background(id: "") { "x" } }
      await(agent, agent.finished_tasks.first? || agent.running_tasks.find! { |t| t.id == "deploy-7" })
      expect_raises(Agent::TaskIdError, "deploy-7 is already in use") { agent.background(id: "deploy-7") { "x" } }

      agent.ask("hi").message
      text_of(requests[0]["messages"][0]).should contain %(<task_result id="deploy-7")
      agent.running_tasks.each(&.cancel)
    end
  end

  it "tells the model when a tool's task id is already in use" do
    with_script([tool_reply([{"c1", "start", "{}"}]), text_reply("ok")]) do |agent, requests|
      agent.register_tool("start", parameters: empty_schema) do
        agent.background(id: "same") { |t| t.sleep(5.seconds); "late" }
        agent.background(id: "same") { "x" }.accepted
      end
      agent.ask("go").message.content.should eq "ok"
      text_of(requests[1]["messages"].as_a.find! { |m| m["role"] == "tool" }).should eq "Error executing tool 'start': task id same is already in use"
    end
  end

  it "estimates its footprint from history and queued task results" do
    with_script([text_reply("ok")]) do |agent, _requests|
      agent.footprint.should eq Agent::Footprint.new(0, 0, 0, 0)

      agent.ask("hello").message
      before = agent.footprint
      before.messages.should eq 2
      before.history_bytes.should eq "hello".bytesize + "ok".bytesize
      before.tasks.should eq 0

      slow = agent.background("a slow thing") { |t| t.sleep(5.seconds); "late" }
      await(agent, agent.background { "x" * 1000 })
      with_tasks = agent.footprint
      with_tasks.tasks.should eq 2
      with_tasks.task_bytes.should eq 1000 + "a slow thing".bytesize
      with_tasks.bytes.should eq with_tasks.history_bytes + with_tasks.task_bytes
      slow.cancel
    end
  end

  it "has a prompt note, random 12-hex ids, and humanises durations" do
    Agent::Task::PROMPT_NOTE.should contain "<task_result"
    ids = Array.new(1000) { Agent::Task.generate_id }
    ids.each(&.should match(/\At-[0-9a-f]{12}\z/))
    ids.uniq.size.should eq 1000
    Agent::Task.humanize(250.milliseconds).should eq "250ms"
    Agent::Task.humanize(42.seconds).should eq "42s"
    Agent::Task.humanize(5.minutes).should eq "5m"
    Agent::Task.humanize(125.seconds).should eq "2m 5s"
    Agent::Task.humanize(2.hours).should eq "2h"
    Agent::Task.humanize(63.minutes).should eq "1h 3m"

    Agent::Task.truncate("short", 10).should eq({"short", false})
    Agent::Task.truncate("x" * 100, 10).should eq({"x" * 10, true})
    text, cut = Agent::Task.truncate("a" * 100 + "b" * 100, 60)
    cut.should be_true
    text.size.should be <= 60
    match = text.match!(/\A(a+)\n\[\.\.\. (\d+) of 200 characters omitted \.\.\.\]\n(b+)\z/)
    # the omitted count is exact: kept plus omitted is the whole
    (match[1].size + match[3].size + match[2].to_i).should eq 200
    Agent::Task.new(1.second).wrapped_result("x", [Agent::Task::Warning::Truncated]).should contain %(warning="truncated">)
  end
end

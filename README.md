# agent-cr

[![CI](https://github.com/anykeyh/agent.cr/actions/workflows/ci.yml/badge.svg)](https://github.com/anykeyh/agent.cr/actions/workflows/ci.yml)

A Crystal shard for building agentic loops with OpenAI-compatible APIs.

Inspired by [RubyLLM](https://rubyllm.com/) — a fantastic Ruby gem for interacting with AI models.

`agent-cr` wraps any OpenAI-compatible streaming chat-completion endpoint behind a fiber-based async interface. It handles streaming, tool calls, multimodal content, conversation history, and automatic tool resolution — all in a background fiber so your code never blocks.

---

## Installation

Add this to your `shard.yml`:

```yaml
dependencies:
  agent-cr:
    github: anykeyh/agent.cr
```

Then run:

```sh
shards install
```

> Built with Crystal >= 1.19 (uses the stdlib `sync` module)

> **Fiber safety:** Safe to call `#ask` from multiple fibers; the agent serialises requests through a single background fiber. Multi-thread mode (`-Dpreview_mt`) is not tested.

---

## Quick start

```crystal
require "agent"

config = Agent::Config.new(
  api_key: ENV["OPENAI_API_KEY"],
  system_prompt: "You are a helpful assistant."
)

agent = Agent.new(config)
resp = agent.ask("What is the capital of France?")

puts resp.message.content # => "The capital of France is Paris."
```

The `#ask` call returns immediately (the HTTP request runs in a background fiber). Calling `.message` blocks until the full response is ready.

---

## Streaming

When you want to show tokens as they arrive, use `.stream`:

```crystal
resp = agent.ask("Write a short poem about AI.")

resp.stream do |chunk|
  print chunk.text  # each token as it arrives
end

puts # newline after stream ends

# The final message is still available after streaming:
puts resp.message.content
puts "Used #{resp.metadata.total_tokens} tokens"
```

### Chunk kinds

Each chunk is tagged with its origin via `chunk.kind`:

```crystal
resp.stream do |chunk|
  case chunk.kind
  when Agent::Response::ChunkKind::Content
    print chunk.text
  when Agent::Response::ChunkKind::Reasoning
    print chunk.text.colorize(:dark_gray)  # dimmed for reasoning
  when Agent::Response::ChunkKind::ToolCallName
    puts "\n[Calling tool: #{chunk.text}]"
  when Agent::Response::ChunkKind::ToolCallArgs
    # tool argument deltas
  end
end
```

### Reasoning models

Reasoning streams as `ChunkKind::Reasoning` chunks and ends up in `message.reasoning`. Set how hard the model thinks with `reasoning_effort`. It's one setting for every provider; the OpenAI provider sends it as `reasoning_effort` (`:max` becomes `"xhigh"`, `:off` becomes `"none"`), or as OpenRouter's `reasoning: {effort: ...}` when the endpoint is openrouter.ai. Override it for one request on `ask`, or per provider call from a handler (`TurnContext#options`). For anything else the provider takes, use `extra_body`:

```crystal
agent = Agent.new(Agent::Config.new(model: "o4-mini", reasoning_effort: :medium))
agent.ask("Plan the migration", reasoning_effort: :high)

# A reasoning budget on OpenRouter
Agent::Config.new(
  api_endpoint: "https://openrouter.ai/api/v1",
  model: "anthropic/claude-sonnet-4.5",
  extra_body: {"reasoning" => JSON::Any.new({"max_tokens" => JSON::Any.new(4000_i64)})},
)
```

Reasoning models in a tool loop need their earlier reasoning back on every request, and some refuse the request without it. So the agent keeps the reasoning in `provider_data` and sends it back on later requests in the shape the endpoint returned it:

| Endpoint returned | Sent back as |
|---|---|
| `reasoning_details` blocks (OpenRouter), signed and encrypted ones included | the same blocks, unchanged and in order |
| a tool call's `extra_content` (Gemini's thought signature) | the same `extra_content` on that call |
| `reasoning_content` text (DeepSeek, which answers 400 without it; Kimi; vLLM) | `reasoning_content` |
| thinking chunks in `content` (Mistral) | a thinking chunk ahead of the answer in `content` |
| `reasoning` or `thinking` text (Groq, Ollama, OpenRouter's summary of its blocks) | nothing: display only |

This works across `dump`/`load` too. The data is kept under the provider's own key, with the endpoint and model that produced it, and goes back only to that endpoint and model, since signed blocks are refused anywhere else. A cancelled reply keeps its text but not this data, since a block cut off before its signature would be refused. Set `replay_reasoning: false` for a server that rejects these fields.

Not every model takes every effort level: OpenAI's `gpt-5` has `:minimal` but no `:off`, for example, and Gemini 3 can't turn reasoning off. An unsupported level is the provider's 400, so pick levels per model.

`resp.metadata.reasoning_tokens` is how many completion tokens went on reasoning, when the provider reports it. They are already counted in `completion_tokens`.

---

## Conversation history

History is tracked automatically — each `#ask` appends the user message, and the agent's reply is added to the conversation.

```crystal
agent.ask("My name is Alice.").join
agent.ask("What is my name?").join

puts agent.history.size # => 4 (user, assistant, user, assistant)
```

### Reset

Start a fresh conversation without creating a new agent:

```crystal
agent.reset
```

### Footprint

`agent.footprint` says how much an agent holds, cheaply enough to poll across every live session and alarm on the ones that have grown: the history's message count and bytes, and the background tasks still running or awaiting delivery with the bytes of their results. The counts are of the strings held, not the heap, so they compare sessions and set a threshold rather than measure exactly.

```crystal
fp = agent.footprint
fp.messages      # 42
fp.history_bytes # 318_204 — an inline image attachment counts whole
fp.tasks         # 1
fp.task_bytes    # 20_480
fp.bytes         # history_bytes + task_bytes
```

### History trimming

Set `max_history` to limit the number of conversation turns kept in memory. A turn is a user message and everything up to the next one (the assistant's reply, and with tools, its tool calls and their results); the oldest turns are dropped whole, so the kept history always starts at a user message and no tool result is ever orphaned.

```crystal
config = Agent::Config.new(
  api_key: ENV["OPENAI_API_KEY"],
  max_history: 10   # keep at most 10 user+assistant turns
)
```

With Anthropic models, trimming invalidates the thinking blocks kept after the cut. The Anthropic provider has the API drop them by default; see [Anthropic](#anthropic).

### Save and restore sessions

Serialise a session (including cache key for prompt caching affinity):

```crystal
# Save
File.write("session.json", agent.dump)

# Restore — register tools first, then load session data:
config = Agent::Config.new(api_key: ENV["OPENAI_API_KEY"])
agent = Agent.new(config)
agent.register_tool(...) { |args| ... }
agent.load(File.read("session.json"))
agent.ask("Continue where we left off")  # 🚀 cache hit
```

---

## Attachments (multimodal)

Pass file paths, URLs, or data URIs as `attachments`. Local files are
auto-detected by extension: images become `image_url` parts, text files
are inlined, audio becomes `input_audio`, and everything else is sent as
`file` parts.

```crystal
# Remote image URL
resp = agent.ask(
  "What's in this image?",
  attachments: ["https://example.com/photo.jpg"]
)
resp.join
puts resp.message.content

# Local file — MIME auto-detected from extension
resp = agent.ask(
  "Review this document",
  attachments: ["./doc.md"]
)
```

See `Agent::ContentPart.from_path` for the MIME detection rules.

---

## Tools / function calling

### Registered tools with auto-resolve (recommended)

Register tools with a callback block. The agent automatically resolves tool calls inline — no manual while-loop needed.

```crystal
# Define JSON Schema parameters using a plain Crystal NamedTuple
params = Agent::JSONConverter.from({
  type: "object",
  properties: {
    city: {type: "string", description: "The city name"},
  },
  required: ["city"],
})

agent.register_tool("get_weather", "Get the current weather for a city",
  parameters: params
) do |args|
  city = args["city"]?.try(&.as_s) || "unknown"
  "The weather in #{city} is sunny."
end

# Registered tools are automatically included in all #ask calls.
# When the model calls a tool, the agent executes it and re-asks the
# model — all in the background fiber. The Response you get back is
# the final one (after all tool resolutions).
resp = agent.ask("What's the weather in Paris?")
resp.stream { |chunk| print chunk.text }
puts resp.message.content # => "The weather in Paris is sunny."
```

You can `register_tool` multiple times. All registered tools are merged into every request automatically. If a per-request tool has the same name as a registered tool, the registered callback wins.

When the model calls a tool that isn't registered or is disabled, the agent answers that call with an error result and the model carries on. When a message calls a tool passed with `ask(tools:)`, the agent runs none of its tools and hands the whole message back to you to dispatch.

A callback that raises gives the model an error result. Only the message of an `Agent::ToolError` is shown to the model; any other exception is logged, and the model sees only that the tool failed, so internal details never reach it:

```crystal
agent.register_tool("get_weather", ...) do |args|
  city = args["city"].as_s
  raise Agent::ToolError.new("Unknown city '#{city}'") unless known?(city)
  fetch_weather(city) # a network error here reaches the model as "the tool failed"
end
```

A `Handler` sees the exception on `ToolCallContext#error` after the callback runs. Use it to record why a tool failed, or carry your own fields on a `ToolError` subclass and return a different result for the model:

```crystal
class QuotaError < Agent::ToolError
  getter tenant : String

  def initialize(message, @tenant)
    super(message)
  end
end

class ToolFailures < Agent::Handler
  def handle(ctx : Agent::ToolCallContext, next_proc) : Agent::Message
    result = next_proc.call(ctx)
    case err = ctx.error
    when QuotaError
      Metrics.tool_quota_hit(err.tenant)
      Agent::Message.tool_result(ctx.tool_call, "Quota reached; tell the user.", is_error: true)
    when Exception
      Sentry.capture(err) # anything that isn't a ToolError
      result
    else
      result
    end
  end
end
```

### Tool activation control

You can register a tool as initially **disabled** — the definition is stored but not sent to the model until you `enable_tool`:

```crystal
agent.register_tool("roll_dice", "Roll dice in NdS format",
  parameters: params,
  enabled: false
) do |args|
  # ...
end

# Later, activate it mid-conversation:
agent.enable_tool("roll_dice")

# Or deactivate without unregistering:
agent.disable_tool("roll_dice")

# Query which tools are currently active:
agent.enabled_tools # => ["get_weather"]
```

### Session persistence

> Saved session state (including enabled tools) is serialized via `agent.dump`. On restore, create a fresh agent, register your tools, then call `agent.load(data)` — it restores the session identity, history, and re-enables any tools that match:

```crystal
agent = Agent.new(config)
agent.register_tool("get_weather", ...) { |args| ... }
agent.register_tool("roll_dice", ..., enabled: false) { |args| ... }

# Restore session identity, history, and enabled-tool state:
saved = File.read("session.json")
agent.load(saved)  # warnings for tools not re-registered
```

### Background tasks

A tool that has to wait, for a build to finish, a condition to hold or a slow command to return, need not hold the conversation. `Agent#background` runs a block in its own fiber, and the tool answers at once with `Task#accepted`. The model ends its turn and the agent is free for other work. When the block returns, raises or hits its timeout, the outcome is queued and delivered to the model at the start of the next model call: the next `#ask`, a `#resume`, or the next iteration of a tool loop that is still running.

```crystal
agent.register_tool("wait_for_build", "Wait for a build to finish", parameters: schema) do |args|
  id = args["id"].as_s
  task = agent.background("build #{id} to finish", timeout: 30.minutes) do |t|
    until build_done?(id)
      break unless t.sleep(10.seconds) # false once cancelled or timed out
    end
    build_done?(id) ? build_summary(id) : "build still running"
  end
  task.accepted # "Started background task t-3fa9c1d2 (build 42 to finish). It times out after 30m. ..."
end

resp = agent.ask("Tell me when build 42 is done")
resp.message.content # => "I'll let you know once it finishes."

# Fires from the task's fiber; park the conversation until then.
agent.on_task_finished { |task| scheduler.wake(conversation_id) }

# Later, when the scheduler comes back to this conversation. Safe even if
# nothing is waiting: then no request is made and the response repeats the
# model's latest reply, with `from_history?` set.
agent.resume.message.content # => "Build 42 passed: 312 tests, 0 failures."
```

The outcome reaches the model as a user-role message whose content is wrapped in a tag it can tell from a person's words:

```
<task_result id="t-3fa9c1d2" tool="wait_for_build" description="build 42 to finish" status="done" elapsed="2m 14s">
passed: 312 tests, 0 failures
</task_result>
```

`status` is `done`, `failed`, `timed_out` or `lost` (the agent was restarted while the task ran). Add `Agent::Task::PROMPT_NOTE` to the system prompt so the model knows the tag and does not poll for results. `Message#source` tells the application which user messages are a person's (`human`), the application's (`application`, via `ask(text, source: :application)`) or a task's (`task`); it is kept through `dump`/`load` and never goes on the wire.

Limits and lifecycle:

- `Config#max_background_tasks` (default 4) caps the tasks running per agent. One more raises `Agent::TaskLimitError`, which from a tool callback becomes an error result the model can act on.
- `Config#background_task_timeout` (default 5 minutes; seconds or a `Time::Span`) is the timeout; `background(timeout:)` overrides it per task.
- A fiber cannot be killed, so stopping is cooperative. `Task#cancel`, `Agent#cancel_task(id)` or a timeout make `Task#cancelled?` true and `Task#sleep` return false, and run the task's `Task#on_cancel` hooks in a fiber of their own, where a block tears down what it is blocked on:

  ```crystal
  agent.background("the test suite", timeout: 20.minutes) do |task|
    process = Process.new("make", ["test"], output: out = IO::Memory.new, error: out)
    task.on_cancel { process.terminate }
    status = process.wait
    raise Agent::ToolError.new("make test exited #{status.exit_code}:\n#{out}") unless status.success?
    out.to_s
  end
  ```
- A failure is reported with `status="failed"`. Only an `Agent::ToolError`'s message reaches the model as the body; any other exception is logged and reported as "the task failed". A `Handler` for `TaskContext` runs as each outcome is delivered: it sees the exception on `ctx.error` and can set `ctx.result` to tell the model why, or to shorten a long output and keep the whole for a tool of yours to read or search later. A handler that raises is logged and the outcome delivered as it is.
- `#reset`, `#load` and `#close` cancel every running task of the agent and drop undelivered outcomes. `#dump` records tasks; one still running when dumped comes back as `lost`, so the model learns its result is never coming.
- Outcomes are delivered with the next `#ask` of a user message or `#resume`, where they follow a reply. `Config#deliver_tasks_mid_turn` (off by default) also delivers them right after tool results, inside a tool loop still running or with a manual `ask(tool_results)`. That is prompter, but a user message directly after tool results is refused by some OpenAI-compatible backends (Mistral's API and chat templates), and once such a history exists every later request fails; turn it on for backends you know accept it.
- `#awaiting_tasks?` is true while tasks run or outcomes wait; `#running_tasks` and `#finished_tasks` list them, `Task#wait` blocks until one ends, and `#discard_task(id)` drops an outcome before the model sees it. `#on_task_finished` returns its callback for `#off_task_finished`.
- Task ids are `t-` and 12 random hex characters. `Config#task_id_generator` (a `-> String`) substitutes your own scheme, and `background(id:)` names one task; an id still running or awaiting delivery raises `Agent::TaskIdError`, which the model sees when it happens inside a tool.
- `#ask` and `#resume` raise when called from the agent's own fiber (inside a tool callback or handler), since they would wait for themselves.

#### Outcomes from outside

Not every outcome comes from a block. A webhook, a scheduler checking state every so often, or a person may be what settles the question. Start the task without a block and finish it from wherever the answer appears, from any fiber:

```crystal
agent.register_tool("rollback", "Roll a deployment back", parameters: schema) do |args|
  start_rollback(args["id"])
  agent.background("rollback of #{args["id"]} to settle", timeout: 45.minutes).accepted
end

# later, in your own checker
agent.complete_task(id, "deployment 42 reverted; the service has no alarm")
# or
agent.fail_task(id, "the rollback did not take: 2 containers still Down")
```

The model reads the same `<task_result>` as for any task, so it needs no second format, and the outcome goes through the same handlers, cap, limit, timeout, cancel and `dump`. `complete` and `fail` queue the outcome before they return, so "load, complete, ask" delivers it in that turn. A task like this still running when dumped stays running after `load`, with the time it had left, for your checker to finish; one whose time ran out while the process was down times out at once. `Task#complete` also ends a block's task early: its `Task#sleep` returns false. If a success needs no words from the model, `discard_task(id)` after completing drops the outcome unread.

An application that keeps its own record of pending work and loads a conversation only to run a turn has no agent to complete when the outcome arrives. It builds the record itself and queues it as it loads:

```crystal
agent = Agent.load(config, saved_json)
task = Agent::Task.new(45.minutes, tool: "rollback", description: row.description, id: row.task_id, started_at: row.started_at)
task.fail("the rollback did not take: 2 containers still Down")
agent.queue_task(task)
agent.resume
```

#### Restarts

A task is a closure, so it cannot survive the process. After a restart `load` queues each task the dump had running as `status="lost"`, and the model is told its result is never coming. To do better, define the work once under a name and start tasks by name and data; `load` then relaunches them from the dump:

```crystal
agent.register_task_kind("wait_for_build") do |task|
  id = task.data.not_nil!["id"].as_s
  until build_done?(id)
    break unless task.sleep(10.seconds)
  end
  build_done?(id) ? build_summary(id) : "build still running"
end

agent.register_tool("wait_for_build", "Wait for a build to finish", parameters: schema) do |args|
  agent.background("build #{args["id"]} to finish", kind: "wait_for_build", data: args["id"]).accepted
end
```

Register kinds before `load`, as with tools. A relaunched task keeps its id, description and start time, so the model's earlier acknowledgement still matches; its timeout starts over. `data` is kept through dump and load and never shown to the model. Only tasks whose work is safe to run again should have a kind; anything else is better left to become `lost`.

#### Long outputs

An outcome longer than `Config#max_task_result_chars` (30,000 by default; nil for no limit) is delivered as its start and end around a line saying how much is missing, and the tag carries `warning="truncated"` so the model knows it is not seeing everything. `Task#result` keeps the whole. That is a backstop, not a plan: a task that produces a lot (a log, a test run, a crawl) should return what matters and keep the rest where a tool can reach it. The model already knows the task id, so key the store by it and register tools to read or search it on demand. This is how Claude Code handles its own background tasks: a short notification, and the full output in a file the model greps only if it needs to.

```crystal
outputs = {} of String => String

agent.register_tool("run_tests", "Run the test suite in the background", parameters: schema) do
  agent.background("the test suite", timeout: 20.minutes) do |task|
    log = run_suite(task)                       # the whole output
    outputs[task.id] = log
    failures = log.lines.select(&.starts_with?("FAIL"))
    "#{failures.size} failures out of #{log.lines.size} lines:\n#{failures.first(10).join("\n")}"
  end.accepted
end

agent.register_tool("search_task_output", "Search a task's full output", parameters: search_schema) do |args|
  log = outputs[args["task_id"].as_s]? || raise Agent::ToolError.new("no output for that task")
  log.lines.select(&.includes?(args["pattern"].as_s)).first(50).join("\n")
end
```

A `TaskContext` handler can apply the same split after the fact, shortening `ctx.result` and stowing the original, when the task code is not yours to change; `ctx.warnings` is where it says so.

Two user messages in a row (an outcome, then what the person typed) go as one user message on the wire, since many OpenAI-compatible backends refuse roles that do not alternate. Plain texts are joined with a blank line, so the content stays a string; it becomes content parts only when an attachment is among them.

### Disable auto-resolve

To handle tool calls manually (e.g. only some tools are registered, or you need user approval):

```crystal
config = Agent::Config.new(
  api_key: ENV["OPENAI_API_KEY"],
  auto_execute_tools: false
)
```

With `auto_execute_tools: false`, the agent returns tool calls to the caller and you dispatch manually in a loop. You pass tool definitions per-request (see [#tool-definitions](#tool-definitions) for the syntax) and implement the tool logic yourself with a `case`:

```crystal
resp = agent.ask("What's the weather in Paris?", tools: [weather_tool])
resp.stream { |chunk| print chunk }
msg = resp.message

while msg.has_tool_calls?
  results = msg.tool_calls.not_nil!.map do |tc|
    args = JSON.parse(tc.arguments).as_h
    case tc.name
    when "get_weather"
      result = fetch_weather(args["city"].to_s)
      Agent::Message.tool_result(tc, result)
      # or, when the tool failed: tool_result(tc, "...", is_error: true)
    else
      Agent::Message.tool_result(tc, "Unknown tool: #{tc.name}")
    end
  end

  resp = agent.ask(results, tools: [weather_tool])
  resp.stream { |chunk| print chunk }
  msg = resp.message
end

puts msg.content
```

### Tool definitions

When passing tools per-request without `register_tool`, you build a `Tool` object directly. Use `Agent::JSONConverter` to avoid the `JSON::Any.new(...)` boilerplate:

```crystal
weather_tool = Agent::Tool.new(Agent::Tool::FunctionDef.new(
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: Agent::JSONConverter.from({
    type:       "object",
    properties: {
      city: {type: "string", description: "The city name"},
    },
    required: ["city"],
  })
))
```

Then pass it to `#ask`: `agent.ask("...", tools: [weather_tool])` — see the manual dispatch loop above for how to handle tool calls when they arrive.

---

## Handler chain

`agent-cr` provides a middleware-style handler chain that wraps every stage of the request pipeline. Handlers can inspect, mutate, or short-circuit requests before they reach the API, and responses before they reach your code.

### Architecture

Five context types correspond to the five pipeline stages. A `Handler` subclass overrides the `handle` overloads it cares about; each receives the context and a `next_proc` that invokes the rest of the chain:

| Context | Stage | Attack surface |
|---|---|---|
| `TurnContext` | Before the HTTP request is built | Mutate `ctx.messages` and `ctx.tools` before they're sent to the model |
| `ChunkContext` | Each streamed SSE delta | Replace `ctx.chunk` to rewrite tokens in-flight |
| `ToolCallContext` | Before a tool callback is executed | Approve/deny tool execution, cancel the response |
| `TaskContext` | As a background task's outcome is delivered to the model | Replace `ctx.result`: explain a failure from `ctx.error`, shorten a long output |
| `ErrorContext` | After an exception escapes the pipeline | Wrap or replace the error before it's recorded |

### Writing a handler

```crystal
class LoggingHandler < Agent::Handler
  # Override only the stages you care about — the default is a pass-through.
  def handle(ctx : Agent::TurnContext, next_proc) : {Agent::Message, Agent::Usage, String?}
    puts "Sending #{ctx.messages.size} messages..."
    result = next_proc.call(ctx)
    puts "Received response."
    result
  end
end
```

> **Mutation rules:** `Chunk` is a class; reassign `ctx.chunk = Chunk.new(...)` to rewrite chunks. `Message` is immutable; replace entries in `ctx.messages` with new `Message` objects to rewrite user input.

### Registration

Append handlers with `agent.use`. Handlers run left-to-right in registration order — the first handler registered wraps the outermost layer:

```crystal
agent.use(LoggingHandler.new)
agent.use(TokenBudgetHandler.new)
```

### Short-circuiting

A handler can prevent the rest of the chain from running by returning a value without calling `next_proc`. This is useful for gating tool calls, denying requests, or serving cached responses:

```crystal
class GateKeeper < Agent::Handler
  def handle(ctx : Agent::ToolCallContext, next_proc) : Agent::Message
    if ctx.tool_call.name == "dangerous_op"
      Agent::Message.new(
        role: Agent::Role::Tool,
        content: "Tool '#{ctx.tool_call.name}' denied by policy.",
        tool_call_id: ctx.tool_call.id,
        name: ctx.tool_call.name,
      )
    else
      next_proc.call(ctx)
    end
  end
end
```

### Example

See `examples/handlers.cr` for a working CLI that demonstrates four handlers:

| Handler | Context | What it does |
|---|---|---|
| `TokenBudgetHandler` | `TurnContext` | Times the turn, prints `tok/s`, enforces a session token budget |
| `WordReplaceHandler` | `TurnContext` + `ChunkContext` | Rewrites words on both sides of the conversation (`hello` → `howdy`, `world` → `galaxy`) |
| `ToolConfirmHandler` | `ToolCallContext` | Prompts the user before executing tools; short-circuits on decline |
| `ErrorRetryHandler` | `ErrorContext` | Colorizes errors by type |

```sh
crystal run examples/handlers.cr
```

---

## Provider system

`agent-cr` uses a pluggable provider abstraction. The default provider is `Agent::Provider::OpenAI`, which works with any OpenAI-compatible API (OpenAI, OpenRouter, local llama.cpp, Ollama, etc.). `Agent::Provider::Anthropic` speaks Anthropic's Messages API.

### Anthropic

```crystal
config = Agent::Config.new(
  api_key: ENV["ANTHROPIC_API_KEY"],
  api_endpoint: Agent::Provider::Anthropic::DEFAULT_ENDPOINT, # or a gateway, path prefix and all
  model: "claude-opus-5-5",
  reasoning_effort: :high,
)
agent = Agent.new(config, provider: Agent::Provider::Anthropic.new(config))
```

It reads `api_key`, `api_endpoint`, `model`, `max_tokens` (default 64,000), `reasoning_effort`, `extra_headers` and `extra_body` from the config. It doesn't send `temperature`, which API version 2026-10-21 refuses.

- **Thinking:** adaptive, with `display: "summarized"`, so `message.reasoning` holds a readable summary. Effort goes to `output_config.effort`. Current models can't turn thinking off, so `:off` and `:minimal` both ask for `"low"`. Pass `thinking: false` for a model without adaptive thinking (Haiku 4.5 and older), and `display: "omitted"` to stream no thinking text.
- **Replay:** every reply's content blocks are kept in `provider_data` and sent back byte for byte, including signed thinking, `redacted_thinking` and server-tool blocks. They go only to the endpoint they came from. A cancelled reply keeps just its text.
- **Trimmed history:** the API checks each thinking block against the conversation that produced it, so `max_history` trimming invalidates the blocks after the cut. By default the provider asks the API to drop those blocks (`prefix_mismatch: "drop_block"`, with the `thinking-binding-controls-2026-08-01` beta) instead of refusing the request. Pass `"error"` to be refused instead, or `nil` to send nothing and take your account's default (refused, for accounts created on or after 2026-08-31).
- **Caching:** markers on the system prompt and the latest turn, plus extra ones in a long turn; see [Prompt caching](#prompt-caching). Every system message goes into the system prompt, so adding one mid-conversation restarts the cache and drops earlier thinking.
- **Tools:** inputs stream as they're generated (`eager_input_streaming`; pass `eager_input_streaming: false` for a gateway that rejects the field). The agent never runs the tool calls of a reply that hit its token limit or a refusal.
- **Betas and options:** `anthropic-beta` in `extra_headers` merges with the provider's own beta, and `extra_body` adds request fields such as `fallbacks`.

### Switch endpoint for a compatible API

```crystal
# Any OpenAI-compatible endpoint works:
config = Agent::Config.new(
  api_key: "not-needed",
  api_endpoint: "http://localhost:8080/v1",
  model: "local-model",
)
```

### Custom provider

Implement `Agent::Provider::Base` to support a different wire format:

```crystal
class MyProvider < Agent::Provider::Base
  def base_uri : URI
    URI.parse("https://myapi.com/v1")
  end

  def build_request(messages, tools) : NamedTuple(path: String, headers: HTTP::Headers, body: String)
    # Build path, auth headers, and JSON body
    {path: "/chat", headers: HTTP::Headers{"X-API-Key" => "secret"}, body: my_body.to_json}
  end

  def parse_stream(io, response, cancel) : {Message, Usage, String?}
    # Parse response body, push chunks, return final message
  end

  def close : Nil
  end
end

agent = Agent.new(config, provider: MyProvider.new(...))
```

---

## Response API

| Method / Property | Description |
|---|---|
| `.stream { \|chunk\| }` | Yield each delta as it arrives from the API |
| `.message` | Block until the final `Message` is ready, then return it |
| `.metadata` | Block until `Usage` (token counts) is ready, then return it. With `auto_execute_tools`, the counts of every tool-loop iteration are summed, so this is the whole turn's usage. `cached_tokens` and `cache_creation_tokens` say how much of the prompt the provider's cache served or stored, and `reasoning_tokens` how much of the completion went on reasoning, when it reports that |
| `.join` | Block until both message and metadata are ready |
| `.finished?` | Poll whether the response is complete |
| `.error?` | Whether the response represents a failed request |
| `.error` | The `Agent::Error` if the request failed, or `nil` |
| `.finish_reason` | Why the stream ended (`"stop"`, `"length"`, `"tool_calls"`, etc.) |
| `.cancel` | Request cancellation of the in-flight request |
| `.cancelled?` | Whether cancellation was requested |

### Message fields

```crystal
msg = resp.message
msg.content        # String? — text content
msg.reasoning      # String? — reasoning content (DeepSeek, Qwen, etc.)
msg.role           # Agent::Role — System, User, Assistant, Tool
msg.tool_calls     # Array(ToolCall)? — function calls from the model
msg.tool_call_id   # String? — id for tool result messages
msg.content_parts  # Array(ContentPart)? — multimodal parts (text + images)
msg.has_tool_calls? # Bool — convenience check
msg.is_error       # Bool? — on a tool result: the tool failed (set by the loop for its own errors)
msg.provider_data  # JSON::Any? — data each provider keeps, under its own key, to replay the message
                   # (OpenAI provider: {"openai" => {"endpoint" => ..., "model" => ..., "reasoning_details" => [...]}}; ToolCall has the same slot)
```

---

## Configuration

```crystal
Agent::Config.new(
  api_key:            String?,             # OpenAI API key (or nil for local models)
  api_endpoint:       String,              # default: "https://api.openai.com/v1"
  model:              String,              # default: "gpt-4o"
  system_prompt:      String?,             # optional system message prepended to every request
  max_tokens:         Int32?,              # optional max completion tokens
  temperature:        Float64?,            # optional sampling temperature (0.0-2.0)
  read_timeout:       Time::Span | Int32?, # optional HTTP read timeout (seconds or span)
  connect_timeout:    Time::Span | Int32?, # optional HTTP connect timeout (seconds or span)
  max_history:        Int32?,              # optional max conversation turns (0 or nil = no limit)
  auto_execute_tools: Bool,                # default: true
  extra_headers:      Hash(String, String)?, # optional extra HTTP headers
  max_tool_iterations: Int32?,             # default: 100 — safety limit for tool loops
  prompt_cache_key:   String?,             # optional explicit prompt cache key
  include_usage:      Bool,                # default: true — ask for token counts on streamed completions
  *,
  # Per-read SSE timeouts (fail-fast on hung connections):
  first_byte_timeout_min:          Time::Span | Int32 = 60,  # TTFT floor (seconds or span)
  first_byte_timeout_max:          Time::Span | Int32 = 300, # TTFT ceiling
  first_byte_timeout_base:          Time::Span | Int32 = 3,   # routing + provider queue baseline
  first_byte_timeout_ms_per_token:  Int32 = 3,                # ms added per estimated input token (~4 chars/token)
  idle_byte_timeout:                Time::Span | Int32 = 60,  # gap allowed between any two consecutive SSE lines
  compute_first_byte_timeout:       Bool = true,              # false → dynamic first-byte disabled, both phases use idle_byte_timeout
  # Retries of a model call that fails transiently before streaming any output:
  max_retries:                      Int32 = 2,                # 0 turns retrying off
  retry_base_delay:                 Time::Span | Int32 = 1,   # first backoff; doubles per retry, with jitter
  retry_max_delay:                  Time::Span | Int32 = 30,  # longest wait; a longer Retry-After isn't retried
  reasoning_effort:                 Agent::ReasoningEffort? = nil, # :off, :minimal, :low, :medium, :high, :xhigh, :max — mapped per provider
  extra_body:                       Hash(String, JSON::Any)? = nil, # extra request fields; never replaces one the provider sets
  replay_reasoning:                 Bool = true,              # send reasoning back on later requests; off for a server that rejects it
  prompt_caching:                   Agent::PromptCaching = :explicit, # :explicit, :automatic, :off — see "Prompt caching"
  cache_ttl:                        Agent::CacheTTL = :five_minutes,  # or :one_hour
)
```

#### Per-read SSE timeouts (fail-fast on hung streams)

The HTTP client's `read_timeout` is per-read: as long as *each* SSE line
arrives within the idle budget, long-lived streams run unbounded. Two
budgets are enforced independently:

- **First-byte** (`compute_first_byte_timeout`): scales with prompt size so
  short prompts fail fast (60s floor) while huge prompts get a generous
  time-to-first-token budget (300s ceiling). Formula:
  `clamp(base + (prompt_bytes / 4) * ms_per_token, min, max)`.
- **Inter-byte (idle)**: 60s fixed by default — gap allowed between any two
  consecutive lines mid-stream. Any byte arrival (comment lines like
  `: OPENROUTER PROCESSING`, empty lines, the `[DONE]` sentinel) resets it.

When the dynamic first-byte is disabled (`compute_first_byte_timeout: false`),
both the first read and subsequent reads fall back to `idle_byte_timeout`.

```crystal
config = Agent::Config.new(
  api_key: ENV["OPENAI_API_KEY"],
  # Tighter fail-fast: 10s first byte, 5s inter-line gap
  first_byte_timeout_min: 10.seconds,
  first_byte_timeout_max: 60.seconds,
  idle_byte_timeout: 5.seconds,
)
# → any prompt that emits nothing for 10s, or any mid-stream 5s stall,
#   raises Agent::IdleTimeoutError without further retries.
```

---

## Error handling

Errors provide a consistent `"Agent error: ..."` prefix on the message content, so you can pattern-match programmatically:

```crystal
resp = agent.ask("Hello")
resp.join

if resp.error?
  puts "Request failed: #{resp.error.message}"
  puts resp.error.class  # Agent::ConnectionError, Agent::ApiError, etc.
  puts "(the text shown above is incomplete)" if resp.partial?
else
  puts resp.message.content
end
```

Where a failure lands decides its type and whether it is retried:

- Before the provider answers: `ConnectionError`, or `ApiError` for a non-2xx status.
- After a 200: `StreamError` (or `IdleTimeoutError` for a stall).
- A transient failure is retried (`max_retries`) only while the call has streamed nothing. After that, `resp.partial?` is true and the caller decides whether to ask again.

Error types:

| Error | When |
|---|---|
| `Agent::ApiError` | API returned a non-2xx status code, after any retries. Has `.status_code`, `.body` (start of the response body), `.headers`, `.request_id` and `.retry_after`; `.message` includes the provider's error message |
| `Agent::StreamError` | The provider accepted the request (HTTP 200) but the stream failed: an error event (`.code` holds its code), a dropped connection, a stream that ended before the model finished, or one that couldn't be read. `.cause` holds the original exception |
| `Agent::ConnectionError` | Network / connection failure before the provider answered |
| `Agent::CancelledError` | Caller called `.cancel` on the response |
| `Agent::IdleTimeoutError` | Provider stream stuck — no bytes within the first-byte or idle per-read timeout (`phase = :first_byte` or `:idle`, `timeout = Time::Span`) |
| `Agent::ToolLoopError` | Tool auto-resolve exceeded `max_tool_iterations` |
| `Agent::ToolError` | Raised by a tool callback: its message is shown to the model (not a response error) |
| `Agent::SessionLoadError` | Loading a saved session failed due to corrupt or missing fields |
| `Agent::ClosedError` | An operation was attempted on a closed agent |

---

## Best practices

### Timeouts

Set timeouts in production — the defaults for the legacy `read_timeout` /
`connect_timeout` are unbounded (nil). The per-read SSE timeouts `first_byte_timeout_*` /
`idle_byte_timeout` have safe defaults (60s idle / 60s first-byte floor) and take
precedence over `read_timeout` for streaming requests:

```crystal
config = Agent::Config.new(
  api_key: ENV["OPENAI_API_KEY"],
  connect_timeout: 10.seconds,         # connect budget (total)
  # Streaming SSE timeouts (per-read, NOT total-time):
  first_byte_timeout_min: 60.seconds,  # floor on TTFT budget
  first_byte_timeout_max: 300.seconds, # ceiling even for huge prompts
  idle_byte_timeout: 60.seconds,       # gap allowed between two consecutive SSE lines
)
```

If a provider stream stalls past these budgets the agent raises
`Agent::IdleTimeoutError` (with `.phase = :first_byte` or `:idle`) and the
HTTP connection is rebuilt to avoid reusing a socket in an unknown state.
Cancellation wins over timeout when both fire (see `process_request_loop`).

### Cleanup

Close the agent when done to shut down the background fiber and release the HTTP connection pool. **Important:** Dropping the reference without calling `#close` leaks the background fiber — it will block indefinitely waiting for requests on its channel.

```crystal
agent.close
```

### Prompt caching

A provider's cache holds the start of a prompt. A later request that begins with the same bytes reads that part from the cache, at a tenth of the price or less, instead of processing it again.

Who places the cache markers depends on the provider:

| Provider | Caching |
| --- | --- |
| OpenAI, DeepSeek, Grok, Gemini (implicit) | Automatic. `prompt_caching` and `cache_ttl` are ignored. |
| `Agent::Provider::Anthropic` | Markers the provider places. |
| Claude models on OpenRouter (OpenAI provider) | The same markers, on OpenRouter's content parts. |

`prompt_caching` picks the markers:
- **`:explicit`** (default): one marker on the system prompt, which covers the tools ahead of it, or on the last tool when there's no system prompt. Another goes on the latest user or tool-result turn, so each call of a tool loop reads the transcript from the cache. On Anthropic, a turn longer than the cache's 20-position lookback also gets a marker every 15 positions, within the API's limit of four. A run of parallel tool calls or results counts as one position.
- **`:automatic`**: the system-prompt marker, plus a top-level `cache_control` so the API places the conversation's marker itself. Anthropic recommends it for agent loops, but some platforms reject the field.
- **`:off`**: no markers.

`cache_ttl: :one_hour` keeps entries for an hour instead of five minutes. A write then costs twice the input price instead of 1.25 times. It pays off when a conversation pauses for longer than five minutes, such as a chat picked up later in the day.

What makes the cache miss, without any error:
- **Editing the history:** trimming with `max_history`, removing or changing an earlier message, or a handler adding text it later takes out. Everything after the edit is processed again.
- **Changing the system prompt or the tools:** a per-request tool (`ask(tools:)`), registering or enabling a tool mid-conversation, or a system message added later. This restarts everything, the tools and system prompt included.
- **Per-request content at the start:** a timestamp or request id in the system prompt, for example. Put it in the user turn instead.
- **Changing `reasoning_effort` or the thinking settings mid-conversation.** The conversation is processed again. Pick them per agent, not per request.
- **A short prompt:** Anthropic caches nothing under the model's minimum, 512 tokens on current models and 1,024 or 4,096 on older ones. OpenAI's minimum is 1,024.
- **Parallel requests:** an entry can be read only once its first response starts streaming, so requests with the same prefix sent at the same moment each pay full price.

`Response#metadata` shows whether caching works: `cached_tokens` counts tokens read from the cache, and `cache_creation_tokens` counts tokens written. A tool loop that reads almost nothing back has broken its prefix.

OpenAI also routes requests by `prompt_cache_key`. Set one explicitly, or use `agent.load` to restore a previous session; both preserve cache affinity:

```crystal
# Auto-generated cache key tied to the session
agent = Agent.new(config)

# Restored session keeps the same cache_key:
agent = Agent.new(config)
agent.load(saved_session_string)
```

---

## Examples

### 🎲 Dungeons & Dragons — Text Adventure

`examples/rpg.cr` is a fully playable D&D-style parody adventure. You pick a hero class, the LLM becomes your Dungeon Master, and game mechanics (HP, damage, ability checks, inventory) are enforced by registered Crystal tools — no cheating allowed, not even for the DM.

```sh
crystal run examples/rpg.cr
```

**How it works:**

- The system prompt instructs the DM to narrate in the spirit of *Le Donjon de Naheulbeuk* and Monty Python — absurdist, self-aware, and allergic to drama.
- Twelve registered tools (`roll_check`, `take_damage`, `heal`, `add_item`, etc.) handle all rule interactions. The DM calls them via function calling; the agent auto-resolves them in the background.
- Sessions auto-save after every turn. You can quit, come back, and resume exactly where you left off — the agent's history and session identity are preserved.
- Reasoning tokens (DeepSeek, Qwen, etc.) are hidden behind a spinner: *"The DM is preparing your adventure..."* while the model thinks, then replaced by the actual narration.
- Tool call internals are kept off-screen. The DM narrates outcomes naturally — you see roll *"🎲 15 — Success!"* as prose.

**Game mechanics? More like game *suggestions*.** The DM has tools, yes. The DM *should* use them. But nobody said the DM was competent. Every `roll_check` is faithfully executed by Crystal, but whether the DM remembers to call `is_alive` after dealing damage is entirely between them and their questionable deity. The wizard has a spellbook; the rogue has lockpicks, and the warrior a _big fat sword_.

```sh
# Optional flags:
crystal run examples/rpg.cr -- --endpoint http://localhost:8080/v1 --model llama3
```

Hero classes include Fighter (solves problems with violence), Wizard (solves problems with different violence), Rogue (solves problems by pretending they aren't there), and Cleric (solves problems by lecturing everyone about it). Sessions are saved to `examples/rpg/`. Try `/hero` to see your sheet, `/reset` when you inevitably die, and `/help` when you forget which button does what.

---

## Contributing

See [AGENTS.md](AGENTS.md) for the architecture overview and development workflow.

### Quick commands

```sh
shards install           # install dependencies
crystal spec             # run tests
crystal tool format      # format code
bin/ameba                # lint
```

---

## License

MIT

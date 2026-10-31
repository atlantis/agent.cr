# AGENTS.md — Architecture & quirks

## Overview

`agent-cr` wraps an OpenAI-compatible streaming chat-completion endpoint behind
a fiber-based async interface. The `Agent` class owns a background fiber that
serialises requests — each `#ask` sends a `Request` on an unbuffered channel,
the fiber processes it, and signals completion through the `Response` object's
channels.

## File layout

```
src/
  agent.cr              # entry point, requires sub-files
  agent/
    version.cr           # VERSION constant
    config.cr            # Agent::Config — generic fields (system_prompt, max_history, timeouts)
    request_options.cr   # Agent::ReasoningEffort, Agent::RequestOptions (per-#ask overrides)
    error.cr             # Agent::Error hierarchy
    json_converter.cr    # JSONConverter helper for tool param schemas
    message.cr           # Agent::Message, ContentPart, ToolCall, Usage
    response.cr          # Agent::Response — async handle with stream/join/finished
    tool.cr              # Agent::Tool / FunctionDef (extracted from agent.cr)
    agent.cr             # Agent — fiber loop, history, tool resolution (no HTTP/SSE code)
    provider.cr          # requires provider/ modules
    provider/
      base.cr            # Agent::Provider::Base — abstract interface
      openai/
        openai.cr        # entry point, requires sub-files
        config.cr        # Provider::OpenAI < Base + OpenAI::Config
        request_body.cr  # OpenAI wire-format request builder
        stream_parser.cr # SSE parser, delta accumulation, usage parsing
```

### Provider separation

All wire-format concerns live in `Provider::Base` implementations. The default `Provider::OpenAI` handles the OpenAI /v1/chat/completions format. The `Agent` core talks only to the abstract interface.

`Message#provider_data` and `ToolCall#provider_data` hold what a provider
needs to replay a message exactly (signed or encrypted reasoning, Gemini
thought signatures). Each provider writes under its own key and reads only
that key; the OpenAI provider also records the endpoint and model and
replays only to them. It replays reasoning in the field it arrived in, but
only for fields providers require back (`reasoning_details`,
`reasoning_content`, Mistral's thinking chunks); `reasoning` and `thinking`
text is display-only, since OpenRouter's repeats its blocks.

## Architecture diagram

```mermaid
sequenceDiagram
    participant Caller
    participant Agent
    participant Fiber
    participant Provider as Provider::OpenAI
    participant HTTP as HTTP::Client
    participant API as OpenAI API

    Caller->>Agent: ask("hello")
    Agent->>Fiber: send(Request) on @request_channel
    Agent-->>Caller: return Response (immediately)

    Fiber->>Fiber: receive(Request) from channel
    Fiber->>Provider: build_request(messages, tools)
    Provider-->>Fiber: {path, headers, body}
    Fiber->>HTTP: POST (path, headers, body)
    HTTP->>API: HTTPS request
    API-->>HTTP: SSE stream
    loop each SSE chunk
        HTTP-->>Fiber: yield line
        Fiber->>Provider: parse_stream(io, response)
        Provider->>Response: push_chunk(delta)
        alt caller called #stream
            Response->>Caller: chunk via channel
        else caller blocks on #message
            Response: buffer in @chunk_channel (size 256)
        end
    end
    Provider-->>Fiber: {Message, Usage, finish_reason}
    Fiber->>Response: finish(message, usage)
    Response->>Caller: #message / #metadata unblock
```

## Key patterns

### Fiber-per-agent

Each `Agent.new(config)` spawns a persistent fiber:

```crystal
@fiber = spawn { run_loop }
```

`run_loop` blocks on `@request_channel.receive`. When a `Request` arrives it
calls `http_post_stream` which delegates to the provider for building the
request and parsing the response. The assistant reply is appended to
`@history`. After the HTTP call completes, the fiber loops back and waits
for the next request. This serialises requests through one fiber — no
locking needed.

### Request / Response decoupling

`#ask` does two things synchronously:
1. Builds a `Message`
2. Creates a fresh `Response` object and sends an `AskRequest` (new_messages + tools + response) on `@request_channel`

The fiber receives the request in `process_request_loop` and appends the
messages to `@history` inside the fiber, ensuring single-owner mutation.

It returns the `Response` immediately. The calling fiber is free to read
chunks, wait for the message, or do other work.

### Provider separation

All wire-format concerns live in `Agent::Provider::Base` implementations.
The `http_post_stream` shim in `agent.cr` is provider-agnostic:

```crystal
req = @provider.build_request(messages, tools)
client.post(req[:path], headers: req[:headers], body: req[:body]) do |http_resp|
  msg, usage, finish_reason = @provider.parse_stream(
    http_resp.body_io, response, ->{ response.cancelled? }
  )
  raise CancelledError.new if response.cancelled?
  {msg, usage, finish_reason}
end
```

Auth headers, request body shape, SSE parsing, and final message assembly are
all owned by the provider. To support a non-OpenAI API, implement `Base` and
pass it to `Agent.new(config, provider: MyProvider.new(...))`.

### Buffered chunk channel

`Response` has three channels:
- `chunk_channel` — size **256** (buffered). The HTTP fiber can push up to
  256 tokens without blocking, even if nobody calls `.stream`. This means
  `agent.ask("...").message` works without deadlocking.
- `message_channel` — size **1** (buffered). Signals the final `Message`.
- `usage_channel` — size **1** (buffered). Signals `Usage`.

The `finish` method sends to both message and usage channels, then closes
the chunk channel:

```crystal
def finish(message, usage)
  @message_channel.send(message)
  @usage_channel.send(usage)
  @done = true
ensure
  @chunk_channel.close
end
```

Closing `chunk_channel` after the block ensures the `ensure` runs even if
`send` raises — and the close signals `#stream` to exit its receive loop.

### Non-blocking push for streams

`push_chunk` uses `send` (not `try_send` / `spawn`) because the buffer is
large enough for typical streaming. If the buffer somehow fills up, the HTTP
fiber will briefly block — this is acceptable backpressure.

## Quirks and gotchas

### `JSON::Any` boilerplate

Crystal's `JSON::Any` doesn't auto-coerce Hash literals. Every value in a
`Hash(String, JSON::Any)` literal needs explicit wrapping:

```crystal
# Right:
{"key" => JSON::Any.new("value")}

# Wrong — compile error:
{"key" => "value"}
```

`build_request_body` and `Message#to_request_body` handle this with `.map { |h| JSON::Any.new(h) }` on intermediate arrays.

### ToolCallDelta struct

Tool-call deltas across SSE chunks are accumulated in a `ToolCallDelta`
private struct (`@id`, `@name`, `@arguments`), keyed by the delta's `index`
field. This replaces an earlier positional-`Array(String)` approach and is
much clearer.

### Fiber scheduling

Crystal's cooperative scheduling means a fiber blocked on `Channel.receive`
won't run until the current fiber yields. The `Channel.send` operation
yields to a waiting receiver, so `@request_channel.send(...)` in `#ask`
transfers control to the agent fiber. No explicit `Fiber.yield` is needed.

### Channel close ordering

`finish` and `finish_with_error` both send to `message_channel` and `usage_channel`
*before* closing `chunk_channel`. Additionally:
- `finish` captures the `finish_reason` from the API ("stop", "length",
  "tool_calls", etc.) exposed as `Response#finish_reason`.
- `finish_with_error` stores an `Agent::Error` and
  sends a synthetic error message so `#message` always unblocks.

This ordering matters:
- The consumer (blocked on `#message` or `#metadata`) can unblock before
  the chunk channel is closed.
- The `ensure` block guarantees the chunk channel is always closed, even
  if a send raises `Channel::ClosedError`.

### Mock server in tests

The integration spec (`spec/agent_spec.cr`) starts a local `HTTP::Server`
on a random port (`bind_tcp(0)`) and passes the port to the test block. The
server is spawned into a background fiber. Each test gets its own port so
tests can run in parallel.

The mock server must call `ctx.response.close` after writing SSE data,
otherwise the HTTP client will block waiting for more data (the `each_line`
reader waits until EOF/connection close).

### Error handling

All errors produced by the agent use a consistent prefix `"Agent error: ..."`
so callers can pattern-match programmatically.

If `http_post_stream` raises (network error, non-200 status, JSON parse
failure), the error is caught in `http_post_stream`'s own rescue block, which
calls `response.finish_with_error` with an `Agent::Error`. This ensures
`#message` and `#join` always unblock. A `CancelledError` is raised when
the caller calls `#cancel` on the response — the HTTP fiber checks the
cancel flag before each SSE line and aborts cleanly, calling
`finish_with_error(CancelledError.new)`.

## Development workflow

### Prerequisites

- Crystal >= 1.19 (the library requires the stdlib `sync` module, added in 1.19.0)
- `OPENAI_API_KEY` environment variable set when running integration tests against a real API.

### Example configuration pattern

All examples follow this precedence for configuration values:

1. **CLI arguments** (`--endpoint`, `--model`, `--api-key`) — checked first
2. **Environment variables** (`LLM_ENDPOINT`, `LLM_MODEL`, `LLM_API_KEY`) — fallback
3. **Raise error** if neither is provided

There are **no hardcoded defaults** in the examples.
Only `LLM_ENDPOINT` is required; `LLM_MODEL` and `LLM_API_KEY` are optional and can be `nil`.
The minimal pattern looks like:

```crystal
endpoint = ENV.fetch("LLM_ENDPOINT") { raise "Missing LLM_ENDPOINT" }
model = ENV["LLM_MODEL"]?
api_key = ENV["LLM_API_KEY"]?
```

### Setup

```sh
shards install
```

### Compile

Check that the shard compiles cleanly:

```sh
crystal build src/agent.cr
```

Or, to type-check without producing a binary:

```sh
crystal tool hierarchy src/agent.cr 2>&1 | head -5
# or just:
crystal build --no-codegen src/agent.cr
```

### Run tests

All specs are written using Crystal's built-in `Spec` framework:

```sh
crystal spec
```

Run a single spec file:

```sh
crystal spec spec/agent_spec.cr
```

Filter tests by name:

```sh
crystal spec --tag ~remote   # skip tests that require a real API
crystal spec -e "streaming" # only specs whose description matches "streaming"
```

> **Note:** The mock server in `spec/agent_spec.cr` uses a random port so tests can run in parallel. Each test starts its own `HTTP::Server` and calls `ctx.response.close` after writing SSE data to avoid hanging the HTTP client.

### Format

All Crystal source must be formatted with the standard formatter before committing:

```sh
crystal tool format
```

### Lint

Run the Ameba style linter (installed via `shards install` → `bin/ameba`):

```sh
bin/ameba
```

Lint and format can be checked together:

```sh
bin/ameba && crystal tool format --check
```

### Adding a new feature

1. Add the implementation in `src/agent/` under the appropriate file.
2. Add corresponding unit tests in `spec/`.
3. If the feature changes the public API (new method/additional parameter), update `README.md` with an example.
4. Run `crystal spec` to confirm nothing is broken.
5. Run `crystal tool format` to ensure consistent formatting.

### Releasing

1. Bump the version in `shard.yml` and `src/agent/version.cr`.
2. Update `CHANGELOG.md` if one exists (otherwise, consider adding one).
3. Tag the release with `git tag v<version>` and push.

## Registered tools & auto-resolve loop

Tools can be registered with a callback via `register_tool(name, description, parameters, &block)`.
The callback receives the parsed JSON arguments hash and returns a string result.

### How it works

When `Config#auto_execute_tools` is `true` (the default):

1. All registered tools are automatically merged into every `#ask` request.
2. The `run_loop` fiber calls `process_request_loop` instead of the old single-pass.
3. After each HTTP response, if the model returned tool calls and all are
   registered, the agent executes the callbacks, appends tool-result messages
   to history, and sends a new request to the model — all within the fiber.
4. The loop continues until the model returns a message without tool calls.
5. `Response#finish` is called only on the final (resolved) message.

```mermaid
flowchart TD
    A[#ask] --> B[Send Request to fiber]
    B --> C[http_post_stream]
    C --> D{Tool calls?}
    D -->|No| E[response.finish]
    D -->|Yes| F{auto_execute?}
    F -->|No| E
    F -->|Yes| G{All registered?}
    G -->|No| E
    G -->|Yes| H[Execute callbacks]
    H --> I[Append results to history]
    I --> C
```

### Error handling in the loop

- A call to an unregistered or disabled tool gets an error result and the
  loop continues. If a call names a per-request tool (`ask(tools:)`) that is
  not an enabled registered one, nothing runs and the whole message goes
  back to the caller (manual dispatch).
- At the start of each turn, any tool call left without a result (the
  caller skipped the follow-up, or a turn failed midway) gets an error
  result, so history is always a valid request.
- If `http_post_stream` fails, the response finishes with the error and the
  turn's usage so far, and the loop exits. A transient failure (connection
  error, 408/409/429/5xx, a stream error event with such a code, a cut-off
  stream) is first retried up to `max_retries` times with backoff and
  `Retry-After`, but only while that model call has pushed no chunk; after
  that, `Response#partial?` tells the caller the shown text is incomplete.
  Anything that fails after the 200 is a `StreamError`. A failed first call
  rolls the turn back; a later failure keeps it, since tools have run.
- Any other exception escaping the loop (a handler, the provider) finishes
  the response with it instead of killing the fiber. A `Handler` can
  therefore raise an `Agent::Error` to stop a turn.
- Every failure rebuilds the HTTP client, since an unread body on the
  kept-alive connection would corrupt the next request.
- A consumer that leaves `#stream` early stops the stream instead of
  blocking the fiber.
- `Response#cancel` is checked between SSE lines, so a request still
  waiting on the provider (headers, first byte) is cancelled only when a
  byte arrives or the read times out. Closing the socket from the caller's
  fiber would wake it, but on TLS that re-enters OpenSSL mid-read.
- Only an `Agent::ToolError`'s message reaches the model; other exceptions
  from tool callbacks are logged and reported as "the tool failed". Either
  way the exception is on `ToolCallContext#error` for handlers, which can
  replace the result the model sees.

### Tool argument validation

Tool call arguments are validated against the registered parameter schema
(required fields + type checking). Validation failures are **soft** — they
produce a tool-result `Message` with an error string (e.g.
`"Error validating arguments for tool 'get_weather': Missing required field 'city'"`)
and the callback is skipped. The model sees the error in its next turn
and can self-correct. No exception is raised.

## To answer your questions directly

### Tool call ordering

**Tool calls always arrive as a single assistant message** with
`finish_reason: "tool_calls"`. The model does NOT interleave content and
 tool calls within a single turn. All tool calls in a response are emitted
 together at the end of the stream. The SSE deltas may arrive in arbitrary
 order (id in one chunk, name in another, arguments in pieces), but the
 final assembled message has all tool calls at once.

### Callback registration vs manual dispatch

You are right that the old pattern — building `Agent::Tool` schema objects
manually, checking `message.has_tool_calls?`, parsing arguments,
constructing `Message` objects with `role: "tool"`, and while-looping —
is cumbersome. The new `register_tool` API eliminates all of that:

```crystal
# OLD: ~25 lines of boilerplate (see examples/cli.cr before the change)
GET_TIME_TOOL = Agent::Tool.new(...)
resp = agent.ask(input, tools: [GET_TIME_TOOL])
resp.stream { |chunk| print chunk }
msg = resp.message
while msg.has_tool_calls?
  results = execute_tool_calls(msg.tool_calls.not_nil!)
  resp = agent.ask(results, tools: [GET_TIME_TOOL])
  resp.stream { |chunk| print chunk }
  msg = resp.message
end

# NEW: 5 lines, no while-loop
agent.register_tool("get_time", "Get the current time",
  parameters: Agent::JSONConverter.from({
    type:       "object",
    properties: {} of String => String,
    required:   [] of String,
  })
) { |args| ... }
resp = agent.ask(input)
resp.stream { |chunk| print chunk }
```

## Future considerations

- **Non-streaming fallback** — Some providers don't support streaming. A
  non-streaming `POST` path could detect `stream` support in the response
  headers and fall back.

- **Request timeout** — Timeouts (`read_timeout`, `connect_timeout`) are
  configurable via `Config` but default to `nil` (no timeout). Setting
  them is recommended for production use.
# Changelog

## [Unreleased]

### Fixed

- History trimming drops whole turns. A turn is a user message and
  everything up to the next one, so `max_history` now counts conversation
  turns as documented, the kept history always starts at a user message,
  and a tool result is never left without the assistant message that called
  it (previously a tool-call turn could be cut in the middle, and the next
  request rejected by the API).
- Streamed completions report token counts: the OpenAI provider sends
  `stream_options: {include_usage: true}`, without which OpenAI, Ollama,
  vLLM and others return no `usage` on a stream and `Response#metadata` was
  all nil. `Agent::Config.new(include_usage: false)` turns it off for a
  server that rejects the field.
- An API error with a body (429, 400, 500) no longer breaks the next
  request on the agent. The unread body stayed on the kept-alive connection
  and was read as the next response (`Unsupported HTTP version: {...`). The
  HTTP client is now rebuilt after every failure.
- A consumer that leaves `Response#stream` early (`break`, or an exception
  such as a disconnected client) no longer blocks the agent fiber forever
  once 256 chunks are queued.
- A call to an unregistered or disabled tool gets an error result and the
  loop continues. Before, the whole message went back to the caller after
  the registered tools in it had already run, discarding their results and
  leaving history that the API rejects. A message that calls a per-request
  tool still goes back to the caller, now before any tool runs.
- Tool calls left without a result (no `ask(tool_results)` follow-up, or a
  turn that failed midway) are answered with an error result at the start
  of the next turn instead of making every later request fail.
- A provider call failing after the first call of a tool loop no longer
  removes the last tool result from history; the turn is kept. The
  response's `metadata` reports the usage of the calls that succeeded, on
  every error and on cancel, instead of nothing.
- A cancelled turn drops the tool calls it streamed instead of keeping them
  in history without results.
- An exception from a `Handler` or the provider finishes the response with
  that error. Before, it killed the agent fiber, and every later `#ask` hung.
- A `null` token count or tool-call `index` in the stream no longer fails
  the request.
- `Agent#close` no longer closes the HTTP client under a request in flight.
  On TLS that shut down the session while the agent fiber was mid-read on
  it. The call in flight now finishes, the turn ends with a closed error
  instead of continuing its tool loop, and the agent fiber releases the
  client and provider as it exits.
- A stream that fails after it started no longer reads as a success. An
  error event inside the stream (OpenRouter sends these), a stream that
  ends with neither a finish reason nor `[DONE]`, or a 200 with a JSON
  body instead of a stream now finishes the response with
  `Agent::StreamError`. Before, the partial reply passed for a complete one.
- Every failure after the provider accepted the request (HTTP 200) is now
  an `Agent::StreamError`, whatever broke. A connection dropped mid-stream
  used to be a `ConnectionError`, and a malformed stream was misreported as
  one too.
- `IdleTimeoutError` and `ToolLoopError` messages no longer start with
  "Agent error: Agent error:".
- An `Agent::Error` raised by a chunk handler reaches the response as is,
  instead of being wrapped in a `ConnectionError`.
- A cancel now stops a tool loop before its next tool call, not only before
  the next model call. Each skipped call gets an error result.
- Chunks pushed before `Response#stream` was called are no longer dropped.
  They queue until `#stream` starts, without blocking the agent fiber.
  Chunk handlers now see every chunk, whether or not anyone streams.
- A tool callback's exception message no longer reaches the model. It is
  logged, and the model sees `Error executing tool 'x': the tool failed`.
  Raise `Agent::ToolError` for a message the model should see.
- The agent no longer runs the tool calls of a reply that ended early
  (`length`, `max_tokens`, `refusal`, `content_filter`,
  `model_context_window_exceeded`): their arguments may be half-written.
  The turn fails with a `StreamError` whose code is the finish reason.

### Added

- Prompt caching settings shared by providers: `Config#prompt_caching`
  (`:explicit`, the default; `:automatic`, where the API marks the
  conversation itself; `:off`) and `Config#cache_ttl` (`:five_minutes` or
  `:one_hour`). The Anthropic provider marks the last tool when there is no
  system prompt, and adds markers inside a turn longer than the cache's
  20-position lookback. The OpenAI provider now marks the system prompt and
  latest turn for Claude models on OpenRouter, which cache nothing
  otherwise, and reads OpenRouter's `cache_write_tokens` into
  `Usage#cache_creation_tokens`.

- `Agent::Provider::Anthropic`, for Anthropic's Messages API, moved in
  from Aureax. It streams text, thinking (summarized into
  `Message#reasoning`), tool calls, and server-tool blocks, and keeps every
  reply's blocks in `provider_data` to send back byte for byte. It maps
  `Config#reasoning_effort` to `output_config.effort`, caches the system
  prompt and the transcript, and has the API drop thinking blocks that
  history trimming invalidated rather than refuse the request.
- Reasoning support:
  - `Config#reasoning_effort` takes an `Agent::ReasoningEffort` (`:off`
    through `:max`), the same for every provider. The OpenAI provider sends
    it as `reasoning_effort`, or as `reasoning: {effort:}` on openrouter.ai.
    `#ask(reasoning_effort:)` overrides it for one request, and a handler
    can change it per provider call through `TurnContext#options`.
  - `Config#extra_body` adds request fields Config doesn't name, such as a
    reasoning budget. It never replaces a field the provider sets.
  - Reasoning is kept in `provider_data` and sent back on later requests
    in the shape the endpoint returned it, so it survives a tool loop and
    `dump`/`load`: `reasoning_details` blocks (OpenRouter), signed and
    encrypted ones included, unchanged; a tool call's `extra_content`
    (Gemini's thought signature); `reasoning_content` text (DeepSeek, which
    refuses a tool request without it; Kimi; vLLM); Mistral's thinking
    chunks. It is keyed by provider, endpoint and model, and only goes back
    where it came from. `Config#replay_reasoning` (default on) turns it off.
  - Mistral's `content` arrays of thinking and text chunks are read. Before,
    they were dropped, answer text included.
  - A cancelled reply keeps its text but drops its replay data, since an
    unsigned block would be refused.
  - `Usage#reasoning_tokens` comes from
    `completion_tokens_details.reasoning_tokens`.
- `ToolCall#provider_data`, the same slot as on `Message`.
- `Agent::RequestOptions` and `Provider::Base#build_request(messages, tools,
  options)`. The default implementation calls the two-argument form, so
  existing providers keep working.
- `#dump` writes version 2; `#load` refuses a dump newer than it supports.
- Long streams no longer cost quadratic time: reasoning blocks and tool
  call arguments are appended in place instead of re-concatenated per
  fragment.
- Chunks waiting for `Response#stream` are joined per kind, so a long reply
  costs its text rather than an object per token.
- Retries: a model call that fails transiently before it streamed any
  output is retried up to `Config#max_retries` times (default 2) with
  jittered exponential backoff (`retry_base_delay`, default 1 s), honouring
  `Retry-After` (seconds or a date) and `retry-after-ms` up to
  `retry_max_delay` (default 30 s). Transient means a connection error or
  connect timeout, a TLS session cut off, a 408/409/429/5xx (or what the
  provider's `x-should-retry` header says), or a stream error with such a
  code, cut off, or ended by an error before its first chunk. Once a call
  has streamed output, nothing is retried, as in the OpenAI and Anthropic
  SDKs.
- `ApiError#body`, `#headers`, `#request_id`, `#retry_after` and
  `#retryable?`. Its message now includes the provider's error message
  (`"429 Too Many Requests: Rate limit reached"`).
- `Agent::StreamError` (with `#code`, `#retryable?` and the original
  exception as `#cause`), for a request accepted with a 200 whose stream
  then failed.
- `Response#partial?`: the response failed after output had streamed, so
  what the caller showed is an incomplete reply. `#output_started?` says
  whether any chunk was produced.
- `Agent::ToolError`: raise it from a tool callback to show the model why
  the tool failed.
- `ToolCallContext#error`: the exception a tool callback raised, set when
  `next_proc` returns. A handler can record it, or match on a `ToolError`
  subclass carrying its own fields and return a different result for the
  model.

- `Usage#cached_tokens` and `Usage#cache_creation_tokens`: the prompt
  tokens served from the provider's prompt cache and written to it. The
  OpenAI parser fills `cached_tokens` from `prompt_tokens_details`; both
  sum with `+` like the other fields.
- `Message#is_error` on tool results (`Message.tool_result(tc, text,
  is_error: true)`, `Message#error?`). The agent loop sets it on the
  results it builds for a tool that raised, had no callback, or whose
  arguments failed to parse or validate. The OpenAI wire format has no such
  flag and ignores it; a custom provider whose format does (Anthropic's
  `tool_result` block) can send it.
- `Message#provider_data : JSON::Any?`: an opaque slot for what a custom
  provider needs to replay a message exactly (signed reasoning blocks,
  citations). Kept through `dump`/`load` and compared by `==`. The OpenAI
  provider keeps `reasoning_details` there (see above).

## [0.2.0] - 2026-10-19

### Fixed

- Tool-argument validation no longer skips the type check for JSON `null`.
  A `null` argument against a property that declares a `type` (e.g.
  `{"type": "array"}`) is now rejected with
  `Field '<name>' expected type '<type>', got 'null'`, giving the model a
  structured signal to self-correct. `null` still passes for properties that
  declare no `type` and for arguments absent from `properties`.
- `validate_tool_args` no longer raises (and kills the agent fiber) on schemas
  that declare `type` as an array of alternatives, e.g. `"type": ["string",
  "null"]`.

### Added

- Tool-argument validation honours `items` on array properties: elements of
  the wrong type are rejected with `Field '<name>' expected array items of
  type '<type>', got '<actual>' at index <i>`.

## [0.1.1] - 2026-09-30

First stable release. This patch release hardens the tool auto-resolve loop
and the background agent fiber, and replaces string-prefix error detection
with structured exception types.

### Fixed

- **Critical: Duplicate tool results in auto-resolve loop** (#1). Tool-result
  messages were being appended to history twice, causing malformed API requests.
- **Critical: Raising tool callbacks kill the agent fiber** (#2). A tool callback
  that raised an exception would silently terminate the background fiber,
  blocking all future `#ask` calls. Callback errors are now captured and returned
  as tool-result messages.
- **Critical: `#reset` races with in-flight requests** (#3). `#reset` now routes
  through the request channel, serializing with any pending request and waiting
  for it to complete before clearing history.
- String-prefix error detection replaced with structured error types (#4/#5).
  Added `Agent::Error`, `Agent::ConnectionError`, `Agent::ApiError`,
  `Agent::ToolError` exception classes and `Response#error?` / `Response#error`.
- SSE parser now handles `data:` (without trailing space) per the SSE spec (#24).
- `ToolCallDelta.id` and `name` use assignment (`=`) instead of `+=` (#17).
- `max_history` trimming now correctly counts user/assistant roles, avoiding
  orphaned tool messages in agentic flows (#7).

### Changed

- Removed misleading `tags: "remote"` from all spec tests (#35). Tests use a
  local mock server and don't require a remote API.

### Added

- Tests for the auto-resolve tool loop, raising callbacks, `register_tool`
  without auto_execute, `max_history` trimming, and structured error responses
  (#32, #33, #34).
- `CHANGELOG.md` (#39).
- Updated `shard.yml` author (#38).

## [0.1.0] - Initial release
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

### Added

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
  citations). Kept through `dump`/`load`, compared by `==`, never sent by
  the OpenAI provider.

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
require "../../spec_helper"

# A mock IO that honours `read_timeout=` (per-read semantics) and lets tests
# program per-line delays before the line is returned. Used by the timeout specs
# below to verify the parser applies the first-byte and idle budgets correctly.
# Returns lines in order; once exhausted, `#gets` returns nil (EOF). Every
# successful read resets the per-read timer because each `#gets` independently
# consults `read_timeout`.
private class MockTimeoutIO < IO
  property read_timeout : Time::Span?

  def initialize(@lines : Array(String), @delays : Array(Time::Span) = [] of Time::Span)
    @i = 0
    @read_timeout = nil
  end

  def read(slice : Bytes) : Int32
    raise "use gets"
  end

  def gets(chomp : Bool = true) : String?
    return if @i >= @lines.size
    delay = @i < @delays.size ? @delays[@i] : Time::Span.zero

    if t = @read_timeout
      # Honor the per-read timeout by racing the delay against it. The channel
      # signals when the programmed delay elapses; the select branch raises
      # IO::TimeoutError if the read_timeout fires first.
      ready = Channel(Nil).new
      spawn do
        sleep delay if delay > Time::Span.zero
        ready.send(nil)
      end
      select
      when ready.receive
        # delay elapsed — proceed to return the line
      when timeout(t)
        raise IO::TimeoutError.new("MockTimeoutIO: read timed out after #{t.total_milliseconds.round}ms")
      end
    elsif delay > Time::Span.zero
      sleep delay
    end

    line = @lines[@i]
    @i += 1
    chomp ? line : "#{line}\n"
  end

  def write(slice : Bytes) : Nil
    raise "not implemented"
  end
end

# Helper: feed `lines` through the parser; collect reasoning chunks and the
# final Message#reasoning.
#
# Mirrors the real caller: `#stream` must be active while the parser pushes
# chunks (otherwise `push_chunk` drops them), and `#finish` closes the chunk
# channel so the stream consumer returns.
private def parse_lines(lines : Array(String))
  response = Agent::Response.new
  reasoning_chunks = [] of String
  collected = Channel(String).new

  spawn do
    response.stream { |c| collected.send(c.text) if c.reasoning? }
    collected.close
  end
  # Let the stream fiber run so `@streaming` is set before the parser pushes
  # chunks (otherwise `push_chunk` drops them).
  Fiber.yield

  io = IO::Memory.new(lines.join('\n'))
  msg, usage, _finish = Agent::Provider::OpenAI::StreamParser.parse(io, response, -> { false })
  response.finish(msg, usage)

  # Channel#each only exists on Crystal >= 1.21; drain with receive? instead so
  # the spec suite compiles on every supported version.
  while text = collected.receive?
    reasoning_chunks << text
  end
  {reasoning_chunks, msg.reasoning}
end

describe Agent::Provider::OpenAI::StreamParser do
  it "reads delta.reasoning_content (DeepSeek native)" do
    chunks, reasoning = parse_lines([
      %(data: {"choices":[{"index":0,"delta":{"reasoning_content":"Hello "}}]}),
      %(data: {"choices":[{"index":0,"delta":{"reasoning_content":"world"}}]}),
      %(data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
      "data: [DONE]",
    ])
    chunks.should eq(["Hello ", "world"])
    reasoning.should eq("Hello world")
  end

  it "reads delta.reasoning (OpenRouter)" do
    chunks, reasoning = parse_lines([
      %(data: {"choices":[{"index":0,"delta":{"content":"","role":"assistant","reasoning":"We","reasoning_details":[{"type":"reasoning.text","text":"We","format":"unknown","index":0}]},"finish_reason":null}]}),
      %(data: {"choices":[{"index":0,"delta":{"content":"","role":"assistant","reasoning":" need","reasoning_details":[{"type":"reasoning.text","text":" need","format":"unknown","index":0}]},"finish_reason":null}]}),
      %(data: {"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":"stop"}]}),
      "data: [DONE]",
    ])
    chunks.should eq(["We", " need"])
    reasoning.should eq("We need")
  end

  it "reads delta.reasoning_details without a string field" do
    chunks, reasoning = parse_lines([
      %(data: {"choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.text","text":"Step "},{"type":"reasoning.text","text":"one"}]}}]}),
      "data: [DONE]",
    ])
    chunks.should eq(["Step one"])
    reasoning.should eq("Step one")
  end

  it "does not double-count when both reasoning and reasoning_details are present" do
    chunks, reasoning = parse_lines([
      %(data: {"choices":[{"index":0,"delta":{"reasoning":"We","reasoning_details":[{"type":"reasoning.text","text":"We","format":"unknown","index":0}]}}]}),
      "data: [DONE]",
    ])
    chunks.should eq(["We"])
    reasoning.should eq("We")
  end
end

describe Agent::Provider::OpenAI::StreamParser do
  it "treats explicit reasoning:null as no chunk" do
    chunks, _reasoning = parse_lines([
      %(data: {"choices":[{"index":0,"delta":{"role":"assistant","reasoning":null},"finish_reason":"stop"}]}),
      "data: [DONE]",
    ])
    chunks.should be_empty
  end
end

# Wraps StreamParser.parse so the stream consumer fiber is set up before the
# parser pushes chunks (mirrors parse_lines above). Returns the parser's tuple
# on success, or raises whatever IO::TimeoutError was thrown.
private def parse_with_timeouts(
  io : IO,
  first_byte_timeout : Time::Span? = nil,
  idle_timeout : Time::Span? = nil,
  cancel : -> Bool = -> { false },
)
  response = Agent::Response.new
  spawn { response.stream { |_| } } # consume chunks so push_chunk doesn't block
  Fiber.yield
  msg, usage, finish = Agent::Provider::OpenAI::StreamParser.parse(
    io, response, cancel,
    first_byte_timeout: first_byte_timeout,
    idle_timeout: idle_timeout,
  )
  response.finish(msg, usage) unless response.finished?
  {response, msg, usage, finish}
end

# Per-read SSE timeout behaviour (AGENTS.md "dynamic read-timeout to fail-fast
# on hung SSE connections"). These exercise `StreamParser.parse`'s use of the
# `first_byte_timeout` / `idle_timeout` keyword args and confirm it:
#
#   1. doesn't time out under normal streaming
#   2. raises IO::TimeoutError when the first read exceeds `first_byte_timeout`
#   3. raises IO::TimeoutError when a mid-stream gap exceeds `idle_timeout`
#   4. treats comment / heartbeat lines as "any byte arrival" that resets the
#      idle timer
#   5. honours `first_byte_timeout_ms_per_token` from `Agent::Config`
#   6. falls back to `idle_timeout` for *all* reads (incl. the first) when
#      `compute_first_byte_timeout?` is false
#   7. lets caller cancellation short-circuit the parse loop and the response
#      surface as `cancelled?` (not `error?`)
describe Agent::Provider::OpenAI::StreamParser do
  it "does not time out under normal streaming" do
    # Each line arrives within 1ms; first-byte budget of 10ms and idle budget
    # of 10ms should both comfortably elapse without firing.
    io = MockTimeoutIO.new([
      %(data: {"choices":[{"index":0,"delta":{"content":"a"}}]}),
      %(data: {"choices":[{"index":0,"delta":{"content":"b"}}]}),
      %(data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
      "data: [DONE]",
    ], delays: [1.milliseconds, 1.milliseconds, 1.milliseconds, 1.milliseconds])

    response, msg, _usage, _finish = parse_with_timeouts(io,
      first_byte_timeout: 10.milliseconds,
      idle_timeout: 10.milliseconds)
    msg.content.should eq("ab")
    response.error?.should be_false
  end

  it "raises IdleTimeoutError(:first_byte) on first-byte timeout" do
    # First read delayed past the 10ms first-byte budget. The parser wraps the
    # IO::TimeoutError raised by the underlying IO as an IdleTimeoutError with
    # phase :first_byte (no successful read has occurred yet).
    io = MockTimeoutIO.new([
      %(data: {"choices":[{"index":0,"delta":{"content":"a"}}]}),
    ], delays: [0.5.seconds])

    err = expect_raises(Agent::IdleTimeoutError) do
      parse_with_timeouts(io,
        first_byte_timeout: 10.milliseconds,
        idle_timeout: 10.milliseconds)
    end
    err.phase.should eq(:first_byte)
    err.cause.should be_a(IO::TimeoutError)
  end

  it "raises IdleTimeoutError(:idle) on inter-byte timeout" do
    # First line arrives fast; second line is delayed past the idle budget.
    # phase :idle because at least one line has been read successfully.
    io = MockTimeoutIO.new([
      %(data: {"choices":[{"index":0,"delta":{"content":"a"}}]}),
      %(data: {"choices":[{"index":0,"delta":{"content":"b"}}]}),
    ], delays: [1.milliseconds, 0.5.seconds])

    err = expect_raises(Agent::IdleTimeoutError) do
      parse_with_timeouts(io,
        first_byte_timeout: 10.milliseconds,
        idle_timeout: 10.milliseconds)
    end
    err.phase.should eq(:idle)
    err.cause.should be_a(IO::TimeoutError)
  end

  it "treats comment / heartbeat lines as idle-timer resets" do
    # Three lines each delayed by 25ms; idle budget is 30ms so each individual
    # gap fits within the budget. The `: OPENROUTER PROCESSING` comment line
    # arrives between two `data: {}` chunks — any byte arrival resets the idle
    # timer, so the gaps straddle two adjacent reads rather than accumulating.
    # Assert no raise (without the reset, the cumulative 75ms would exceed 30ms
    # — but per-read semantics mean only the gap between two consecutive
    # reads counts; each is 25ms so all is well).
    io = MockTimeoutIO.new([
      %(data: {"choices":[{"index":0,"delta":{"content":"a"}}]}),
      ": OPENROUTER PROCESSING",
      %(data: {"choices":[{"index":0,"delta":{"content":"b"}}]}),
    ], delays: [25.milliseconds, 25.milliseconds, 25.milliseconds])

    response, msg, _usage, _finish = parse_with_timeouts(io,
      first_byte_timeout: 1.second,
      idle_timeout: 30.milliseconds)
    msg.content.should eq("ab")
    response.error?.should be_false
  end

  it "honours Agent::Config first_byte_timeout_ms_per_token knob" do
    # With ms_per_token = 1000 (1s/token), a 1k-char prompt (~250 tokens)
    # yields ~250s added to the 3s base = 253s, below the 300s default max.
    config = Agent::Config.new(
      api_key: "test",
      api_endpoint: "http://localhost:1",
      first_byte_timeout_ms_per_token: 1000,
    )
    # 1000 chars / 4 = 250 tokens * 1000ms = 250000ms + 3000ms = 253s.
    config.compute_first_byte_timeout(1000).should eq(253.seconds)

    # At 100k chars, 25000 tokens * 1s + 3s = 25003s — clamp to 300s max.
    config.compute_first_byte_timeout(100_000).should eq(300.seconds)

    # Smaller base + tighter knobs: 1000 chars / 4 = 250 tokens * 10ms = 2500ms
    # = 2.5s; clamp to [1s, 5s] = 2.5s.
    tight = Agent::Config.new(
      api_key: "test",
      api_endpoint: "http://localhost:1",
      first_byte_timeout_ms_per_token: 10,
      first_byte_timeout_min: 1.seconds,
      first_byte_timeout_max: 5.seconds,
      first_byte_timeout_base: 0.seconds,
    )
    tight.compute_first_byte_timeout(1000).should eq(2.5.seconds)
  end

  it "uses idle_timeout for all reads when compute_first_byte_timeout? is false" do
    config = Agent::Config.new(
      api_key: "test",
      api_endpoint: "http://localhost:1",
      compute_first_byte_timeout: false,
      idle_byte_timeout: 30.milliseconds,
    )
    config.compute_first_byte_timeout(1_000_000).should be_nil
    config.idle_byte_timeout.should eq(30.milliseconds)

    # When the parser is called with first_byte_timeout: nil, both the first
    # and subsequent reads should fall back to idle_timeout. The first read is
    # fast (1ms — well within 30ms) but would time out if first-byte budget
    # were 0ms (which compute_first_byte_timeout would return at 0 prompt bytes).
    # Inverse test: a first-read delay of 0.5s with idle_timeout=10ms must
    # raise — i.e. idle_timeout applies to the first read too.
    io = MockTimeoutIO.new([
      %(data: {"choices":[{"index":0,"delta":{"content":"a"}}]}),
    ], delays: [0.5.seconds])

    err = expect_raises(Agent::IdleTimeoutError) do
      parse_with_timeouts(io, first_byte_timeout: nil, idle_timeout: 10.milliseconds)
    end
    err.phase.should eq(:first_byte)
    err.timeout.should eq(10.milliseconds)
  end

  it "lets caller cancellation short-circuit the parse loop" do
    # The mock IO's first read is delayed long enough that the parser is parked
    # when the caller flips the cancel flag; on the next loop iteration
    # `break if cancel.call` exits cleanly. We use a 0-delay then a long-delay:
    # the parser blocks on the second read; the cancel proc returns true, the
    # parser breaks before raising IO::TimeoutError.
    io = MockTimeoutIO.new([
      %(data: {"choices":[{"index":0,"delta":{"content":"a"}}]}),
      %(data: {"choices":[{"index":0,"delta":{"content":"b"}}]}),
    ], delays: [1.milliseconds, 1.second])

    response = Agent::Response.new
    cancelled = false
    spawn { response.stream { |_| } }
    Fiber.yield

    spawn do
      sleep 50.milliseconds
      cancelled = true
      response.cancel
    end

    cancel_proc = -> { cancelled }
    msg, _usage, _finish = Agent::Provider::OpenAI::StreamParser.parse(
      io, response, cancel_proc,
      first_byte_timeout: 10.seconds,
      idle_timeout: 10.seconds,
    )
    response.finish(msg, Agent::Usage.new) unless response.finished?

    response.cancelled?.should be_true
    response.error?.should be_false
  end
end

# Tool-call delta routing. Regression specs for the positional fallback that
# previously mapped every id-less, index-less fragment onto entry 0, merging
# distinct tool calls (id/name overwritten by the later call, argument
# fragments concatenated into invalid JSON).
#
# The motivating shape: some OpenRouter upstreams omit `index` on tool-call
# deltas, and argument fragments carry neither `id` nor `name`, so routing
# must fall back to ids and — as a last resort — the most recently routed
# entry with a name-mismatch guard.
def tool_call_lines(fragments : Array(Hash(String, JSON::Any))) : Array(String)
  fragments.map do |frag|
    body = {"choices" => [{"delta" => {"tool_calls" => [frag]}}]}
    %(data: #{body.to_json})
  end + ["data: [DONE]"]
end

def tool_fragment(id : String?, name : String?, arguments : String?) : Hash(String, JSON::Any)
  frag = {} of String => JSON::Any
  frag["id"] = JSON::Any.new(id) if id
  fn = {} of String => JSON::Any
  fn["name"] = JSON::Any.new(name) if name
  fn["arguments"] = JSON::Any.new(arguments) if arguments
  frag["function"] = JSON::Any.new(fn)
  frag
end

def parse_tool_lines(lines : Array(String)) : Agent::Message
  response = Agent::Response.new
  spawn { response.stream { |_| } } # consume chunks so push_chunk doesn't block
  Fiber.yield
  io = IO::Memory.new(lines.join('\n'))
  msg, usage, _finish = Agent::Provider::OpenAI::StreamParser.parse(io, response, -> { false })
  response.finish(msg, usage)
  msg
end

def parse_tool_fragments(fragments : Array(Hash(String, JSON::Any))) : Agent::Message
  parse_tool_lines(tool_call_lines(fragments))
end

describe Agent::Provider::OpenAI::StreamParser do
  it "separates interleaved parallel calls when fragments carry ids but no index" do
    msg = parse_tool_fragments([
      tool_fragment("call_a", "update_document_view", ""),
      tool_fragment("call_b", "preview_view", ""),
      tool_fragment("call_a", nil, %({"path":"x"})),
      tool_fragment("call_b", nil, %({"kind":"page"})),
    ])

    tcs = msg.tool_calls.should_not be_nil
    tcs.size.should eq(2)
    tcs[0].id.should eq("call_a")
    tcs[0].name.should eq("update_document_view")
    tcs[0].arguments.should eq(%({"path":"x"}))
    tcs[1].id.should eq("call_b")
    tcs[1].name.should eq("preview_view")
    tcs[1].arguments.should eq(%({"kind":"page"}))
  end

  it "separates sequential calls when fragments have no index and the id appears only on each call's first fragment" do
    # Previously: every fragment fell back to positional entry 0, merging
    # the two calls into one (id/name from the later call, argument
    # fragments concatenated into invalid JSON).
    msg = parse_tool_fragments([
      tool_fragment("call_a", "update_document_view", ""),
      tool_fragment(nil, nil, %({"template":{"type":"page"}})),
      tool_fragment("call_b", "preview_view", ""),
      tool_fragment(nil, nil, %({"kind":"page"})),
    ])

    tcs = msg.tool_calls.should_not be_nil
    tcs.size.should eq(2)
    tcs[0].id.should eq("call_a")
    tcs[0].name.should eq("update_document_view")
    tcs[0].arguments.should eq(%({"template":{"type":"page"}}))
    tcs[1].id.should eq("call_b")
    tcs[1].name.should eq("preview_view")
    tcs[1].arguments.should eq(%({"kind":"page"}))
  end

  it "separates calls whose fragments reuse one index value with distinct ids" do
    # Guard against upstreams that re-emit index 0 for every parallel call:
    # a fresh id must never merge into an entry already carrying another id.
    lines = [
      %(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"update_document_view","arguments":""}}]}}]}),
      %(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"template\\":{\\"type\\":\\"page\\"}}"}}]}}]}),
      %(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_b","function":{"name":"preview_view","arguments":""}}]}}]}),
      %(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"kind\\":\\"page\\"}"}}]}}]}),
      "data: [DONE]",
    ]
    msg = parse_tool_lines(lines)

    tcs = msg.tool_calls.should_not be_nil
    tcs.size.should eq(2)
    tcs[0].id.should eq("call_a")
    tcs[0].arguments.should eq(%({"template":{"type":"page"}}))
    tcs[1].id.should eq("call_b")
    tcs[1].arguments.should eq(%({"kind":"page"}))
  end

  it "opens a new call when an id-less, index-less stream starts naming a different function" do
    # Worst case: no id and no index anywhere. The name-mismatch guard is
    # the only thing standing between the two calls.
    msg = parse_tool_fragments([
      tool_fragment(nil, "update_document_view", ""),
      tool_fragment(nil, nil, %({"template":{"type":"page"}})),
      tool_fragment(nil, "preview_view", ""),
      tool_fragment(nil, nil, %({"kind":"page"})),
    ])

    tcs = msg.tool_calls.should_not be_nil
    tcs.size.should eq(2)
    tcs[0].name.should eq("update_document_view")
    tcs[0].arguments.should eq(%({"template":{"type":"page"}}))
    tcs[1].name.should eq("preview_view")
    tcs[1].arguments.should eq(%({"kind":"page"}))
  end

  it "keeps index-based accumulation unchanged for standard OpenAI-shaped streams" do
    lines = [
      %(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"update_document_view","arguments":""}}]}}]}),
      %(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"template\\":{\\"type\\":\\"page\\"}}"}}]}}]}),
      %(data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"preview_view","arguments":""}}]}}]}),
      %(data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"kind\\":\\"page\\"}"}}]}}]}),
      %(data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
      "data: [DONE]",
    ]
    msg = parse_tool_lines(lines)

    tcs = msg.tool_calls.should_not be_nil
    tcs.size.should eq(2)
    tcs[0].id.should eq("call_a")
    tcs[0].name.should eq("update_document_view")
    tcs[0].arguments.should eq(%({"template":{"type":"page"}}))
    tcs[1].id.should eq("call_b")
    tcs[1].name.should eq("preview_view")
    tcs[1].arguments.should eq(%({"kind":"page"}))
  end

  it "joins fragments sharing an id even when the id repeats on every fragment" do
    # Providers that repeat the full name on every fragment must not open a
    # new entry per fragment, and repeated names must not duplicate the
    # ToolCallName chunk stream state.
    msg = parse_tool_fragments([
      tool_fragment("call_a", "update_document_view", ""),
      tool_fragment("call_a", "update_document_view", %({"path")),
      tool_fragment("call_a", "update_document_view", %(:"x"})),
    ])

    tcs = msg.tool_calls.should_not be_nil
    tcs.size.should eq(1)
    tcs[0].id.should eq("call_a")
    tcs[0].name.should eq("update_document_view")
    tcs[0].arguments.should eq(%({"path":"x"}))
  end
end

describe Agent::Provider::OpenAI::StreamParser do
  it "reads cached_tokens from prompt_tokens_details" do
    response = Agent::Response.new
    lines = [
      %(data: {"choices":[{"delta":{"content":"ok"},"index":0,"finish_reason":"stop"}]}),
      %(data: {"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":3,"total_tokens":123,"prompt_tokens_details":{"cached_tokens":100}}}),
      "data: [DONE]",
    ]
    io = IO::Memory.new(lines.join('\n'))
    _msg, usage, _finish = Agent::Provider::OpenAI::StreamParser.parse(io, response, -> { false })
    usage.prompt_tokens.should eq(120)
    usage.cached_tokens.should eq(100)
    usage.cache_creation_tokens.should be_nil
  end
end

describe Agent::Provider::OpenAI::RequestBody do
  it "asks for usage on the stream unless told not to" do
    messages = [Agent::Message.user("hi")]
    body = Agent::Provider::OpenAI::RequestBody.build(messages, nil, "m", nil, nil, nil)
    body["stream_options"].as_h["include_usage"].as_bool.should be_true

    body = Agent::Provider::OpenAI::RequestBody.build(messages, nil, "m", nil, nil, nil, include_usage: false)
    body.has_key?("stream_options").should be_false
  end

  it "takes include_usage from the config" do
    config = Agent::Config.new(api_key: "k", include_usage: false)
    provider = Agent::Provider::OpenAI.new(config, "cache")
    request = provider.build_request([Agent::Message.user("hi")], nil)
    JSON.parse(request[:body]).as_h.has_key?("stream_options").should be_false
  end
end

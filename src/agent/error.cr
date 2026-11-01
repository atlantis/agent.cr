require "http/headers"
require "http/common"

class Agent
  # Base error for all Agent-related failures.
  class Error < Exception
    # Statuses worth retrying: timeout, conflict, rate limit, server error.
    def self.retryable_status?(code : Int32) : Bool
      code.in?(408, 409, 429) || code >= 500
    end
  end

  # Raised when a network or connection error occurs during an API request.
  class ConnectionError < Error
    def initialize(message : String, cause : Exception? = nil)
      super(message, cause: cause)
    end
  end

  # Raised when the API returns a non-2xx status code.
  class ApiError < Error
    getter status_code : Int32
    # The start of the response body (up to BODY_LIMIT bytes), usually the
    # provider's JSON error.
    getter body : String?
    getter headers : HTTP::Headers?

    BODY_LIMIT = 8192

    def initialize(@status_code : Int32, message : String, @body : String? = nil, @headers : HTTP::Headers? = nil)
      super(message)
    end

    # The provider's id for the failed request, for its support team.
    def request_id : String?
      @headers.try { |h| h["x-request-id"]? || h["request-id"]? }
    end

    # Longest wait `#retry_after` reports; anything beyond is clamped to it.
    MAX_RETRY_AFTER = 365.days

    # How long the provider asked to wait: `retry-after-ms`, or
    # `Retry-After` in seconds or as an HTTP date. A value that isn't a plain
    # non-negative number or a date is ignored; a past date means now.
    def retry_after : Time::Span?
      headers = @headers || return
      if ms = plain_number(headers["retry-after-ms"]?)
        return clamp_wait(ms / 1000)
      end
      value = headers["Retry-After"]? || return
      if seconds = plain_number(value)
        return clamp_wait(seconds)
      end
      HTTP.parse_time(value).try { |at| clamp_wait((at - Time.utc).total_seconds) }
    end

    private def plain_number(value : String?) : Float64?
      value = value.try(&.strip) || return
      return unless value.matches?(/\A\d+(\.\d+)?\z/)
      value.to_f? || Float64::INFINITY # too many digits for a Float64
    end

    private def clamp_wait(seconds : Float64) : Time::Span
      return Time::Span.zero unless seconds > 0
      seconds >= MAX_RETRY_AFTER.total_seconds ? MAX_RETRY_AFTER : seconds.seconds
    end

    # Rate limits, conflicts and server errors may succeed on a later try.
    # The provider's `x-should-retry` header, when sent, decides instead.
    def retryable? : Bool
      case @headers.try(&.["x-should-retry"]?)
      when "true"  then true
      when "false" then false
      else              Error.retryable_status?(@status_code)
      end
    end
  end

  # Raised when a request was accepted (HTTP 200) but its stream failed: the
  # provider sent an error event, the connection dropped, the stream ended
  # before the model finished, or it couldn't be read. `Response#partial?`
  # says whether output had already streamed.
  class StreamError < Error
    # The provider's error code, when it sent one.
    getter code : String?

    # Error codes, besides retryable HTTP statuses, that name a transient
    # failure (OpenAI and Anthropic spellings).
    TRANSIENT_CODES = %w[server_error rate_limit_exceeded rate_limit_error overloaded_error api_error timeout]

    # Whether the failure may succeed on a later try. It is only acted on
    # before any output streamed; after that nothing is retried.
    getter? retryable : Bool

    def initialize(message : String, @code : String? = nil, cause : Exception? = nil, retryable : Bool? = nil)
      super(message, cause: cause)
      @retryable = retryable.nil? ? transient_code? : retryable
    end

    private def transient_code? : Bool
      code = @code || return false
      code.to_i?.try { |status| Error.retryable_status?(status) } || code.in?(TRANSIENT_CODES)
    end
  end

  # Raised when the automatic tool-resolution loop exceeds the iteration limit.
  class ToolLoopError < Error
    getter max_iterations : Int32

    def initialize(@max_iterations : Int32, message : String)
      super(message)
    end
  end

  # Raise from a tool callback to show the model why the tool failed. Its
  # message goes into the tool result verbatim; any other exception is
  # logged and the model sees only that the tool failed, so internal
  # details (SQL errors, hosts, tokens) never reach it.
  class ToolError < Error
  end

  # Raised when the caller cancels an in-flight response.
  class CancelledError < Error
    def initialize
      super("Response was cancelled by caller")
    end
  end

  # Raised when an SSE stream stalls — no bytes arrive on `body_io` within
  # the configured first-byte or idle read timeout. The HTTP fiber catches
  # `IO::TimeoutError` from the underlying socket and re-raises it as this so
  # callers can pattern-match on `Agent::Error` types.
  #
  # `phase` is `:first_byte` (no data between POST-send and the first SSE
  # line) or `:idle` (gap between two consecutive lines mid-stream).
  class IdleTimeoutError < Error
    getter phase : Symbol
    getter timeout : Time::Span?

    def initialize(@phase : Symbol, @timeout : Time::Span? = nil, cause : Exception? = nil)
      budget = timeout.try(&.total_seconds.try(&.round(2)))
      msg = case @phase
            when :first_byte then "provider did not emit the first byte within #{budget}s of sending the request"
            else                  "provider stream stalled — no bytes received within the #{budget}s idle timeout"
            end
      super(msg, cause: cause)
    end
  end

  # Raised when loading a session from serialised data fails due to
  # missing fields, type mismatches, or invalid message format.
  class SessionLoadError < Error
    getter reason : String

    def initialize(@reason : String, cause : Exception? = nil)
      super("Failed to load session: #{reason}", cause: cause)
    end
  end
end

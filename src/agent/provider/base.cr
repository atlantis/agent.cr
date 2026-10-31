require "http"

class Agent
  module Provider
    # Abstract interface for API providers.
    #
    # A provider owns all wire-format concerns: how to build the request
    # (path, headers, JSON body) and how to parse the streaming response.
    # The Agent core (fiber loop, history, tool resolution) talks only to
    # this interface and to canonical domain types (Message, ToolCall, Usage).
    abstract class Base
      # Build an HTTP request for the given messages and optional tools.
      # Returns the path (relative to the provider's base URI), headers,
      # and the JSON body string.
      abstract def build_request(messages : Array(Message), tools : Array(Tool)?) : NamedTuple(path: String, headers: HTTP::Headers, body: String)

      # Like the above, with the per-request options of the #ask. The agent
      # calls this one; the default ignores the options, so a provider written
      # before they existed keeps working.
      def build_request(messages : Array(Message), tools : Array(Tool)?, options : RequestOptions) : NamedTuple(path: String, headers: HTTP::Headers, body: String)
        build_request(messages, tools)
      end

      # Parse a streaming SSE response body and push Chunks onto `response`.
      # Returns the final assembled {Message, Usage, finish_reason}.
      # The `cancel` callback returns true if the caller has requested cancellation.
      #
      # `first_byte_timeout` / `idle_byte_timeout` are per-read Time::Span
      # budgets (AGENTS.md). Implementations should apply `first_byte_timeout`
      # to the read preceding the first SSE byte, and switch to
      # `idle_byte_timeout` afterward; any byte arrival (including comment /
      # heartbeat lines, empty lines, `[DONE]` sentinel) resets the timer.
      # When the IO does not support a per-read timeout, providers may safely
      # fall back to the HTTP client's configured `read_timeout`.
      abstract def parse_stream(
        io : IO,
        response : Response,
        cancel : -> Bool,
        first_byte_timeout : Time::Span? = nil,
        idle_timeout : Time::Span? = nil,
      ) : {Message, Usage, String?}

      # The base URI for the HTTP client (scheme, host, port, optional path prefix).
      abstract def base_uri : URI

      # Release any provider-owned resources (HTTP clients, etc.).
      abstract def close : Nil
    end
  end
end

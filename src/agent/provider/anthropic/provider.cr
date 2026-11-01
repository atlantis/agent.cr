require "http"
require "json"
require "uri"

class Agent
  module Provider
    # The Anthropic Messages API, streamed. An assistant reply's content
    # blocks are kept as the API returned them, in `provider_data`, and sent
    # back unchanged: a thinking block carries a signature the API checks,
    # and a rebuilt one is refused.
    #
    # Settings come from `Agent::Config`: `api_endpoint` (the API root,
    # e.g. DEFAULT_ENDPOINT, or a gateway with a path prefix), `api_key`,
    # `model`, `max_tokens`, `reasoning_effort`, `extra_headers` and
    # `extra_body`. `temperature` is not sent: API version 2026-10-21
    # refuses sampling parameters.
    class Anthropic < Base
      DEFAULT_ENDPOINT   = "https://api.anthropic.com"
      API_VERSION        = "2026-10-21"
      DEFAULT_MAX_TOKENS = 64_000
      # Key of this provider's entry in `Message#provider_data`.
      DATA_KEY = "anthropic"
      # Lets a request set `thinking.block_binding.prefix_mismatch_behavior`.
      BINDING_BETA    = "thinking-binding-controls-2026-08-01"
      DISPLAYS        = {"summarized", "omitted"}
      PREFIX_MISMATCH = {"drop_block", "error"}

      getter model : String
      getter max_tokens : Int32
      getter reasoning_effort : ReasoningEffort?
      # Sends `thinking: {type: "adaptive"}`. Off only for a model without
      # adaptive thinking (Haiku 4.5 and older).
      getter? thinking : Bool
      # `thinking.display`: "summarized" streams a readable summary into
      # `Message#reasoning`; "omitted" (the API default) streams empty
      # thinking. Thinking is billed the same either way.
      getter display : String?
      # What the API does with a thinking block whose conversation changed
      # since it was produced, which trimming history (`max_history`) does:
      # "drop_block" drops it and the reply goes on without that reasoning;
      # "error" refuses the request (400); nil sends nothing, leaving the
      # account's default, which is "error" for accounts created on or after
      # 2026-08-31.
      getter prefix_mismatch : String?
      # Streams tool inputs as they are generated rather than buffered. The
      # agent validates the arguments before it runs a tool, and does not run
      # the calls of a reply cut off by max_tokens.
      getter? eager_input_streaming : Bool

      @base_uri : URI
      @endpoint : String
      @api_key : String?
      @extra_headers : Hash(String, String)?
      @extra_body : Hash(String, JSON::Any)?
      @caching : PromptCaching
      @cache_ttl : CacheTTL

      def initialize(
        config : Agent::Config,
        *,
        endpoint : String? = nil,
        @thinking : Bool = true,
        @display : String? = "summarized",
        @prefix_mismatch : String? = "drop_block",
        @eager_input_streaming : Bool = true,
      )
        endpoint ||= config.api_endpoint
        uri = URI.parse(endpoint)
        unless uri.scheme && uri.host
          raise ArgumentError.new("endpoint must be a url like #{DEFAULT_ENDPOINT}, got #{endpoint.inspect}")
        end
        if uri.host == "api.openai.com"
          raise ArgumentError.new("api_endpoint is OpenAI's (the Config default); set it to #{DEFAULT_ENDPOINT} or pass endpoint:")
        end
        if (d = @display) && !d.in?(DISPLAYS)
          raise ArgumentError.new("display must be one of #{DISPLAYS.join(", ")}, got #{d.inspect}")
        end
        if (pm = @prefix_mismatch) && !pm.in?(PREFIX_MISMATCH)
          raise ArgumentError.new("prefix_mismatch must be one of #{PREFIX_MISMATCH.join(", ")}, got #{pm.inspect}")
        end

        @base_uri = uri
        @endpoint = endpoint.rstrip('/')
        @api_key = config.api_key
        @model = config.model
        @max_tokens = config.max_tokens || DEFAULT_MAX_TOKENS
        @reasoning_effort = config.reasoning_effort
        @extra_headers = config.extra_headers
        @extra_body = config.extra_body
        @caching = config.prompt_caching
        @cache_ttl = config.cache_ttl
      end

      def base_uri : URI
        @base_uri
      end

      def close : Nil
      end

      # The endpoint's path prefix (a gateway) plus /v1/messages; an
      # endpoint that already ends in /v1 is taken as the API root's /v1.
      def path : String
        prefix = @base_uri.path.gsub(/\/+/, "/").rstrip('/').chomp("/v1")
        "#{prefix}/v1/messages"
      end

      def build_request(messages : Array(Message), tools : Array(Tool)?) : NamedTuple(path: String, headers: HTTP::Headers, body: String)
        build_request(messages, tools, RequestOptions.new)
      end

      def build_request(messages : Array(Message), tools : Array(Tool)?, options : RequestOptions) : NamedTuple(path: String, headers: HTTP::Headers, body: String)
        body = RequestBody.build(
          messages, tools,
          model: @model,
          max_tokens: @max_tokens,
          effort: options.reasoning_effort || @reasoning_effort,
          thinking: @thinking,
          display: @display,
          prefix_mismatch: @prefix_mismatch,
          eager_input_streaming: @eager_input_streaming,
          extra_body: @extra_body,
          endpoint: @endpoint,
          caching: @caching,
          ttl: @cache_ttl,
        )
        {path: path, headers: headers, body: body.to_json}
      end

      private def headers : HTTP::Headers
        headers = HTTP::Headers{
          "content-type"      => "application/json",
          "accept"            => "text/event-stream",
          "anthropic-version" => API_VERSION,
        }
        @api_key.try { |key| headers["x-api-key"] = key }

        betas = [] of String
        @extra_headers.try &.each do |k, v|
          if k.downcase == "anthropic-beta"
            betas.concat(v.split(',').map(&.strip).reject(&.empty?))
          else
            headers[k] = v
          end
        end
        betas << BINDING_BETA if @thinking && @prefix_mismatch && !betas.includes?(BINDING_BETA)
        headers["anthropic-beta"] = betas.join(",") unless betas.empty?
        headers
      end

      def parse_stream(
        io : IO,
        response : Response,
        cancel : -> Bool,
        first_byte_timeout : Time::Span? = nil,
        idle_timeout : Time::Span? = nil,
      ) : {Message, Usage, String?}
        StreamParser.parse(io, response, cancel,
          first_byte_timeout: first_byte_timeout,
          idle_timeout: idle_timeout,
          endpoint: @endpoint)
      end

      # Replay data for an assistant reply: its content blocks, with the
      # endpoint that produced them.
      def self.replay_data(endpoint : String, blocks : Array(JSON::Any)) : JSON::Any
        JSON::Any.new({DATA_KEY => JSON::Any.new({
          "endpoint" => JSON::Any.new(endpoint),
          "blocks"   => JSON::Any.new(blocks),
        })})
      end

      # The content blocks kept on an assistant reply, nil when it wasn't
      # this provider's (or a cancel kept only its text). With `endpoint`,
      # only blocks that came from it.
      def self.blocks(message : Message, endpoint : String? = nil) : Array(JSON::Any)?
        ours = message.provider_data.try(&.as_h?).try(&.[DATA_KEY]?).try(&.as_h?) || return
        return if endpoint && ours["endpoint"]?.try(&.as_s?) != endpoint
        ours["blocks"]?.try(&.as_a?)
      end
    end
  end
end

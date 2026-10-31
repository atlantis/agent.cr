require "uri"

class Agent
  module Provider
    # OpenAI-compatible chat completions provider.
    #
    # Implements the Provider::Base interface for the OpenAI wire format,
    # including SSE streaming, tool calls, reasoning content, and
    # the standard /chat/completions endpoint.
    class OpenAI < Base
      # Configuration specific to the OpenAI-compatible provider.
      class Config
        getter api_endpoint : String
        getter api_key : String?
        getter model : String
        getter system_prompt : String?
        getter max_tokens : Int32?
        getter temperature : Float64?
        getter read_timeout : Time::Span?
        getter connect_timeout : Time::Span?
        getter extra_headers : Hash(String, String)?
        getter prompt_cache_key : String?
        # Ask for a `usage` object on the stream's last chunk
        # (`stream_options: {include_usage: true}`). Without it OpenAI,
        # Ollama, vLLM and others report no token counts on streamed
        # completions. On by default; turn it off for a server that rejects
        # the field.
        getter? include_usage : Bool
        # See Agent::Config#reasoning_effort.
        getter reasoning_effort : ReasoningEffort?
        # Extra top-level body fields; see Agent::Config#extra_body.
        getter extra_body : Hash(String, JSON::Any)?
        # See Agent::Config#replay_reasoning?.
        getter? replay_reasoning : Bool

        # Cached parsed URI — computed once at construction.
        getter parsed_uri : URI

        def initialize(
          @api_key : String? = nil,
          @api_endpoint : String = "https://api.openai.com/v1",
          @model : String = "gpt-4o",
          @system_prompt : String? = nil,
          @max_tokens : Int32? = nil,
          @temperature : Float64? = nil,
          read_timeout : Time::Span | Int32? = nil,
          connect_timeout : Time::Span | Int32? = nil,
          @extra_headers : Hash(String, String)? = nil,
          @prompt_cache_key : String? = nil,
          @include_usage : Bool = true,
          @reasoning_effort : ReasoningEffort? = nil,
          @extra_body : Hash(String, JSON::Any)? = nil,
          @replay_reasoning : Bool = true,
        )
          validate_temperature(@temperature)
          validate_max_tokens(@max_tokens)

          # Accept Int32 seconds for timeouts (convenience)
          @read_timeout = parse_timeout(read_timeout)
          @connect_timeout = parse_timeout(connect_timeout)

          # Validate and parse the endpoint URI
          @parsed_uri = URI.parse(@api_endpoint)
          unless @parsed_uri.scheme && @parsed_uri.host
            raise ArgumentError.new("api_endpoint must be a valid URL, got #{@api_endpoint}")
          end
        end

        # Identifies the endpoint in the replay data this provider stores:
        # signed and encrypted blocks are only valid where they came from.
        def replay_endpoint : String
          @api_endpoint.rstrip('/')
        end

        # OpenRouter takes effort as `reasoning: {effort: ...}`; OpenAI and
        # servers that follow it take `reasoning_effort`.
        def openrouter? : Bool
          @parsed_uri.host.try(&.ends_with?("openrouter.ai")) || false
        end

        # The chat completions path derived from api_endpoint.
        def chat_path : String
          base_path = @parsed_uri.path.empty? || @parsed_uri.path == "/" ? "" : @parsed_uri.path.rstrip('/')
          "#{base_path}/chat/completions"
        end

        private def validate_temperature(t : Float64?) : Nil
          if t && (t < 0.0 || t > 2.0)
            raise ArgumentError.new("temperature must be between 0.0 and 2.0, got #{t}")
          end
        end

        private def validate_max_tokens(mt : Int32?) : Nil
          if mt && mt <= 0
            raise ArgumentError.new("max_tokens must be positive, got #{mt}")
          end
        end

        private def parse_timeout(timeout : Time::Span | Int32?) : Time::Span?
          case timeout
          when Int32      then timeout.seconds
          when Time::Span then timeout
          end
        end
      end

      # Key of this provider's entry in `Message#provider_data` and
      # `ToolCall#provider_data`.
      DATA_KEY = "openai"

      # Replay data for a message or tool call, under DATA_KEY with the
      # endpoint and model that produced it.
      def self.replay_data(endpoint : String?, model : String?, entries : Hash(String, JSON::Any)) : JSON::Any
        scope = {"endpoint" => JSON::Any.new(endpoint), "model" => JSON::Any.new(model)}
        JSON::Any.new({DATA_KEY => JSON::Any.new(scope.merge(entries))})
      end

      # What this provider stored, if it came from the same endpoint and
      # model: signed and encrypted blocks are refused anywhere else, and one
      # endpoint (OpenRouter) serves many models.
      def self.replay_entries(data : JSON::Any?, endpoint : String?, model : String?) : Hash(String, JSON::Any)?
        ours = data.try(&.as_h?).try(&.[DATA_KEY]?).try(&.as_h?) || return
        return unless ours["endpoint"]?.try(&.as_s?) == endpoint && ours["model"]?.try(&.as_s?) == model
        ours
      end

      # --- Provider::Base implementation ---

      @config : Config
      @cache_key : String

      def initialize(config : Agent::Config, cache_key : String)
        @config = Config.new(
          api_key: config.api_key,
          api_endpoint: config.api_endpoint,
          model: config.model,
          system_prompt: config.system_prompt,
          max_tokens: config.max_tokens,
          temperature: config.temperature,
          read_timeout: config.read_timeout,
          connect_timeout: config.connect_timeout,
          extra_headers: config.extra_headers,
          prompt_cache_key: config.prompt_cache_key,
          include_usage: config.include_usage?,
          reasoning_effort: config.reasoning_effort,
          extra_body: config.extra_body,
          replay_reasoning: config.replay_reasoning?,
        )
        @cache_key = cache_key
      end

      def initialize(config : Config, cache_key : String)
        @config = config
        @cache_key = cache_key
      end

      # The base URI for the HTTP client.
      def base_uri : URI
        @config.parsed_uri
      end

      # Build an HTTP request for the OpenAI chat completions API.
      def build_request(messages : Array(Message), tools : Array(Tool)?) : NamedTuple(path: String, headers: HTTP::Headers, body: String)
        build_request(messages, tools, RequestOptions.new)
      end

      def build_request(messages : Array(Message), tools : Array(Tool)?, options : RequestOptions) : NamedTuple(path: String, headers: HTTP::Headers, body: String)
        body_hash = RequestBody.build(
          messages,
          tools,
          @config.model,
          @config.max_tokens,
          @config.temperature,
          @config.prompt_cache_key ? @cache_key : nil,
          include_usage: @config.include_usage?,
          reasoning_effort: options.reasoning_effort || @config.reasoning_effort,
          openrouter: @config.openrouter?,
          extra_body: @config.extra_body,
          endpoint: @config.replay_endpoint,
          replay: @config.replay_reasoning?,
        )

        headers = HTTP::Headers.new
        if key = @config.api_key
          headers["Authorization"] = "Bearer #{key}"
        end
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        if extra = @config.extra_headers
          extra.each { |k, v| headers[k] = v }
        end

        {path: @config.chat_path, headers: headers, body: body_hash.to_json}
      end

      # Parse an OpenAI SSE streaming response.
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
          endpoint: @config.replay_endpoint,
          model: @config.model)
      end

      # Release any provider-owned resources (no-op for OpenAI).
      def close : Nil
        # Nothing to release.
      end
    end
  end
end

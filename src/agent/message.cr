require "json"

class Agent
  # Valid roles for messages as per the OpenAI API.
  enum Role
    System
    User
    Assistant
    Tool

    # Serialise to JSON as lowercase string.
    def to_s : String
      case self
      in System    then "system"
      in User      then "user"
      in Assistant then "assistant"
      in Tool      then "tool"
      end
    end

    # Parse from a lowercase string. Raises ArgumentError for invalid input.
    def self.parse(value : String) : Role
      case value.strip.downcase
      when "system"    then System
      when "user"      then User
      when "assistant" then Assistant
      when "tool"      then Tool
      else                  raise ArgumentError.new("Unknown role: '#{value}'")
      end
    end
  end

  # JSON converter for Role serialization with JSON::Serializable.
  module RoleConverter
    extend self

    def to_json(value : Role, json : JSON::Builder) : Nil
      json.string(value.to_s)
    end

    def from_json(value : JSON::PullParser) : Role
      Role.parse(value.read_string)
    end
  end

  # A content part used in multimodal messages.
  # Supports text, image URLs, audio data, and generic file attachments.
  @[JSON::Serializable::Options(emit_nulls: true)]
  class ContentPart
    include JSON::Serializable

    # The type of content this part represents.
    enum PartType
      Text
      ImageUrl
      InputAudio
      File
    end

    # The content type discriminator.
    getter type : PartType = PartType::Text
    # Text content (for Text parts).
    getter text : String?
    # URL or data URI (for ImageUrl, File parts).
    getter url : String?
    # Raw base64-encoded data (for InputAudio parts).
    getter data : String?
    # MIME type (for ImageUrl, InputAudio, File parts).
    getter mime_type : String?
    # Image detail level (for ImageUrl parts only).
    getter image_detail : String?
    # Filename (for File parts).
    getter filename : String?

    def initialize(
      @type : PartType = PartType::Text,
      @text : String? = nil,
      @url : String? = nil,
      @data : String? = nil,
      @mime_type : String? = nil,
      @image_detail : String? = "auto",
      @filename : String? = nil,
    )
      case @type
      when PartType::Text
        raise ArgumentError.new("Text ContentPart must have text content") if @text.nil?
      when PartType::ImageUrl
        raise ArgumentError.new("ImageUrl ContentPart must have a url") if @url.nil?
      when PartType::InputAudio
        raise ArgumentError.new("InputAudio ContentPart must have data") if @data.nil?
        raise ArgumentError.new("InputAudio ContentPart must have mime_type") if @mime_type.nil?
      when PartType::File
        raise ArgumentError.new("File ContentPart must have a url") if @url.nil?
      end
    end

    # Resolve a file path or URL into a ContentPart, auto-detecting the MIME type
    # from the file extension.
    #
    # - **Remote URLs** (`http://`, `https://`) are passed through as-is.
    # - **Data URIs** (`data:`) are parsed to determine the part type.
    # - **Local file paths** are read, base64-encoded, and typed by MIME:
    #   - `image/*` → ImageUrl
    #   - `audio/*` → InputAudio
    #   - `text/*`, `application/json` → Text (content embedded inline)
    #   - everything else → File
    def self.from_path(path : String) : self
      if path.starts_with?("http://") || path.starts_with?("https://")
        from_remote_url(path)
      elsif path.starts_with?("data:")
        from_data_uri(path)
      else
        from_local_file(path)
      end
    end

    # Build a ContentPart from a remote URL.
    private def self.from_remote_url(path : String) : self
      mime = mime_type_from_extension(path)
      type = mime.starts_with?("image/") ? PartType::ImageUrl : PartType::File
      filename = begin
        File.basename(URI.parse(path).path || "")
      rescue
        File.basename(path.split("?").first)
      end
      new(type: type, url: path, mime_type: mime, filename: filename)
    end

    # Build a ContentPart from a data URI.
    private def self.from_data_uri(path : String) : self
      mime = path[5..].split(";").first?
      mime = "application/octet-stream" if mime.nil? || mime.empty?
      # Reject non-base64 data URIs — the provider API won't accept raw data.
      unless path.includes?(";base64,")
        raise ArgumentError.new("Data URI must use base64 encoding (;base64,) — got: #{path[0..50]}")
      end
      type = mime.starts_with?("image/") ? PartType::ImageUrl : PartType::File
      new(type: type, url: path, mime_type: mime)
    end

    # Build a ContentPart from a local file path, reading & encoding as needed.
    # Raises ArgumentError if the file does not exist or exceeds the size limit.
    private def self.from_local_file(path : String) : self
      bytes = read_file_bytes(path)
      mime = mime_type_from_extension(path)

      if mime.starts_with?("image/")
        data_uri = encode_as_data_uri(mime, bytes)
        new(type: PartType::ImageUrl, url: data_uri, mime_type: mime, filename: File.basename(path))
      elsif mime.starts_with?("audio/")
        new(type: PartType::InputAudio, data: Base64.strict_encode(bytes), mime_type: mime, filename: File.basename(path))
      elsif inlinable_text_type?(mime)
        new(type: PartType::Text, text: String.new(bytes), mime_type: mime, filename: File.basename(path))
      else
        data_uri = encode_as_data_uri(mime, bytes)
        new(type: PartType::File, url: data_uri, mime_type: mime, filename: File.basename(path))
      end
    end

    # Encode bytes as a base64 data URI.
    private def self.encode_as_data_uri(mime : String, bytes : Bytes) : String
      "data:#{mime};base64,#{Base64.strict_encode(bytes)}"
    end

    # Returns true if the MIME type should be inlined as text content.
    private def self.inlinable_text_type?(mime : String) : Bool
      mime.starts_with?("text/") ||
        mime == "application/json" ||
        mime == "application/xml" ||
        mime == "application/x-yaml"
    end

    # :nodoc:
    def to_json_body : Hash(String, JSON::Any)
      case @type
      in PartType::Text
        {"type" => JSON::Any.new("text"), "text" => JSON::Any.new(@text || "")}
      in PartType::ImageUrl
        {
          "type"      => JSON::Any.new("image_url"),
          "image_url" => JSON::Any.new({
            "url"    => JSON::Any.new(@url || ""),
            "detail" => JSON::Any.new(@image_detail || "auto"),
          }.to_h),
        }
      in PartType::InputAudio
        format = self.class.audio_format_from_mime(@mime_type || "wav")
        {
          "type"        => JSON::Any.new("input_audio"),
          "input_audio" => JSON::Any.new({
            "data"   => JSON::Any.new(@data || ""),
            "format" => JSON::Any.new(format),
          }.to_h),
        }
      in PartType::File
        {
          "type" => JSON::Any.new("file"),
          "file" => JSON::Any.new({
            "file_data" => JSON::Any.new(@url || ""),
            "filename"  => JSON::Any.new(@filename || ""),
          }.to_h),
        }
      end
    end

    # --- helpers ---

    # Default maximum attachment size (25 MB, matching OpenAI's limit).
    MAX_ATTACHMENT_SIZE = 25 * 1024 * 1024

    def self.read_file_bytes(path : String) : Bytes
      raise ArgumentError.new("File not found: #{path}") unless File.exists?(path)

      File.open(path, "rb") do |f|
        size = f.size
        if size > MAX_ATTACHMENT_SIZE
          raise ArgumentError.new("Attachment too large: #{size} bytes (max #{MAX_ATTACHMENT_SIZE})")
        end
        slice = Slice(UInt8).new(size)
        f.read_fully(slice)
        slice
      end
    rescue e : ArgumentError
      raise e
    rescue File::Error
      raise ArgumentError.new("Cannot read attachment: #{path}")
    end

    # Map a file extension to a MIME type.
    # Uses a lookup table of ~30 extension-to-MIME mappings.
    MIME_MAP = {
      ".jpg"      => "image/jpeg",
      ".jpeg"     => "image/jpeg",
      ".png"      => "image/png",
      ".gif"      => "image/gif",
      ".webp"     => "image/webp",
      ".bmp"      => "image/bmp",
      ".svg"      => "image/svg+xml",
      ".tiff"     => "image/tiff",
      ".tif"      => "image/tiff",
      ".ico"      => "image/x-icon",
      ".mp3"      => "audio/mpeg",
      ".wav"      => "audio/wav",
      ".ogg"      => "audio/ogg",
      ".flac"     => "audio/flac",
      ".aac"      => "audio/aac",
      ".opus"     => "audio/opus",
      ".mp4"      => "video/mp4",
      ".pdf"      => "application/pdf",
      ".json"     => "application/json",
      ".xml"      => "application/xml",
      ".yaml"     => "application/x-yaml",
      ".yml"      => "application/x-yaml",
      ".csv"      => "text/csv",
      ".html"     => "text/html",
      ".htm"      => "text/html",
      ".md"       => "text/markdown",
      ".markdown" => "text/markdown",
      ".txt"      => "text/plain",
      ".rb"       => "text/x-ruby",
      ".cr"       => "text/x-crystal",
      ".py"       => "text/x-python",
      ".js"       => "text/javascript",
      ".ts"       => "text/typescript",
      ".css"      => "text/css",
      ".sh"       => "text/x-shellscript",
    }

    def self.mime_type_from_extension(path : String) : String
      ext = File.extname(path).downcase
      MIME_MAP.fetch(ext, "application/octet-stream")
    end

    # Map a MIME type to an OpenAI audio format string.
    # Raises ArgumentError for unsupported audio MIME types.
    def self.audio_format_from_mime(mime : String) : String
      case mime
      when "audio/mpeg" then "mp3"
      when "audio/wav"  then "wav"
      when "audio/ogg"  then "ogg"
      when "audio/flac" then "flac"
      when "audio/aac"  then "aac"
      when "audio/opus" then "opus"
      else                   raise ArgumentError.new("Unsupported audio format: #{mime}")
      end
    end
  end

  # Represents a tool call from the assistant.
  @[JSON::Serializable::Options(emit_nulls: true)]
  class ToolCall
    include JSON::Serializable

    getter id : String
    getter name : String
    getter arguments : String

    def initialize(@id : String, @name : String, @arguments : String)
    end

    # :nodoc:
    def ==(other : self) : Bool
      @id == other.id && @name == other.name && @arguments == other.arguments
    end

    def ==(other) : Bool
      false
    end
  end

  # Token usage metadata from the API.
  @[JSON::Serializable::Options(emit_nulls: true)]
  class Usage
    include JSON::Serializable

    getter prompt_tokens : Int32?
    getter completion_tokens : Int32?
    getter total_tokens : Int32?
    # Prompt tokens served from the provider's prompt cache (OpenAI
    # `prompt_tokens_details.cached_tokens`, Anthropic `cache_read_input_tokens`).
    # Part of `prompt_tokens`, not in addition to it.
    getter cached_tokens : Int32?
    # Prompt tokens written to the cache on this request (Anthropic
    # `cache_creation_input_tokens`); nil for providers that don't report it.
    getter cache_creation_tokens : Int32?

    def initialize(
      @prompt_tokens = nil,
      @completion_tokens = nil,
      @total_tokens = nil,
      @cached_tokens = nil,
      @cache_creation_tokens = nil,
    )
    end

    # Nil-safe addition: sums each field over both usages. A field is nil in
    # the result only when it is nil on both sides, so `Usage.new + Usage.new`
    # stays all-nil. Lets callers accumulate usage across a multi-iteration
    # tool loop instead of keeping only the last iteration's counts.
    def +(other : Usage) : Usage
      Usage.new(
        add(prompt_tokens, other.prompt_tokens),
        add(completion_tokens, other.completion_tokens),
        add(total_tokens, other.total_tokens),
        add(cached_tokens, other.cached_tokens),
        add(cache_creation_tokens, other.cache_creation_tokens),
      )
    end

    private def add(a : Int32?, b : Int32?) : Int32?
      return if a.nil? && b.nil?
      (a || 0) + (b || 0)
    end
  end

  # A message in the conversation.
  @[JSON::Serializable::Options(emit_nulls: true)]
  class Message
    include JSON::Serializable

    @[JSON::Field(key: "role", converter: Agent::RoleConverter)]
    getter role : Role
    getter content : String?
    @[JSON::Field(key: "content_parts")]
    getter content_parts : Array(ContentPart)?
    @[JSON::Field(key: "tool_calls")]
    getter tool_calls : Array(ToolCall)?
    @[JSON::Field(key: "tool_call_id")]
    getter tool_call_id : String?
    getter name : String?
    getter reasoning : String?
    # For tool-result messages: whether the result is an error (the tool
    # raised, or its arguments were rejected). The OpenAI wire format has no
    # such flag, so the OpenAI provider ignores it; providers whose format
    # has one (e.g. `is_error` on an Anthropic `tool_result` block) send it.
    @[JSON::Field(key: "is_error")]
    getter is_error : Bool?
    # Provider-specific data a custom provider needs to replay the message
    # exactly as it came (signed reasoning blocks, citations, ...). Opaque to
    # the agent loop: kept through `dump`/`load` and `history`, never sent by
    # the OpenAI provider.
    @[JSON::Field(key: "provider_data")]
    getter provider_data : JSON::Any?

    def initialize(
      @role : Role,
      @content : String? = nil,
      @content_parts : Array(ContentPart)? = nil,
      @tool_calls : Array(ToolCall)? = nil,
      @tool_call_id : String? = nil,
      @name : String? = nil,
      @reasoning : String? = nil,
      @is_error : Bool? = nil,
      @provider_data : JSON::Any? = nil,
    )
    end

    # Create a system message.
    def self.system(content : String) : self
      new(role: Role::System, content: content)
    end

    # Create a user message.
    def self.user(content : String) : self
      new(role: Role::User, content: content)
    end

    # Create an assistant message.
    def self.assistant(content : String? = nil, tool_calls : Array(ToolCall)? = nil, reasoning : String? = nil, provider_data : JSON::Any? = nil) : self
      new(role: Role::Assistant, content: content, tool_calls: tool_calls, reasoning: reasoning, provider_data: provider_data)
    end

    # Create a tool-result message from a completed tool call.
    # Convenient shorthand for building manual-dispatch results.
    # Pass `is_error: true` when the tool failed, so a provider that can
    # mark errors on the wire does.
    #
    # ```
    # msg = Agent::Message.tool_result(tc, "The weather is 22°C and sunny")
    # msg = Agent::Message.tool_result(tc, "no such city", is_error: true)
    # ```
    def self.tool_result(tool_call : ToolCall, content : String, is_error : Bool? = nil) : self
      new(role: Role::Tool, content: content, tool_call_id: tool_call.id, name: tool_call.name, is_error: is_error)
    end

    # Whether this is a tool result that reports an error.
    def error? : Bool
      @is_error == true
    end

    def has_tool_calls? : Bool
      tc = @tool_calls
      !tc.nil? && !tc.empty?
    end

    def ==(other : self) : Bool
      @role == other.role && @content == other.content &&
        @content_parts == other.content_parts &&
        @tool_call_id == other.tool_call_id && @name == other.name &&
        @tool_calls == other.tool_calls && @reasoning == other.reasoning &&
        @is_error == other.is_error && @provider_data == other.provider_data
    end

    def ==(other) : Bool
      false
    end

    def inspect(io : IO) : Nil
      io << "#<Message:"
      io << @role.to_s
      if c = @content
        io << " #{c[0, Math.min(c.size, 80)].inspect}"
      elsif @content_parts
        io << " (multimodal)"
      end
      if tc = @tool_calls
        io << " tool_calls=#{tc.size}"
      end
      if @reasoning
        io << " (reasoning)"
      end
      io << ">"
    end

    # :nodoc:
    def to_request_body : Hash(String, JSON::Any)
      body = {"role" => JSON::Any.new(@role.to_s)}

      # Build content
      if parts = @content_parts
        body["content"] = JSON::Any.new(parts.map(&.to_json_body).map { |h| JSON::Any.new(h) })
      elsif text = @content
        body["content"] = JSON::Any.new(text)
      elsif @role == Role::Tool
        body["content"] = JSON::Any.new("")
      else
        body["content"] = JSON::Any.new(nil)
      end

      body["tool_call_id"] = JSON::Any.new(@tool_call_id) if @tool_call_id
      body["name"] = JSON::Any.new(@name) if @name

      if tcs = @tool_calls
        body["tool_calls"] = JSON::Any.new(tcs.map do |tc|
          JSON::Any.new({
            "id"       => JSON::Any.new(tc.id),
            "type"     => JSON::Any.new("function"),
            "function" => JSON::Any.new({
              "name"      => JSON::Any.new(tc.name),
              "arguments" => JSON::Any.new(tc.arguments),
            }),
          })
        end)
      end

      body
    end
  end
end

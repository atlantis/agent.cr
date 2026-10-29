require "./spec_helper"

describe Agent::Message do
  it "builds a simple text message body" do
    msg = Agent::Message.new(role: Agent::Role::User, content: "Hello")
    body = msg.to_request_body
    body["role"].as_s.should eq("user")
    body["content"].as_s.should eq("Hello")
  end

  it "builds a multimodal message body" do
    parts = [
      Agent::ContentPart.new(type: :text, text: "What is this?"),
      Agent::ContentPart.new(type: :image_url, url: "https://example.com/img.jpg"),
    ]
    msg = Agent::Message.new(role: Agent::Role::User, content_parts: parts)
    body = msg.to_request_body
    content = body["content"].as_a
    content[0].as_h["type"].as_s.should eq("text")
    content[1].as_h["type"].as_s.should eq("image_url")
  end

  it "builds a message with tool calls" do
    tool_calls = [Agent::ToolCall.new(id: "call_1", name: "get_weather", arguments: %({"city":"Paris"}))]
    msg = Agent::Message.new(role: Agent::Role::Assistant, tool_calls: tool_calls)
    body = msg.to_request_body
    body["tool_calls"].as_a.size.should eq(1)
  end

  it "detects tool calls via has_tool_calls?" do
    msg = Agent::Message.new(role: Agent::Role::Assistant, content: "Hello")
    msg.has_tool_calls?.should be_false

    tcs = [Agent::ToolCall.new(id: "c1", name: "fn", arguments: "{}")]
    msg2 = Agent::Message.new(role: Agent::Role::Assistant, tool_calls: tcs)
    msg2.has_tool_calls?.should be_true
  end
end

describe Agent::ContentPart do
  it "produces image JSON" do
    part = Agent::ContentPart.new(type: :image_url, url: "https://ex.com/i.png", image_detail: "high")
    json = part.to_json_body
    json["type"].as_s.should eq("image_url")
    json["image_url"].as_h["url"].as_s.should eq("https://ex.com/i.png")
  end

  it "produces text JSON" do
    part = Agent::ContentPart.new(type: :text, text: "Hello")
    json = part.to_json_body
    json["type"].as_s.should eq("text")
    json["text"].as_s.should eq("Hello")
  end
end

describe Agent::ToolCall do
  it "stores tool call data" do
    tc = Agent::ToolCall.new(id: "tc_1", name: "fn", arguments: "{}")
    tc.id.should eq("tc_1")
    tc.name.should eq("fn")
    tc.arguments.should eq("{}")
  end
end

describe Agent::Usage do
  it "defaults to nil fields" do
    u = Agent::Usage.new
    u.prompt_tokens.should be_nil
    u.completion_tokens.should be_nil
    u.total_tokens.should be_nil
  end

  it "produces file JSON" do
    part = Agent::ContentPart.new(type: :file, url: "data:application/pdf;base64,AAAA", filename: "doc.pdf", mime_type: "application/pdf")
    json = part.to_json_body
    json["type"].as_s.should eq("file")
    json["file"].as_h["file_data"].as_s.should eq("data:application/pdf;base64,AAAA")
    json["file"].as_h["filename"].as_s.should eq("doc.pdf")
  end

  it "produces audio JSON" do
    part = Agent::ContentPart.new(type: :input_audio, data: "base64data", mime_type: "audio/wav")
    json = part.to_json_body
    json["type"].as_s.should eq("input_audio")
    json["input_audio"].as_h["format"].as_s.should eq("wav")
    json["input_audio"].as_h["data"].as_s.should eq("base64data")
  end

  it "resolves a remote URL to ImageUrl or File" do
    part = Agent::ContentPart.from_path("https://example.com/photo.jpg")
    part.type.should eq(Agent::ContentPart::PartType::ImageUrl)
    part.url.should eq("https://example.com/photo.jpg")
    part.mime_type.should eq("image/jpeg")

    part2 = Agent::ContentPart.from_path("https://example.com/doc.pdf")
    part2.type.should eq(Agent::ContentPart::PartType::File)
    part2.url.should eq("https://example.com/doc.pdf")
    part2.mime_type.should eq("application/pdf")
  end

  it "resolves a local text file to Text part" do
    # Use a tempfile
    filename = File.tempname("agent-test", ".md")
    File.write(filename, "# Hello\n\nThis is a test.")
    part = Agent::ContentPart.from_path(filename)
    part.type.should eq(Agent::ContentPart::PartType::Text)
    part.text.should eq("# Hello\n\nThis is a test.")
    part.mime_type.should eq("text/markdown")
    part.filename.should eq(File.basename(filename))
  ensure
    File.delete(filename) if filename && File.exists?(filename)
  end

  it "resolves a local image file to ImageUrl part" do
    filename = File.tempname("agent-test", ".png")
    File.write(filename, Bytes[0x89, 0x50, 0x4E, 0x47]) # minimal PNG header
    part = Agent::ContentPart.from_path(filename)
    part.type.should eq(Agent::ContentPart::PartType::ImageUrl)
    part.url.should_not be_nil
    part.url.should match(/^data:image\/png;base64,/)
    part.mime_type.should eq("image/png")
  ensure
    File.delete(filename) if filename && File.exists?(filename)
  end

  it "accepts values" do
    u = Agent::Usage.new(prompt_tokens: 10, completion_tokens: 20, total_tokens: 30)
    u.total_tokens.should eq(30)
  end

  it "carries and sums the cache counts" do
    a = Agent::Usage.new(prompt_tokens: 10, completion_tokens: 5, total_tokens: 15, cached_tokens: 8, cache_creation_tokens: 2)
    b = Agent::Usage.new(prompt_tokens: 20, completion_tokens: 8, total_tokens: 28, cached_tokens: 20)
    sum = a + b

    sum.cached_tokens.should eq(28)
    sum.cache_creation_tokens.should eq(2)
    (Agent::Usage.new + Agent::Usage.new).cached_tokens.should be_nil
    Agent::Usage.from_json(a.to_json).cached_tokens.should eq(8)
  end

  it "sums every field with nil-safe +" do
    a = Agent::Usage.new(prompt_tokens: 10, completion_tokens: 5, total_tokens: 15)
    b = Agent::Usage.new(prompt_tokens: 20, completion_tokens: 8, total_tokens: 28)
    sum = a + b

    sum.prompt_tokens.should eq(30)
    sum.completion_tokens.should eq(13)
    sum.total_tokens.should eq(43)

    # The operands stay unchanged.
    a.prompt_tokens.should eq(10)
    b.total_tokens.should eq(28)
  end

  it "treats nil as zero in +, staying nil only when both sides are nil" do
    a = Agent::Usage.new(prompt_tokens: 10)
    b = Agent::Usage.new(completion_tokens: 8)
    sum = a + b

    sum.prompt_tokens.should eq(10)
    sum.completion_tokens.should eq(8)
    sum.total_tokens.should be_nil

    # All-nil stays all-nil (preserved by the emit_nulls payload).
    empty = Agent::Usage.new + Agent::Usage.new
    empty.prompt_tokens.should be_nil
    empty.completion_tokens.should be_nil
    empty.total_tokens.should be_nil
  end
end

describe Agent::Message do
  it "marks a tool result as an error" do
    tc = Agent::ToolCall.new("call_1", "fn", "{}")
    ok = Agent::Message.tool_result(tc, "22°C")
    ok.is_error.should be_nil
    ok.error?.should be_false

    failed = Agent::Message.tool_result(tc, "no such city", is_error: true)
    failed.error?.should be_true
    Agent::Message.from_json(failed.to_json).error?.should be_true
    # the OpenAI wire format has no such field
    failed.to_request_body.has_key?("is_error").should be_false
  end

  it "keeps provider data through JSON and out of the request body" do
    data = JSON.parse(%({"blocks":[{"type":"thinking","signature":"abc"}]}))
    msg = Agent::Message.assistant("hi", provider_data: data)
    back = Agent::Message.from_json(msg.to_json)
    back.provider_data.should eq(data)
    back.should eq(msg)
    msg.to_request_body.has_key?("provider_data").should be_false
    Agent::Message.assistant("hi").should_not eq(msg)
  end
end

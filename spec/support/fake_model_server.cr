require "http/server"

# Scripted Anthropic streams and a fake server that answers each request
# with the next one, for the Anthropic provider specs.
def sse(events : Array(NamedTuple)) : String
  events.join { |e| "event: #{e[:type]}\ndata: #{e.to_json}\n\n" }
end

# Thinking (empty text, signed), a text block, one tool_use whose json
# arrives in two pieces, then the stop.
def anthropic_tool_call_stream(tool_id = "toolu_1", name = "get_finding", args = {"id" => "f1"}) : String
  json = args.to_json
  half = json.size // 2
  sse([
    {type: "message_start", message: {id: "msg_1", usage: {input_tokens: 40, cache_read_input_tokens: 1200, cache_creation_input_tokens: 0, output_tokens: 1}}},
    {type: "content_block_start", index: 0, content_block: {type: "thinking", thinking: ""}},
    {type: "content_block_delta", index: 0, delta: {type: "signature_delta", signature: "sig-abc"}},
    {type: "content_block_stop", index: 0},
    {type: "content_block_start", index: 1, content_block: {type: "text", text: ""}},
    {type: "content_block_delta", index: 1, delta: {type: "text_delta", text: "Looking at "}},
    {type: "content_block_delta", index: 1, delta: {type: "text_delta", text: "the finding."}},
    {type: "content_block_stop", index: 1},
    {type: "content_block_start", index: 2, content_block: {type: "tool_use", id: tool_id, name: name, input: {} of String => String}},
    {type: "content_block_delta", index: 2, delta: {type: "input_json_delta", partial_json: json[0, half]}},
    {type: "content_block_delta", index: 2, delta: {type: "input_json_delta", partial_json: json[half..]}},
    {type: "content_block_stop", index: 2},
    {type: "message_delta", delta: {stop_reason: "tool_use"}, usage: {output_tokens: 57}},
    {type: "message_stop"},
  ])
end

def anthropic_final_stream(text : String) : String
  sse([
    {type: "message_start", message: {id: "msg_2", usage: {input_tokens: 90, output_tokens: 1}}},
    {type: "content_block_start", index: 0, content_block: {type: "text", text: ""}},
    {type: "content_block_delta", index: 0, delta: {type: "text_delta", text: text}},
    {type: "content_block_stop", index: 0},
    {type: "message_delta", delta: {stop_reason: "end_turn"}, usage: {output_tokens: 12}},
    {type: "message_stop"},
  ])
end

class FakeModelServer
  getter requests = [] of NamedTuple(path: String, headers: HTTP::Headers, body: JSON::Any)
  getter url : String

  def initialize(@streams : Array(String))
    @server = HTTP::Server.new do |context|
      body = context.request.body.try(&.gets_to_end) || ""
      @requests << {path: context.request.path, headers: context.request.headers, body: JSON.parse(body)}
      stream = @streams.shift? || raise "the fake model ran out of scripted answers"
      context.response.content_type = "text/event-stream"
      context.response.print(stream)
    end
    address = @server.bind_tcp("127.0.0.1", 0)
    @url = "http://127.0.0.1:#{address.port}"
    spawn { @server.listen }
  end

  def close
    @server.close
  end
end

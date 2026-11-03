require "./spec_helper"

describe Agent::Config do
  it "defaults endpoint and model" do
    config = Agent::Config.new(api_key: "sk-test")
    config.api_endpoint.should eq("https://api.openai.com/v1")
    config.model.should eq("gpt-4o")
  end

  it "accepts custom values" do
    config = Agent::Config.new(
      api_key: "sk-custom",
      api_endpoint: "https://myapi.local/v1",
      model: "gpt-4o-mini",
      system_prompt: "Be helpful",
      max_tokens: 512,
      temperature: 0.7
    )
    config.api_key.should eq("sk-custom")
    config.api_endpoint.should eq("https://myapi.local/v1")
    config.model.should eq("gpt-4o-mini")
    config.system_prompt.should eq("Be helpful")
    config.max_tokens.should eq(512)
    config.temperature.should eq(0.7)
  end
end

describe Agent::Config, "background tasks" do
  it "defaults to four tasks and five minutes, taking seconds or a span" do
    config = Agent::Config.new
    config.max_background_tasks.should eq 4
    config.background_task_timeout.should eq 5.minutes
    Agent::Config.new(max_background_tasks: 0, background_task_timeout: 30).background_task_timeout.should eq 30.seconds
    Agent::Config.new(background_task_timeout: 2.hours).background_task_timeout.should eq 2.hours
    expect_raises(ArgumentError, "max_background_tasks") { Agent::Config.new(max_background_tasks: -1) }
    expect_raises(ArgumentError, "background_task_timeout") { Agent::Config.new(background_task_timeout: 0) }
    config.max_task_result_chars.should eq 30_000
    config.deliver_tasks_mid_turn?.should be_false
    Agent::Config.new(max_task_result_chars: nil).max_task_result_chars.should be_nil
    expect_raises(ArgumentError, "max_task_result_chars") { Agent::Config.new(max_task_result_chars: 0) }
  end
end

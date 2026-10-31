class Agent
  # How hard a reasoning model should think. Each provider maps it to its own
  # request field; a provider without one ignores it.
  enum ReasoningEffort
    Off
    Minimal
    Low
    Medium
    High
    Xhigh
    Max
  end

  # Settings for one #ask that override Config. Handlers can change them per
  # provider call through `TurnContext#options`.
  record RequestOptions, reasoning_effort : ReasoningEffort? = nil
end

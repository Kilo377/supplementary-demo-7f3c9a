from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CumulativeRewardResult:
    intent_text: str
    intent_satisfaction: float
    execution_steps: int
    cumulative_reward: float
    reason: str = ""
    final_intent_status: str = "active"
    prompt: str = ""
    raw_response: str = ""
    provider_name: str = "ollama"
    model: str | None = None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "intent_text": self.intent_text,
            "intent_satisfaction": self.intent_satisfaction,
            "execution_steps": self.execution_steps,
            "cumulative_reward": self.cumulative_reward,
            "reason": self.reason,
            "final_intent_status": self.final_intent_status,
            "error": self.error,
        }

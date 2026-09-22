from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardEvaluation:
    action_id: str
    reward: int
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "reward": self.reward,
            "reason": self.reason,
        }


@dataclass
class RewardEvaluationResult:
    evaluations: list[RewardEvaluation] = field(default_factory=list)
    prompt: str = ""
    raw_response: str = ""
    provider_name: str = "ollama"
    model: str | None = None
    error: str = ""

    def evaluation_by_id(self, action_id: str) -> RewardEvaluation | None:
        return next(
            (
                evaluation
                for evaluation in self.evaluations
                if evaluation.action_id == action_id
            ),
            None,
        )

    def to_dict(self) -> dict:
        return {
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "provider_name": self.provider_name,
            "model": self.model,
            "error": self.error,
        }

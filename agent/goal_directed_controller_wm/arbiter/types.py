from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArbitrationEntry:
    action_id: str
    action_text: str
    reward: int
    normalized_reward: float
    habit_strength: float
    model_based_contribution: float
    habitual_contribution: float
    decision_value: float

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "action_text": self.action_text,
            "reward": self.reward,
            "normalized_reward": self.normalized_reward,
            "habit_strength": self.habit_strength,
            "model_based_contribution": self.model_based_contribution,
            "habitual_contribution": self.habitual_contribution,
            "decision_value": self.decision_value,
        }


@dataclass(frozen=True)
class ActionSelectionResult:
    selected_action_id: str
    method: str
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "selected_action_id": self.selected_action_id,
            "method": self.method,
            "probabilities": dict(self.probabilities),
        }


@dataclass(frozen=True)
class ArbitrationResult:
    degree_of_model_based_control: float
    mode: str
    entries: tuple[ArbitrationEntry, ...]
    selection: ActionSelectionResult

    @property
    def selected_action_id(self) -> str:
        return self.selection.selected_action_id

    def entry_by_id(self, action_id: str) -> ArbitrationEntry | None:
        return next(
            (entry for entry in self.entries if entry.action_id == action_id),
            None,
        )

    def to_dict(self) -> dict:
        return {
            "degree_of_model_based_control": self.degree_of_model_based_control,
            "mode": self.mode,
            "entries": [entry.to_dict() for entry in self.entries],
            "selection": self.selection.to_dict(),
        }

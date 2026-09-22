from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreparedHabitualResponse:
    association_id: str
    response_key: str
    response_text: str
    source_dimension: str
    trigger_cue_ids: tuple[str, ...]
    habit_strength: float
    activation: float
    awareness_type: str
    estimated_duration: str = ""
    required_body_resources: tuple[str, ...] = ()
    cooldown_seconds: int = 0
    memory_sources: tuple[str, ...] = ()

    @property
    def usually_unconscious(self) -> bool:
        return self.awareness_type == "unconscious"


@dataclass
class HabitualActivationResult:
    tendencies: list[PreparedHabitualResponse] = field(default_factory=list)
    retrieval_status: str = "no_match"

    @property
    def strongest(self) -> PreparedHabitualResponse | None:
        return self.tendencies[0] if self.tendencies else None


@dataclass(frozen=True)
class GoalConflictResult:
    conflict: bool
    reason: str
    raw_response: str = ""


@dataclass(frozen=True)
class AwarenessResult:
    conscious: bool
    reason: str
    raw_response: str = ""


@dataclass(frozen=True)
class HabitualAwarenessGateResult:
    route: str
    response: PreparedHabitualResponse | None = None
    awareness: AwarenessResult | None = None
    awareness_source: str = "none"
    state_awareness_probability: float = 0.0
    state_awareness_sample: float | None = None
    state_awareness_triggered: bool = False
    state_awareness_factors: dict[str, int] = field(default_factory=dict)

    @property
    def starts_goal_directed_controller(self) -> bool:
        return self.route != "habitual_direct"

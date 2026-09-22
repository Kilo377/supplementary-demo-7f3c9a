from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StateTransitionPromptContext:
    agent_name: str
    action_text: str
    intent_text: str = ""
    spatial_belief_text: str = ""
    time_text: str = ""
    physical_state_text: str = ""
    internal_state_text: str = ""
    recent_experience_text: str = ""


@dataclass(frozen=True)
class TransitionOutcome:
    status: str = "failed"
    description: str = ""
    reason: str = ""


@dataclass(frozen=True)
class TransitionIntentSatisfaction:
    status: str = "not_satisfied"
    reason: str = ""

    @property
    def is_satisfied(self) -> bool:
        return self.status in {"satisfied", "approximately_satisfied"}


@dataclass(frozen=True)
class TransitionSelfState:
    area: str = ""
    near_element: str = ""
    posture: str = ""
    facing_or_gaze: str = ""
    holding: tuple[str, ...] = ()
    interacting_with: tuple[str, ...] = ()
    worn_items_change: str = ""
    body_surface_change: str = ""


@dataclass(frozen=True)
class TransitionInternalStateChanges:
    hunger: int = 0
    thirst: int = 0
    hygiene: int = 0
    stress: int = 0
    tension: int = 0
    fatigue: int = 0
    mental_change: str = ""


@dataclass(frozen=True)
class TransitionSpatialBeliefUpdate:
    element: str = ""
    state_change: str = ""


@dataclass(frozen=True)
class TransitionFailureContext:
    action_text: str = ""
    failed_module: str = ""
    error: str = ""
    world_feedback: str = ""

    def format_for_recovery_prompt(self, agent_name: str) -> str:
        name = str(agent_name or "Agent").strip()
        lines = [
            f"{name}刚才设想做“{self.action_text}”，但这个方案没有成功落实。"
        ]
        if self.world_feedback:
            lines.append(f"{name}预计会遇到：{self.world_feedback}")
        if self.failed_module:
            lines.append(f"问题发生在：{self.failed_module}。")
        if self.error:
            lines.append(f"具体原因是：{self.error}。")
        lines.append(
            "接下来不要机械重复这个失败动作。可以先解决缺失的前置条件，"
            "也可以选择当前环境真正支持的替代动作。"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class StateTransition:
    outcome: TransitionOutcome = field(default_factory=TransitionOutcome)
    intent_satisfaction: TransitionIntentSatisfaction = field(
        default_factory=TransitionIntentSatisfaction
    )
    elapsed_seconds: int = 0
    next_self_state: TransitionSelfState = field(default_factory=TransitionSelfState)
    internal_state_changes: TransitionInternalStateChanges = field(
        default_factory=TransitionInternalStateChanges
    )
    spatial_belief_updates: tuple[TransitionSpatialBeliefUpdate, ...] = ()
    expected_feedback: str = ""
    uncertainty: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateTransition":
        outcome = _dict_value(data, "transition_outcome")
        satisfaction = _dict_value(data, "intent_satisfaction")
        self_state = _dict_value(data, "next_self_state")
        internal = _dict_value(data, "internal_state_changes")
        updates = data.get("spatial_belief_updates", [])
        if not isinstance(updates, list):
            updates = []
        return cls(
            outcome=TransitionOutcome(
                status=_outcome_status(outcome.get("status")),
                description=_text(outcome.get("description")),
                reason=_text(outcome.get("reason")),
            ),
            intent_satisfaction=TransitionIntentSatisfaction(
                status=_satisfaction_status(satisfaction.get("status")),
                reason=_text(satisfaction.get("reason")),
            ),
            elapsed_seconds=max(0, _int_value(data.get("elapsed_seconds"))),
            next_self_state=TransitionSelfState(
                area=_text(self_state.get("area")),
                near_element=_text(self_state.get("near_element")),
                posture=_text(self_state.get("posture")),
                facing_or_gaze=_text(self_state.get("facing_or_gaze")),
                holding=_text_tuple(self_state.get("holding")),
                interacting_with=_text_tuple(self_state.get("interacting_with")),
                worn_items_change=_text(self_state.get("worn_items_change")),
                body_surface_change=_text(self_state.get("body_surface_change")),
            ),
            internal_state_changes=TransitionInternalStateChanges(
                hunger=_state_delta(internal.get("hunger")),
                thirst=_state_delta(internal.get("thirst")),
                hygiene=_state_delta(internal.get("hygiene")),
                stress=_state_delta(internal.get("stress")),
                tension=_state_delta(internal.get("tension")),
                fatigue=_state_delta(internal.get("fatigue")),
                mental_change=_text(internal.get("mental_change")),
            ),
            spatial_belief_updates=tuple(
                TransitionSpatialBeliefUpdate(
                    element=_text(item.get("element")),
                    state_change=_text(item.get("state_change")),
                )
                for item in updates
                if isinstance(item, dict)
                and (_text(item.get("element")) or _text(item.get("state_change")))
            ),
            expected_feedback=_text(data.get("expected_feedback")),
            uncertainty=_text(data.get("uncertainty")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_outcome": {
                "status": self.outcome.status,
                "description": self.outcome.description,
                "reason": self.outcome.reason,
            },
            "intent_satisfaction": {
                "status": self.intent_satisfaction.status,
                "reason": self.intent_satisfaction.reason,
            },
            "elapsed_seconds": self.elapsed_seconds,
            "next_self_state": {
                "area": self.next_self_state.area,
                "near_element": self.next_self_state.near_element,
                "posture": self.next_self_state.posture,
                "facing_or_gaze": self.next_self_state.facing_or_gaze,
                "holding": list(self.next_self_state.holding),
                "interacting_with": list(self.next_self_state.interacting_with),
                "worn_items_change": self.next_self_state.worn_items_change,
                "body_surface_change": self.next_self_state.body_surface_change,
            },
            "internal_state_changes": {
                "hunger": self.internal_state_changes.hunger,
                "thirst": self.internal_state_changes.thirst,
                "hygiene": self.internal_state_changes.hygiene,
                "stress": self.internal_state_changes.stress,
                "tension": self.internal_state_changes.tension,
                "fatigue": self.internal_state_changes.fatigue,
                "mental_change": self.internal_state_changes.mental_change,
            },
            "spatial_belief_updates": [
                {
                    "element": update.element,
                    "state_change": update.state_change,
                }
                for update in self.spatial_belief_updates
            ],
            "expected_feedback": self.expected_feedback,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class StateTransitionResult:
    transition: StateTransition | None
    prompt: str
    raw_response: str = ""
    provider_name: str = "ollama"
    model: str | None = None
    world_model_mode: str = "simple"
    failure_context: TransitionFailureContext | None = None
    error: str = ""


def _dict_value(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _text_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(text for item in value if (text := _text(item)))


def _int_value(value: Any) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _state_delta(value: Any) -> int:
    return max(-10, min(10, _int_value(value)))


def _outcome_status(value: Any) -> str:
    status = _text(value).lower()
    return status if status in {"success", "partial", "failed"} else "failed"


def _satisfaction_status(value: Any) -> str:
    status = _text(value).lower().replace(" ", "_")
    aliases = {
        "approximate": "approximately_satisfied",
        "approximately": "approximately_satisfied",
        "almost_satisfied": "approximately_satisfied",
        "partial": "approximately_satisfied",
        "unsatisfied": "not_satisfied",
        "not_satisfy": "not_satisfied",
    }
    status = aliases.get(status, status)
    if status in {"satisfied", "approximately_satisfied", "not_satisfied"}:
        return status
    return "not_satisfied"

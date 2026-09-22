from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent.desire.desire_state import DesireState
from agent.habitual_controller.cue_memory import CueMemoryRecord


@dataclass
class WorldAgentState:
    worn_items: list[str] | None = None
    posture: str = ""
    body_surface: str = ""
    facing: float | None = None
    field_of_view_degrees: float | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "WorldAgentState":
        data = data or {}
        worn_items = data.get("worn_items", None)
        return cls(
            worn_items=_string_list(worn_items) if worn_items is not None else None,
            posture=str(data.get("posture", "") or "").strip(),
            body_surface=str(data.get("body_surface", "") or "").strip(),
            facing=_optional_float(data.get("facing", None)),
            field_of_view_degrees=_optional_float(data.get("field_of_view_degrees", None)),
        )

    def to_dict(self) -> dict:
        data: dict = {}
        if self.worn_items is not None:
            data["worn_items"] = list(self.worn_items)
        if self.posture:
            data["posture"] = self.posture
        if self.body_surface:
            data["body_surface"] = self.body_surface
        if self.facing is not None:
            data["facing"] = self.facing
        if self.field_of_view_degrees is not None:
            data["field_of_view_degrees"] = self.field_of_view_degrees
        return data

    def apply_to_agent(self, agent, *, update_base: bool = True) -> None:
        state = self.to_dict()
        apply_world_agent_state = getattr(agent, "apply_world_agent_state", None)
        if callable(apply_world_agent_state):
            apply_world_agent_state(state, update_base=update_base)
            return
        for key, value in state.items():
            setattr(agent, key, value)


@dataclass
class Avatar:
    name: str
    personality: str
    biography: str
    desire_state: DesireState
    world_agent_state: WorldAgentState
    cue_memory_records: list[CueMemoryRecord]
    habit_strength_threshold: float
    degree_of_model_based_control: float = 0.5

    @classmethod
    def from_dict(cls, data: dict, *, fallback_name: str = "Agent") -> "Avatar":
        name = str(data.get("name", "") or fallback_name).strip()
        personality = str(data.get("personality", "") or "").strip()
        biography = _biography_text(data.get("biography", ""))
        desire_state = _desire_state_from_avatar_data(
            data.get("desire", {}) or {},
            agent_name=name,
        )
        return cls(
            name=name,
            personality=personality,
            biography=biography,
            desire_state=desire_state,
            world_agent_state=WorldAgentState.from_dict(data.get("world_agent_state", {}) or {}),
            cue_memory_records=_cue_memory_records(data.get("cue_memory", []), agent_name=name),
            habit_strength_threshold=float(data.get("habit_strength_threshold", 0.70) or 0.70),
            degree_of_model_based_control=_unit_interval(
                data.get("degree_of_model_based_control", 0.5)
            ),
        )


def load_avatar(
    path: str | Path,
    *,
    fallback_name: str = "Agent",
    override_name: str = "",
) -> Avatar:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Avatar file must contain a JSON object.")
    if override_name.strip():
        data["name"] = override_name.strip()
    return Avatar.from_dict(data, fallback_name=fallback_name)


def _desire_state_from_avatar_data(data: dict, *, agent_name: str) -> DesireState:
    base = DesireState.for_agent(agent_name).to_dict()
    physiological_state = dict(base.get("physiological_state", {}))
    if isinstance(data.get("physiological_state"), dict):
        physiological_state.update(data["physiological_state"])
    internal_state = dict(base.get("internal_state", {}))
    if isinstance(data.get("internal_state"), dict):
        internal_state.update(data["internal_state"])

    mental = str(data.get("mental", "") or base.get("mental", "")).format(agent_name=agent_name)
    work_goal = data.get("work_goal", base.get("work_goal", []))
    return DesireState.from_dict(
        {
            "physiological_state": physiological_state,
            "internal_state": internal_state,
            "mental": mental,
            "work_goal": _format_work_goals(work_goal, agent_name=agent_name),
        }
    )


def _format_work_goals(items, *, agent_name: str) -> list:
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        if isinstance(item, dict):
            formatted = dict(item)
            formatted["text"] = str(formatted.get("text", "")).format(agent_name=agent_name)
            result.append(formatted)
        else:
            result.append(str(item).format(agent_name=agent_name))
    return result


def _biography_text(value) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _unit_interval(value) -> float:
    return min(1.0, max(0.0, float(value)))


def _cue_memory_records(items, *, agent_name: str) -> list[CueMemoryRecord]:
    if not isinstance(items, list):
        return []
    return [
        CueMemoryRecord.from_dict(item, agent_name=agent_name)
        for item in items
        if isinstance(item, dict)
    ]

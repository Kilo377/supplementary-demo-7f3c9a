from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

from contextual_world.world_old.spatial_rules.agent_vision import bounds_intersect_view
from contextual_world.world_old.spatial_rules.physics_types import AABB

from .types import DemonstratedActionObservation, DemonstratedCueFeature


_REPLAY_DATA = re.compile(
    r'<script id="replay-data" type="application/json">(.*?)</script>',
    re.DOTALL,
)
_FAILURE_MARKERS = (
    "Unable to", "Failed", "error", "Not found", "Did not receive stable", "Not yet implemented",
    "failed", "error", "cannot",
)


def load_replay_payload(path: str | Path) -> dict:
    source = Path(path)
    match = _REPLAY_DATA.search(source.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Replay data was not found in {source}.")
    return json.loads(match.group(1))


def observations_from_replay(
    payload: dict,
    *,
    source_run_id: str,
    initial_posture: str = "standing",
    initial_worn_items: tuple[str, ...] = (),
    max_visual_features: int = 24,
) -> list[DemonstratedActionObservation]:
    areas = list(payload.get("home", {}).get("areas", []) or [])
    steps = list(payload.get("steps", []) or [])
    states = _initial_element_states(areas, steps)
    elements = {
        str(element.get("id", "")): element
        for area in areas
        for element in list(area.get("elements", []) or [])
        if str(element.get("id", "")).strip()
    }
    element_areas = {
        str(element.get("id", "")): str(area.get("id", ""))
        for area in areas
        for element in list(area.get("elements", []) or [])
    }
    previous_actor = {
        "facing": float(payload.get("agent", {}).get("facing", 90.0) or 90.0),
        "field_of_view_degrees": float(
            payload.get("agent", {}).get("field_of_view_degrees", 160.0) or 160.0
        ),
        "posture": initial_posture,
        "worn_items": list(initial_worn_items),
    }
    previous_successful_action = ""
    observations: list[DemonstratedActionObservation] = []

    for step in steps:
        position = _point(step.get("from_position"))
        area = _area_for_point(areas, position)
        action = str(step.get("proposal", "") or "").strip()
        feedback = str(step.get("feedback", "") or "").strip()
        successful = _successful_semantic_step(step, action=action, feedback=feedback)
        if successful and str(step.get("control_source", "")) in {
            "goal_directed", "world_model",
        }:
            features = _features_before_step(
                area=area,
                position=position,
                actor=previous_actor,
                elements=elements,
                element_areas=element_areas,
                states=states,
                target_ids={str(item) for item in step.get("target_ids", []) or []},
                previous_successful_action=previous_successful_action,
                max_visual_features=max_visual_features,
            )
            if features:
                observations.append(
                    DemonstratedActionObservation(
                        observation_id=(
                            f"{_safe_id(source_run_id)}_step_"
                            f"{int(step.get('step_id', 0) or 0)}"
                        ),
                        step_id=int(step.get("step_id", 0) or 0),
                        action_text=action,
                        result_text=feedback,
                        area_name=str((area or {}).get("name", "")),
                        estimated_duration=str(step.get("estimated_duration", "") or ""),
                        previous_action_text=previous_successful_action,
                        cue_features=tuple(features),
                        source_run_id=source_run_id,
                    )
                )

        if successful:
            previous_successful_action = action
        _apply_changes(states, list(step.get("changes", []) or []), value_key="new")
        actor = step.get("actor_state", {}) or {}
        if isinstance(actor, dict):
            previous_actor = {**previous_actor, **actor}
    return observations


def _initial_element_states(areas: list[dict], steps: list[dict]) -> dict[str, dict]:
    states = {
        str(element.get("id", "")): {
            "physical_status": str(element.get("physical_status", "regular") or "regular"),
            "state_details": deepcopy(element.get("state_details", {}) or {}),
        }
        for area in areas
        for element in list(area.get("elements", []) or [])
    }
    for step in reversed(steps):
        _apply_changes(states, reversed(list(step.get("changes", []) or [])), value_key="old")
    return states


def _apply_changes(states: dict[str, dict], changes, *, value_key: str) -> None:
    for change in changes:
        element_id = str(change.get("element_id", "") or "")
        if not element_id:
            continue
        state = states.setdefault(
            element_id, {"physical_status": "regular", "state_details": {}}
        )
        field = str(change.get("field", "") or "")
        value = change.get(value_key, "")
        if field == "physical_status":
            state["physical_status"] = str(value or "regular")
        elif field.startswith("state_details."):
            key = field.split(".", 1)[1]
            if str(value) in {"", "unset"}:
                state.setdefault("state_details", {}).pop(key, None)
            else:
                state.setdefault("state_details", {})[key] = str(value)


def _features_before_step(
    *,
    area: dict | None,
    position: tuple[float, float],
    actor: dict,
    elements: dict[str, dict],
    element_areas: dict[str, str],
    states: dict[str, dict],
    target_ids: set[str],
    previous_successful_action: str,
    max_visual_features: int,
) -> list[DemonstratedCueFeature]:
    features: list[DemonstratedCueFeature] = []
    area_id = str((area or {}).get("id", "") or "")
    if area_id:
        features.append(DemonstratedCueFeature(
            f"location:{area_id}", str((area or {}).get("name", area_id)), "location"
        ))
    posture = str(actor.get("posture", "") or "")
    if posture:
        features.append(DemonstratedCueFeature(f"posture:{posture}", posture, "posture"))
    for item in list(actor.get("worn_items", []) or []):
        features.append(DemonstratedCueFeature(f"wearing:{item}", str(item), "wearing"))
    if previous_successful_action:
        features.append(DemonstratedCueFeature(
            f"completed:{previous_successful_action}",
            previous_successful_action,
            "completed",
            "recent",
        ))

    visible: list[tuple[int, DemonstratedCueFeature]] = []
    facing = float(actor.get("facing", 90.0) or 90.0)
    field_of_view = float(actor.get("field_of_view_degrees", 160.0) or 160.0)
    for element_id, element in elements.items():
        if area_id and element_areas.get(element_id) != area_id:
            continue
        center = _point(element.get("center"))
        size = _point(element.get("size"))
        bounds = AABB(
            x_min=center[0] - size[0] / 2,
            y_min=center[1] - size[1] / 2,
            x_max=center[0] + size[0] / 2,
            y_max=center[1] + size[1] / 2,
        )
        if not bounds_intersect_view(
            observer=position,
            facing_degrees=facing,
            field_of_view_degrees=field_of_view,
            bounds=bounds,
        ):
            continue
        state = states.get(element_id, {})
        name = str(element.get("name", element_id))
        priority = 0 if element_id in target_ids else 1
        visible.append((priority, DemonstratedCueFeature(
            f"visible:{element_id}", name, "visual_object"
        )))
        physical = str(state.get("physical_status", "regular") or "regular")
        if physical != "regular":
            visible.append((priority, DemonstratedCueFeature(
                f"visible_state:{element_id}:physical:{physical}",
                f"{name}:{physical}",
                "visual_state",
            )))
        for key, value in sorted((state.get("state_details", {}) or {}).items()):
            visible.append((priority, DemonstratedCueFeature(
                f"visible_state:{element_id}:{key}:{value}",
                f"{name}:{key}={value}",
                "visual_state",
            )))
    visible.sort(key=lambda item: item[0])
    features.extend(item[1] for item in visible[:max(0, max_visual_features)])
    return features


def _successful_semantic_step(step: dict, *, action: str, feedback: str) -> bool:
    if not action or not bool(step.get("counts_as_intent_action", True)):
        return False
    if str(step.get("execution_kind", "")) in {
        "action_error", "move_precondition", "chat_todo",
    }:
        return False
    lowered = feedback.lower()
    return not any(marker in lowered or marker in feedback for marker in _FAILURE_MARKERS)


def _area_for_point(areas: list[dict], point: tuple[float, float]) -> dict | None:
    x, y = point
    for area in areas:
        bounds = list(area.get("bounds", []) or [])
        if len(bounds) == 4 and bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
            return area
    return None


def _point(value) -> tuple[float, float]:
    items = list(value or [0.0, 0.0])
    return float(items[0]), float(items[1])


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")

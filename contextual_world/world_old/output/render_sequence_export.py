from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.interaction_engine import EngineStepResult, WorldStateChange
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from contextual_world.structure.scene_schema import Element, Home


SCHEMA_VERSION = "render_sequence.v1"


def build_render_sequence_export(
    home: Home,
    *,
    initial_position: tuple[float, float],
    results: list[EngineStepResult],
) -> dict:
    engine = PhysicsEngine(home)
    sequence: list[dict] = []
    previous_position = initial_position

    for result in results:
        from_point = _movement_point(engine, previous_position)
        to_point = _movement_point(engine, result.final_position)
        moved = _positions_differ(previous_position, result.final_position)

        if moved:
            sequence.append(
                {
                    "sequence_id": len(sequence) + 1,
                    "turn_step_id": result.step_id,
                    "turn_action_index": result.intent_action_index,
                    "type": "move",
                    "reference_human_action": result.action_proposal_text,
                    "intent_text": result.intent_text,
                    "from": from_point,
                    "to": to_point,
                    "current_element": _element_reference_payload(from_point.get("anchor_element")),
                    "destination_element": _element_reference_payload(to_point.get("anchor_element")),
                    "path_hint": {
                        "from_position": list(previous_position),
                        "to_position": list(result.final_position),
                    },
                }
            )

        if _should_emit_action_segment(result, moved=moved):
            sequence.append(
                {
                    "sequence_id": len(sequence) + 1,
                    "turn_step_id": result.step_id,
                    "turn_action_index": result.intent_action_index,
                    "type": _action_export_type(result),
                    "reference_human_action": result.action_proposal_text,
                    "intent_text": result.intent_text,
                    "intent_status": result.intent_status,
                    "action_proposal": result.action_proposal_text,
                    "world_feedback": _world_summary(result) or result.execution_narration or result.action_text,
                    "motion": _motion_payload(result),
                    "element_changes": _element_changes_payload(result),
                    "temporary_element_changes": list(result.temporary_element_changes or []),
                    "graph_transition": _graph_transition_payload(result),
                    "referenced_elements": _referenced_elements_payload(result),
                    "duration": result.estimated_duration,
                    "position": to_point,
                }
            )
        previous_position = result.final_position

    return {
        "schema_version": SCHEMA_VERSION,
        "scene": {
            "id": home.node_id,
            "name": home.name,
            "default_agent_start": list(home.default_agent_start or initial_position),
        },
        "agent": {
            "name": results[0].agent_name if results else "Agent",
            "initial_position": list(initial_position),
        },
        "sequence": sequence,
        "source_turns": [_turn_payload(result) for result in results],
    }


def write_render_sequence_export(
    home: Home,
    *,
    initial_position: tuple[float, float],
    results: list[EngineStepResult],
    output_path: str | Path,
) -> Path:
    payload = build_render_sequence_export(
        home,
        initial_position=initial_position,
        results=results,
    )
    path = Path(output_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _turn_payload(result: EngineStepResult) -> dict:
    return {
        "step_id": result.step_id,
        "action_index": result.intent_action_index,
        "intent_text": result.intent_text,
        "intent_status": result.intent_status,
        "action_type": result.action_type,
        "execution_kind": getattr(result, "execution_kind", "") or result.action_type,
        "action_proposal": result.action_proposal_text,
        "world_feedback": _world_summary(result) or result.execution_narration or result.action_text,
        "final_position": list(result.final_position),
    }


def _should_emit_action_segment(result: EngineStepResult, *, moved: bool) -> bool:
    if getattr(result, "execution_kind", "") == "move":
        return False
    if result.action_type == "move_to_area" and moved and not _motion_text(result):
        return False
    return True


def _action_export_type(result: EngineStepResult) -> str:
    if result.action_type == "wait":
        return "wait"
    return "action"


def _motion_payload(result: EngineStepResult) -> dict:
    actor_state = dict(result.actor_state or {})
    actor_update = dict(result.actor_state_update or {})
    motion_text = _motion_text(result)
    return {
        "text_to_motion": motion_text,
        "interaction_method": actor_update.get("interaction_method")
        or actor_state.get("interaction_method")
        or "",
        "posture": actor_update.get("posture") or actor_state.get("posture") or "",
        "gaze_target": actor_update.get("gaze_target") or actor_state.get("gaze_target") or "",
        "body_surface": actor_update.get("body_surface") or actor_state.get("body_surface") or "",
        "worn_items": actor_update.get("worn_items", actor_state.get("worn_items", [])) or [],
        "interaction_elements": actor_update.get(
            "interaction_elements",
            actor_state.get("interaction_elements", []),
        )
        or [],
    }


def _motion_text(result: EngineStepResult) -> str:
    actor_update = dict(result.actor_state_update or {})
    actor_state = dict(result.actor_state or {})
    return str(
        actor_update.get("text_to_motion_description")
        or actor_state.get("text_to_motion_description")
        or ""
    ).strip()


def _element_changes_payload(result: EngineStepResult) -> list[dict]:
    return [_world_change_payload(change) for change in result.world_changes]


def _world_change_payload(change: WorldStateChange) -> dict:
    fields = []
    if change.old_physical_status != change.new_physical_status:
        fields.append(
            {
                "field": "physical_status",
                "old": change.old_physical_status,
                "new": change.new_physical_status,
            }
        )
    if change.old_evolution_status != change.new_evolution_status:
        fields.append(
            {
                "field": "evolution_status",
                "old": change.old_evolution_status,
                "new": change.new_evolution_status,
            }
        )
    if change.old_interaction_status != change.new_interaction_status:
        fields.append(
            {
                "field": "interaction_status",
                "old": change.old_interaction_status,
                "new": change.new_interaction_status,
            }
        )

    detail_keys = set(change.old_state_details) | set(change.new_state_details)
    for key in sorted(detail_keys):
        old = change.old_state_details.get(key, "")
        new = change.new_state_details.get(key, "")
        if old != new:
            fields.append(
                {
                    "field": f"state_details.{key}",
                    "old": old or "unset",
                    "new": new or "unset",
                }
            )

    return {
        "area_id": change.area_id,
        "element_id": change.element_id,
        "element_name": change.element_name,
        "fields": fields,
    }


def _graph_transition_payload(result: EngineStepResult) -> dict:
    feedback = result.environment_feedback or {}
    if not isinstance(feedback, dict):
        return {}
    report = feedback.get("graph_transition_report") or {}
    return dict(report) if isinstance(report, dict) else {}


def _referenced_elements_payload(result: EngineStepResult) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()

    for item in _motion_payload(result).get("interaction_elements", []) or []:
        if isinstance(item, dict):
            element_id = str(item.get("element_id") or item.get("id") or "").strip()
            name = str(item.get("element_name") or item.get("name") or element_id).strip()
        else:
            name = str(item).strip()
            element_id = name if _looks_like_element_id(name) else ""
        key = element_id or name
        if key and key not in seen:
            seen.add(key)
            items.append({"element_id": element_id, "element_name": name})

    for change in result.world_changes:
        if change.element_id not in seen:
            seen.add(change.element_id)
            items.append({"element_id": change.element_id, "element_name": change.element_name})

    transition = _graph_transition_payload(result)
    for section in ("created_temporary_nodes", "updated_temporary_nodes"):
        for item in transition.get(section, []) or []:
            node_id = str(item.get("node_id", "") or "").strip()
            name = str(item.get("name", "") or node_id).strip()
            if node_id and node_id not in seen:
                seen.add(node_id)
                items.append({"element_id": node_id, "element_name": name, "temporary": True})

    return items


def _looks_like_element_id(value: str) -> bool:
    return bool(value) and ("_" in value or value.startswith("temporary"))


def _world_summary(result: EngineStepResult) -> str:
    feedback = result.environment_feedback or {}
    if not isinstance(feedback, dict):
        return ""
    return str(feedback.get("perception_summary", "") or "").strip()


def _movement_point(engine: PhysicsEngine, position: tuple[float, float]) -> dict:
    area = engine.get_area_for_point(*position)
    anchor = _nearest_anchor(engine, position)
    return {
        "position": [round(position[0], 4), round(position[1], 4)],
        "area_id": area.node_id if area is not None else "",
        "area_name": area.name if area is not None else "",
        "anchor_element": _anchor_payload(engine, anchor),
    }


def _nearest_anchor(
    engine: PhysicsEngine,
    position: tuple[float, float],
    *,
    radius: float = 1.4,
) -> Element | None:
    nearby = engine.get_nearby_elements(center=position, radius=radius)
    if not nearby:
        return None
    current_area = engine.get_area_for_point(*position)
    if current_area is not None:
        same_area = [item for item in nearby if item.area_id == current_area.node_id]
        if same_area:
            nearby = same_area
    nearby.sort(
        key=lambda item: (
            (item.element.center[0] - position[0]) ** 2
            + (item.element.center[1] - position[1]) ** 2
        )
    )
    return nearby[0].element


def _anchor_payload(engine: PhysicsEngine, element: Element | None) -> dict:
    if element is None:
        return {}
    area_id = ""
    area_name = ""
    for area in engine.home.areas:
        if area.find_element(element.node_id) is not None:
            area_id = area.node_id
            area_name = area.name
            break
    return {
        "element_id": element.node_id,
        "element_name": element.name,
        "area_id": area_id,
        "area_name": area_name,
    }


def _element_reference_payload(value: Any) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _positions_differ(
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    epsilon: float = 1e-6,
) -> bool:
    return abs(a[0] - b[0]) > epsilon or abs(a[1] - b[1]) > epsilon

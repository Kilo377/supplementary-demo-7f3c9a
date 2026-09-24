from __future__ import annotations

from agent.habitual_controller.cue_extraction import CueExtractionResult
from agent.perceive import PerceiveResult

from .types import PreparedHabitualResponse


def habitual_trigger_text(
    response: PreparedHabitualResponse,
    perception: PerceiveResult,
    context: CueExtractionResult,
) -> str:
    trigger_ids = set(response.trigger_cue_ids)
    lines = []
    features = context.features
    if features:
        items = [
            (feature.feature_id, feature.feature_type, feature.value, feature.text)
            for feature in features
        ]
    else:
        items = [
            (item.cue.cue_id, item.cue.cue_type, item.cue.value, item.cue.text)
            for item in context.active_cues
        ]
    for feature_id, feature_type, value, feature_text in items:
        if feature_id not in trigger_ids:
            continue
        if feature_type == "entered_area" and value == perception.area_id:
            text = f"{perception.agent_name} just entered {perception.area_name}"
        else:
            text = feature_text.strip() or value.strip() or feature_id
        text = _strip_sentence_end(text)
        if text and text not in lines:
            lines.append(text)
    return "；".join(lines)


def habitual_current_state_text(
    perception: PerceiveResult,
    context: CueExtractionResult,
) -> str:
    lines = []
    features = context.features
    if features:
        items = [
            (feature.feature_type, feature.value, feature.text)
            for feature in features
            if feature.temporal_state in {"onset", "maintained", "changed"}
        ]
    else:
        items = [(cue.cue_type, cue.value, cue.text) for cue in context.frame.state_cues]
    for feature_type, value, feature_text in items:
        if feature_type not in {"posture", "feeling"}:
            continue
        if feature_type == "posture":
            text = {
                "standing": f"{perception.agent_name} is standing",
                "sitting": f"{perception.agent_name} is sitting",
                "lying": f"{perception.agent_name} is lying down",
                "crouching": f"{perception.agent_name} is squatting",
                "walking": f"{perception.agent_name} is walking around",
            }.get(value, feature_text.strip() or value.strip())
        else:
            text = feature_text.strip() or value.strip()
        text = _strip_sentence_end(text)
        if text and text not in lines:
            lines.append(text)
    physical = perception.self_input.physical
    if physical.body_surface and physical.body_surface != "dry_clean":
        lines.append(f"The surface state of {perception.agent_name}'s body is {physical.body_surface}")
    return "；".join(lines)


def _strip_sentence_end(text: str) -> str:
    return str(text or "").strip().rstrip("。！？.!?").strip()

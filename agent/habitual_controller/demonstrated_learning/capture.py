from __future__ import annotations

from .types import DemonstratedActionObservation, DemonstratedCueFeature


def capture_demonstrated_action_observation(
    result,
    cue_extraction,
    *,
    previous_successful_action: str = "",
    area_name: str = "",
) -> DemonstratedActionObservation | None:
    if not _is_successful_goal_directed_action(result):
        return None
    features = []
    for feature in list(getattr(cue_extraction, "features", []) or []):
        temporal_state = str(getattr(feature, "temporal_state", "") or "")
        if temporal_state == "offset":
            continue
        feature_id = str(getattr(feature, "feature_id", "") or "").strip()
        if not feature_id:
            continue
        features.append(
            DemonstratedCueFeature(
                feature_id=feature_id,
                text=str(
                    getattr(feature, "text", "")
                    or getattr(feature, "value", "")
                    or feature_id
                ).strip(),
                feature_type=str(getattr(feature, "feature_type", "") or ""),
                temporal_state=temporal_state or "current",
            )
        )
    if not features:
        return None
    perceived_area = next(
        (
            feature.text
            for feature in features
            if feature.feature_type == "location"
        ),
        "",
    )
    step_id = int(getattr(result, "step_id", 0) or 0)
    action_text = str(getattr(result, "action_proposal_text", "") or "").strip()
    return DemonstratedActionObservation(
        observation_id=f"step_{step_id}",
        step_id=step_id,
        action_text=action_text,
        result_text=str(
            getattr(result, "execution_narration", "")
            or getattr(result, "action_text", "")
            or ""
        ).strip(),
        area_name=str(perceived_area or area_name or "").strip(),
        estimated_duration=str(getattr(result, "estimated_duration", "") or "").strip(),
        previous_action_text=str(previous_successful_action or "").strip(),
        cue_features=tuple(features),
    )


def successful_semantic_action_text(result) -> str:
    if not _is_successful_semantic_action(result):
        return ""
    return str(getattr(result, "action_proposal_text", "") or "").strip()


def _is_successful_goal_directed_action(result) -> bool:
    return (
        _is_successful_semantic_action(result)
        and not str(getattr(result, "habitual_response_key", "") or "").strip()
    )


def _is_successful_semantic_action(result) -> bool:
    if not bool(getattr(result, "counts_as_intent_action", True)):
        return False
    if str(getattr(result, "execution_kind", "") or "") in {
        "action_error",
        "move_precondition",
        "chat_todo",
    }:
        return False
    if not str(getattr(result, "action_proposal_text", "") or "").strip():
        return False
    debug = getattr(result, "execution_debug", {}) or {}
    if isinstance(debug, dict) and (debug.get("failed_module") or debug.get("error")):
        return False
    return True

from __future__ import annotations

from datetime import datetime

from agent.perceive import PerceiveResult

from .prompt import build_context_cue_prompt
from .types import (
    ContextCue,
    ContextCueBuffer,
    ContextCueFrame,
    CueExtractionResult,
    PreviousExecutionInput,
)


EVENT_TTL_SECONDS = 300


def extract_cues(
    agent,
    perception: PerceiveResult,
    *,
    previous_execution: PreviousExecutionInput | None = None,
    cue_buffer: ContextCueBuffer | None = None,
) -> CueExtractionResult:
    now = _current_datetime(agent)
    state_cues = _state_cues(agent, perception, now=now)
    event_cues = _event_cues(perception, previous_execution, now=now)
    preceding_responses = []
    if previous_execution is not None and previous_execution.succeeded and previous_execution.response_key.strip():
        preceding_responses.append(previous_execution.response_key.strip())
    frame = ContextCueFrame(
        agent_name=perception.agent_name,
        timestamp=now,
        state_cues=state_cues,
        event_cues=event_cues,
        preceding_responses=preceding_responses,
    )
    buffer = cue_buffer or ContextCueBuffer()
    active_cues = buffer.update(frame)
    prompt_text = build_context_cue_prompt(
        frame,
        area_name=perception.area_name,
        posture_text=_posture_text(perception.self_input.physical.posture),
        worn_items=list(perception.self_input.physical.worn_items),
        visual_text=perception.visual_prompt_text(),
        feeling_texts=[
            (item.source, item.text)
            for item in perception.self_input.psychological.desire_feelings
        ],
        time_text=_time_text(agent),
        previous_execution=previous_execution,
    )
    return CueExtractionResult(
        frame=frame,
        active_cues=active_cues,
        prompt_text=prompt_text,
        features=list(buffer.latest_features),
    )


def previous_execution_from_memory_episode(episode) -> PreviousExecutionInput | None:
    if episode is None:
        return None
    response_text = str(getattr(episode, "intended_action", "") or "").strip()
    result_text = str(getattr(episode, "experienced_result", "") or "").strip()
    if not response_text and not result_text:
        return None
    succeeded = not _looks_failed(result_text)
    return PreviousExecutionInput(
        response_key=response_text,
        response_text=response_text if succeeded else "",
        result_text=result_text,
        succeeded=succeeded,
        area_before=str(getattr(episode, "area_id", "") or ""),
        area_after=str(getattr(episode, "area_id", "") or ""),
        interacted_element_ids=tuple(getattr(episode, "interacted_element_ids", []) or []),
        interacted_element_names=tuple(getattr(episode, "interacted_element_names", []) or []),
    )


def _state_cues(agent, perception: PerceiveResult, *, now: datetime) -> list[ContextCue]:
    cues = [
        _cue(
            f"location:{perception.area_id}",
            "location",
            perception.area_id,
            "world_position",
            f"{perception.agent_name} is in {perception.area_name}",
            now,
            persistence="while_present",
        ),
        _cue(
            f"time_period:{_time_period(now.hour)}",
            "time_period",
            _time_period(now.hour),
            "time_belief",
            _time_period(now.hour),
            now,
            persistence="time_period",
        ),
    ]
    physical = perception.self_input.physical
    if physical.posture:
        cues.append(_cue(f"posture:{physical.posture}", "posture", physical.posture, "physical_self", physical.posture, now))
    for item in physical.worn_items:
        cues.append(_cue(f"wearing:{item}", "wearing", item, "physical_self", item, now, persistence="while_present"))
    for element in perception.visual_input.visible_elements:
        cues.append(_cue(f"visible:{element.element_id}", "visual_object", element.element_id, "visual_input", element.name, now, persistence="while_visible"))
        semantic_type = element.semantic_type.strip()
        if semantic_type:
            cues.append(_cue(f"visible_type:{semantic_type}", "visual_object_type", semantic_type, "visual_input", element.name, now, persistence="while_visible"))
        if element.physical_status and element.physical_status != "regular":
            cues.append(_cue(f"visible_state:{element.element_id}:physical:{element.physical_status}", "visual_state", element.physical_status, "visual_input", f"{element.name}:{element.physical_status}", now, persistence="while_visible"))
        for key, value in sorted(element.state_details.items()):
            cues.append(_cue(f"visible_state:{element.element_id}:{key}:{value}", "visual_state", value, "visual_input", f"{element.name}:{key}={value}", now, persistence="while_visible"))
            if semantic_type:
                cues.append(_cue(f"visible_type_state:{semantic_type}:{key}:{value}", "visual_type_state", value, "visual_input", f"{semantic_type}:{key}={value}", now, persistence="while_visible"))
    for feeling in perception.self_input.psychological.desire_feelings:
        if feeling.source == "work_goal":
            continue
        semantic_keys = feeling.semantic_keys or [""]
        for semantic_key in semantic_keys:
            cue_source = _feeling_source(feeling.source, semantic_key)
            cues.append(
                _cue(
                    f"feeling:{cue_source}:{feeling.intensity}",
                    "feeling",
                    feeling.intensity,
                    cue_source,
                    _context_feeling_text(feeling.source, feeling.text),
                    now,
                    persistence="current",
                )
            )
    return cues


def _event_cues(
    perception: PerceiveResult,
    previous_execution: PreviousExecutionInput | None,
    *,
    now: datetime,
) -> list[ContextCue]:
    cues: list[ContextCue] = []
    if previous_execution is None:
        return cues
    response_key = previous_execution.response_key.strip()
    if response_key:
        prefix = "completed" if previous_execution.succeeded else "failed"
        text = previous_execution.response_text or previous_execution.result_text or response_key
        cues.append(_cue(f"{prefix}:{response_key}", prefix, response_key, "previous_execution", text, now, persistence="decay", ttl_seconds=EVENT_TTL_SECONDS))
    if previous_execution.area_after and previous_execution.area_after != previous_execution.area_before:
        cues.append(_cue(f"entered:{previous_execution.area_after}", "entered_area", previous_execution.area_after, "previous_execution", previous_execution.area_after, now, persistence="decay", ttl_seconds=EVENT_TTL_SECONDS))
    return cues


def _cue(
    cue_id: str,
    cue_type: str,
    value: str,
    source: str,
    text: str,
    now: datetime,
    *,
    persistence: str = "current",
    ttl_seconds: int = 0,
) -> ContextCue:
    return ContextCue(
        cue_id=cue_id,
        cue_type=cue_type,
        value=str(value),
        source=source,
        text=str(text),
        persistence=persistence,
        occurred_at=now,
        ttl_seconds=ttl_seconds,
    )


def _current_datetime(agent) -> datetime:
    return getattr(getattr(agent, "time_belief", None), "current_datetime", None) or datetime.now()


def _time_text(agent) -> str:
    formatter = getattr(getattr(agent, "time_belief", None), "format_short_label", None)
    return str(formatter() or "").strip() if callable(formatter) else ""


def _time_period(hour: int) -> str:
    if 5 <= hour < 8:
        return "early morning"
    if 8 <= hour < 12:
        return "AM"
    if 12 <= hour < 14:
        return "noon"
    if 14 <= hour < 18:
        return "PM"
    if 18 <= hour < 23:
        return "evening"
    return "late night"


def _posture_text(posture: str) -> str:
    return {"standing": "standing", "sitting": "sitting", "lying": "lying down", "crouching": "crouching", "walking": "walking around"}.get(posture, posture)


def _looks_failed(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered or marker in text for marker in ["Unable to", "Failed", "error", "not completed", "Cannot", "failed", "error", "cannot"])


def _context_feeling_text(source: str, text: str) -> str:
    cleaned = str(text or "").strip()
    if source == "mental":
        return cleaned.split("；", 1)[0].strip()
    return cleaned


def _feeling_source(source: str, semantic_key: str) -> str:
    cleaned_key = str(semantic_key or "").strip()
    if source == "mental" and cleaned_key:
        return f"mental:{cleaned_key}"
    return source

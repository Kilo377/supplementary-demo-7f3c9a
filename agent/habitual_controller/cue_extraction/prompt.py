from __future__ import annotations

from .types import ContextCueFrame, PreviousExecutionInput


def build_context_cue_prompt(
    frame: ContextCueFrame,
    *,
    area_name: str,
    posture_text: str,
    worn_items: list[str],
    visual_text: str,
    feeling_texts: list[tuple[str, str]],
    time_text: str,
    previous_execution: PreviousExecutionInput | None,
) -> str:
    name = frame.agent_name
    parts = [f"At this moment, {name} is in {area_name} {posture_text}."]
    if worn_items:
        parts.append(f"{name} is wearing {'、'.join(worn_items)}.")
    if visual_text.strip():
        parts.append(visual_text.strip())
    parts.extend(_context_feeling_text(source, text) for source, text in feeling_texts if source != "work_goal" and text.strip())
    if previous_execution is not None and previous_execution.response_text.strip():
        if previous_execution.succeeded:
            parts.append(f"Just now, {previous_execution.response_text.strip()}.")
        elif previous_execution.result_text.strip():
            parts.append(f"Just now, {previous_execution.result_text.strip()}.")
    if time_text:
        parts.append(f"It is now {time_text}.")
    return " ".join(_clean_sentence(part) for part in parts if part.strip())


def _clean_sentence(text: str) -> str:
    cleaned = text.strip().replace("。。", "。")
    if cleaned and cleaned[-1] not in "。！？.!?":
        cleaned += "。"
    return cleaned


def _context_feeling_text(source: str, text: str) -> str:
    cleaned = text.strip()
    if source == "mental":
        cleaned = cleaned.split("；", 1)[0].strip()
    return cleaned

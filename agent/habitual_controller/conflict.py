from __future__ import annotations

import json

from llm.api_manager import APIManager

from .conflict_prompt import build_goal_conflict_prompt
from .types import GoalConflictResult, PreparedHabitualResponse


def judge_goal_conflict(
    *,
    intent_text: str,
    habitual_response: PreparedHabitualResponse,
    provider_name: str = "ollama",
    model: str | None = None,
) -> GoalConflictResult:
    prompt = build_goal_conflict_prompt(
        intent_text=intent_text,
        habitual_response=habitual_response,
    )
    try:
        raw = APIManager(
            provider_name=provider_name,
            task_name="agent.habitual_conflict",
        ).generate(prompt, model=model)
        parsed = json.loads(_json_text(raw))
        return GoalConflictResult(
            conflict=bool(parsed.get("conflict", False)),
            reason=str(parsed.get("reason", "") or "").strip(),
            raw_response=raw,
        )
    except Exception:
        return _fallback_conflict(habitual_response)


def _fallback_conflict(response: PreparedHabitualResponse) -> GoalConflictResult:
    if response.required_body_resources:
        return GoalConflictResult(False, "没有足够信息表明这个身体反应会妨碍当前目标。")
    return GoalConflictResult(False, "没有发现明确的目标冲突。")


def _json_text(text: str) -> str:
    stripped = str(text or "").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object in goal conflict output.")
    return stripped[start : end + 1]

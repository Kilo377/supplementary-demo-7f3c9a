from __future__ import annotations

import json

from llm.api_manager import APIManager

from agent.habitual_controller import PreparedHabitualResponse

from .prompt import build_arbiter_prompt
from .types import ArbiterResult


def run_arbiter(
    *,
    agent_name: str,
    personality: str,
    intent_text: str,
    trigger_context_text: str,
    current_state_text: str,
    intuition_text: str,
    think_text: str,
    goal_directed_action: str,
    habitual_response: PreparedHabitualResponse,
    entry_reason: str,
    provider_name: str = "ollama",
    model: str | None = None,
) -> ArbiterResult:
    prompt = build_arbiter_prompt(
        agent_name=agent_name,
        personality=personality,
        intent_text=intent_text,
        trigger_context_text=trigger_context_text,
        current_state_text=current_state_text,
        intuition_text=intuition_text,
        think_text=think_text,
        goal_directed_action=goal_directed_action,
        habitual_response=habitual_response,
        entry_reason=entry_reason,
    )
    try:
        raw = APIManager(
            provider_name=provider_name,
            task_name="agent.arbiter",
        ).generate(prompt, model=model)
        parsed = json.loads(_json_text(raw))
        mode = _decision_mode(parsed.get("decision_mode"))
        action_text = str(parsed.get("action", "") or "").strip()
        if not action_text:
            raise ValueError("Arbiter returned no action.")
        return ArbiterResult(
            decision_mode=mode,
            action_text=action_text,
            reason=str(parsed.get("thought", "") or "").strip(),
            raw_response=raw,
        )
    except Exception:
        return ArbiterResult(
            decision_mode="goal_directed",
            action_text=goal_directed_action,
            reason="当前没有形成足够明确的理由改变原来的行动。",
        )


def _decision_mode(value) -> str:
    cleaned = str(value or "").strip().lower()
    allowed = {"goal_directed", "habitual", "sequence", "combine", "inhibit_habit"}
    return cleaned if cleaned in allowed else "goal_directed"


def _json_text(text: str) -> str:
    stripped = str(text or "").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object in arbiter output.")
    return stripped[start : end + 1]

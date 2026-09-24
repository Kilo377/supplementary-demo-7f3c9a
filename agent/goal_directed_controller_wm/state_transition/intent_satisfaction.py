from __future__ import annotations

from llm.api_manager import APIManager

from .intent_satisfaction_prompt import build_transition_intent_satisfaction_prompt
from .json_output import parse_json_object
from .types import TransitionIntentSatisfaction


def evaluate_transition_intent_satisfaction(
    *,
    agent_name: str,
    intent_text: str,
    action_text: str,
    world_feedback: str,
    next_state_text: str,
    action_failed: bool = False,
    provider_name: str = "ollama",
    model: str | None = None,
) -> tuple[TransitionIntentSatisfaction, str, str, str]:
    prompt = build_transition_intent_satisfaction_prompt(
        agent_name=agent_name,
        intent_text=intent_text,
        action_text=action_text,
        world_feedback=world_feedback,
        next_state_text=next_state_text,
    )
    if action_failed:
        return (
            TransitionIntentSatisfaction(
                status="not_satisfied",
                reason="World feedback indicates the action was not successfully executed.",
            ),
            prompt,
            "",
            "",
        )

    raw_response = ""
    try:
        raw_response = APIManager(
            provider_name=provider_name,
            task_name="agent.world_model.intent_satisfaction",
        ).generate(prompt, model=model)
        parsed = parse_json_object(raw_response)
        status = _normalize_status(parsed.get("status"))
        return (
            TransitionIntentSatisfaction(
                status=status,
                reason=str(parsed.get("reason", "") or "").strip(),
            ),
            prompt,
            raw_response,
            "",
        )
    except Exception as error:
        return (
            TransitionIntentSatisfaction(
                status="not_satisfied",
                reason="Unable to stably determine if the Intent has been satisfied; continue treating it as unsatisfied.",
            ),
            prompt,
            raw_response,
            f"Intent satisfaction failed: {error}",
        )


def _normalize_status(value: object) -> str:
    status = str(value or "").strip().lower().replace(" ", "_")
    aliases = {
        "approximate": "approximately_satisfied",
        "approximately": "approximately_satisfied",
        "almost_satisfied": "approximately_satisfied",
        "partial": "approximately_satisfied",
        "unsatisfied": "not_satisfied",
    }
    status = aliases.get(status, status)
    if status in {"satisfied", "approximately_satisfied", "not_satisfied"}:
        return status
    return "not_satisfied"

from __future__ import annotations

from llm.api_manager import APIManager

from .context import build_state_transition_context
from .json_output import parse_json_object
from .prompt import build_state_transition_prompt_from_context
from .types import (
    StateTransition,
    StateTransitionPromptContext,
    StateTransitionResult,
)


def predict_state_transition(
    agent,
    *,
    action_text: str,
    intent_text: str = "",
    recent_experience_count: int = 8,
    provider_name: str = "ollama",
    model: str | None = None,
) -> StateTransitionResult:
    context = build_state_transition_context(
        agent,
        action_text=action_text,
        intent_text=intent_text,
        recent_experience_count=recent_experience_count,
    )
    return run_state_transition(
        context,
        provider_name=provider_name,
        model=model,
    )


def run_state_transition(
    context: StateTransitionPromptContext,
    *,
    provider_name: str = "ollama",
    model: str | None = None,
) -> StateTransitionResult:
    prompt = build_state_transition_prompt_from_context(context)
    raw_response = ""
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.world_model.state_transition",
    )
    try:
        raw_response = api.generate(prompt, model=model)
        parsed = parse_json_object(raw_response)
        transition = StateTransition.from_dict(parsed)
        return StateTransitionResult(
            transition=transition,
            prompt=prompt,
            raw_response=raw_response,
            provider_name=api.provider_name,
            model=api.route.model,
            world_model_mode="simple",
        )
    except Exception as error:
        return StateTransitionResult(
            transition=None,
            prompt=prompt,
            raw_response=raw_response,
            provider_name=api.provider_name,
            model=api.route.model,
            world_model_mode="simple",
            error=f"State transition failed: {error}",
        )

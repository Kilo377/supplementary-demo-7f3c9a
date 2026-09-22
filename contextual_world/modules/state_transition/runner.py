from __future__ import annotations

from contextual_world.json_utils import parse_json_object
from contextual_world.llm_trace import run_traced_llm_module
from contextual_world.types import ContextualWorldTraceStep

from .prompt import build_world_state_transition_prompt
from .types import WorldStateTransitionResult, WorldStateTransitionVariables


def run_world_state_transition(
    variables: WorldStateTransitionVariables,
    *,
    provider_name: str = "ollama",
    model: str | None = None,
) -> tuple[WorldStateTransitionResult, ContextualWorldTraceStep]:
    prompt = build_world_state_transition_prompt(variables)
    return run_traced_llm_module(
        module_name="world_state_transition",
        variables=variables.to_dict(),
        prompt=prompt,
        provider_name=provider_name,
        model=model,
        parse_result=_parse_result,
    )


def _parse_result(raw: str) -> WorldStateTransitionResult:
    parsed = parse_json_object(raw)
    return WorldStateTransitionResult(
        accepted=bool(parsed.get("accepted", False)),
        next_state_delta=dict(parsed.get("next_state_delta", {}) or {}),
        execution_result=dict(parsed.get("execution_result", {}) or {}),
        raw_response=raw,
    )

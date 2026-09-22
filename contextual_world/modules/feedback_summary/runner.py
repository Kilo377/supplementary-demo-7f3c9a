from __future__ import annotations

from contextual_world.llm_trace import run_traced_llm_module
from contextual_world.types import ContextualWorldTraceStep

from .prompt import build_world_feedback_summary_prompt
from .types import WorldFeedbackSummaryResult, WorldFeedbackSummaryVariables


def run_world_feedback_summary(
    variables: WorldFeedbackSummaryVariables,
    *,
    provider_name: str = "ollama",
    model: str | None = None,
    use_llm: bool = True,
) -> tuple[WorldFeedbackSummaryResult, ContextualWorldTraceStep]:
    prompt = build_world_feedback_summary_prompt(variables)
    if use_llm:
        return run_traced_llm_module(
            module_name="world_feedback_summary",
            variables=variables.to_dict(),
            prompt=prompt,
            provider_name=provider_name,
            model=model,
            parse_result=_parse_result,
        )
    summary = str(variables.execution_result.get("actual_event", "") or "").strip()
    result = WorldFeedbackSummaryResult(perception_summary=summary, raw_response="")
    return result, ContextualWorldTraceStep(
        module_name="world_feedback_summary",
        variables=variables.to_dict(),
        prompt=prompt,
        raw_output="",
        parsed_output=result.to_dict(),
        provider_name="deterministic",
        model="none",
        phase="complete",
    )


def _parse_result(raw: str) -> WorldFeedbackSummaryResult:
    return WorldFeedbackSummaryResult(
        perception_summary=raw.strip(),
        raw_response=raw,
    )

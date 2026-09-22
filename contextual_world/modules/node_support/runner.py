from __future__ import annotations

from contextual_world.json_utils import dict_list, parse_json_object
from contextual_world.llm_trace import run_traced_llm_module
from contextual_world.types import ContextualWorldTraceStep

from .prompt import build_world_node_support_prompt
from .types import WorldNodeSupportResult, WorldNodeSupportVariables


def run_world_node_support(
    variables: WorldNodeSupportVariables,
    *,
    provider_name: str = "ollama",
    model: str | None = None,
) -> tuple[WorldNodeSupportResult, ContextualWorldTraceStep]:
    prompt = build_world_node_support_prompt(variables)
    return run_traced_llm_module(
        module_name="world_node_support",
        variables=variables.to_dict(),
        prompt=prompt,
        provider_name=provider_name,
        model=model,
        parse_result=_parse_result,
    )


def _parse_result(raw: str) -> WorldNodeSupportResult:
    parsed = parse_json_object(raw)
    return WorldNodeSupportResult(
        support_status=_normalize_support_status(parsed.get("support_status")),
        support_reason=str(parsed.get("support_reason", "") or ""),
        required_existing_nodes=dict_list(parsed.get("required_existing_nodes")),
        temporary_node_creations=dict_list(parsed.get("temporary_node_creations")),
        context_facts=[
            str(item).strip()
            for item in (parsed.get("context_facts", []) or [])
            if str(item).strip()
        ],
        fallback_result=dict(parsed.get("fallback_result", {}) or {}),
        raw_response=raw,
    )


def _normalize_support_status(value) -> str:
    status = str(value or "").strip()
    if status in {"supported", "unsupported"}:
        return status
    return "supported"

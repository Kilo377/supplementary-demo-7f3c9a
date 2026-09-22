from __future__ import annotations

from contextual_world.json_utils import dict_list, parse_json_object
from contextual_world.llm_trace import run_traced_llm_module
from contextual_world.types import ContextualWorldTraceStep

from .prompt import build_world_interaction_focus_prompt
from .types import WorldInteractionFocusResult, WorldInteractionFocusVariables


def run_world_interaction_focus(
    variables: WorldInteractionFocusVariables,
    *,
    provider_name: str = "ollama",
    model: str | None = None,
) -> tuple[WorldInteractionFocusResult, ContextualWorldTraceStep]:
    prompt = build_world_interaction_focus_prompt(variables)
    return run_traced_llm_module(
        module_name="world_interaction_focus",
        variables=variables.to_dict(),
        prompt=prompt,
        provider_name=provider_name,
        model=model,
        parse_result=_parse_result,
    )


def _parse_result(raw: str) -> WorldInteractionFocusResult:
    parsed = parse_json_object(raw)
    return WorldInteractionFocusResult(
        focused_node_ids=_node_id_list(parsed.get("focused_node_ids")),
        interaction_frame=_interaction_frame_list(parsed.get("interaction_frame")),
        focus_reason=str(parsed.get("focus_reason", "") or ""),
        raw_response=raw,
    )


def _node_id_list(value) -> list[str]:
    result: list[str] = []
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, list):
            for nested in _node_id_list(item):
                if nested not in result:
                    result.append(nested)
            continue
        if isinstance(item, dict):
            continue
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _interaction_frame_list(value) -> list[dict]:
    frames: list[dict] = []
    for item in dict_list(value):
        subject_ids = _node_id_list(item.get("subject_id"))
        object_ids = _node_id_list(item.get("object_id"))
        if not subject_ids or not object_ids:
            continue
        for subject_id in subject_ids:
            for object_id in object_ids:
                frames.append({
                    "subject_id": subject_id,
                    "relation_hint": _relation_hint(item.get("relation_hint")),
                    "object_id": object_id,
                    "reason": str(item.get("reason", "") or ""),
                })
    return frames


def _relation_hint(value) -> str:
    if isinstance(value, list):
        parts = [str(item or "").strip() for item in value if str(item or "").strip()]
        return "_and_".join(parts)
    if isinstance(value, dict):
        return ""
    return str(value or "").strip()

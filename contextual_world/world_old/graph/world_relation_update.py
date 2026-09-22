from __future__ import annotations

import json
from dataclasses import dataclass, field

from llm.api_manager import APIManager
from contextual_world.world_old.prompts.world_relation_update_prompt import build_world_relation_update_prompt


@dataclass
class WorldRelationUpdateResult:
    fact_edges_to_add: list[dict] = field(default_factory=list)
    fact_edges_to_remove: list[dict] = field(default_factory=list)
    agent_state_patch: dict = field(default_factory=dict)
    feedback_narration: str = ""
    raw_response: str = ""


def judge_world_relation_update(
    *,
    agent_name: str,
    action_proposal: str,
    world_action_event: dict | None = None,
    element_support_result: dict,
    agent_node: dict,
    relevant_nodes: list[dict],
    current_fact_edges: list[dict],
    provider_name: str = "ollama",
    model: str | None = None,
    print_output: bool = True,
) -> WorldRelationUpdateResult:
    prompt = build_world_relation_update_prompt(
        agent_name=agent_name,
        action_proposal=action_proposal,
        world_action_event=world_action_event or {},
        element_support_result=element_support_result,
        agent_node=agent_node,
        relevant_nodes=relevant_nodes,
        current_fact_edges=current_fact_edges,
    )
    api = APIManager(provider_name=provider_name)
    raw = api.generate(prompt, model=model)
    check_world_relation_update_output(raw, enabled=print_output)
    parsed = json.loads(_extract_json_text(raw))
    return WorldRelationUpdateResult(
        fact_edges_to_add=_dict_list(parsed.get("fact_edges_to_add")),
        fact_edges_to_remove=_dict_list(parsed.get("fact_edges_to_remove")),
        agent_state_patch=_dict(parsed.get("agent_state_patch")),
        feedback_narration=str(parsed.get("feedback_narration", "") or ""),
        raw_response=raw,
    )


def check_world_relation_update_output(raw_output: str, *, enabled: bool = True) -> None:
    # TODO: Replace this print hook with a meta-check that validates node ids,
    # stale fact removal, and contradiction before applying relation updates.
    if not enabled:
        return
    print("World relation update output:")
    print(raw_output)


def _dict_list(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict(value) -> dict:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    if "```json" in stripped:
        after = stripped.split("```json", 1)[1]
        return after.split("```", 1)[0].strip()
    if "```" in stripped:
        after = stripped.split("```", 1)[1]
        return after.split("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    raise ValueError("No JSON object found in world relation update response.")

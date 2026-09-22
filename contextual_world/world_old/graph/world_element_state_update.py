from __future__ import annotations

import json
from dataclasses import dataclass, field

from llm.api_manager import APIManager
from contextual_world.world_old.prompts.world_element_state_update_prompt import build_world_element_state_update_prompt


@dataclass
class WorldElementStateUpdateResult:
    element_state_updates: list[dict] = field(default_factory=list)
    raw_response: str = ""


def judge_world_element_state_update(
    *,
    agent_name: str,
    action_proposal: str,
    world_action_event: dict | None = None,
    element_support_result: dict,
    relation_update_result: dict,
    relevant_nodes: list[dict],
    element_recent_history: dict | None = None,
    provider_name: str = "ollama",
    model: str | None = None,
    print_output: bool = True,
) -> WorldElementStateUpdateResult:
    prompt = build_world_element_state_update_prompt(
        agent_name=agent_name,
        action_proposal=action_proposal,
        world_action_event=world_action_event or {},
        element_support_result=element_support_result,
        relation_update_result=relation_update_result,
        relevant_nodes=relevant_nodes,
        element_recent_history=element_recent_history,
    )
    raw = APIManager(provider_name=provider_name).generate(prompt, model=model)
    check_world_element_state_update_output(raw, enabled=print_output)
    parsed = json.loads(_extract_json_text(raw))
    return WorldElementStateUpdateResult(
        element_state_updates=_dict_list(parsed.get("element_state_updates")),
        raw_response=raw,
    )


def check_world_element_state_update_output(raw_output: str, *, enabled: bool = True) -> None:
    # TODO: Replace this print hook with a meta-check that validates element ids
    # and prevents relation facts from being smuggled into element state fields.
    if not enabled:
        return
    print("World element state update output:")
    print(raw_output)


def _dict_list(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


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
    raise ValueError("No JSON object found in world element state update response.")

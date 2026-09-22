from __future__ import annotations

import json
from dataclasses import dataclass, field

from llm.api_manager import APIManager
from contextual_world.world_old.prompts.agent_body_state_update_prompt import build_agent_body_state_update_prompt


@dataclass
class AgentBodyStateUpdateResult:
    agent_state_patch: dict = field(default_factory=dict)
    feedback_hint: str = ""
    raw_response: str = ""

    def to_relation_update_dict(self) -> dict:
        return {
            "fact_edges_to_add": [],
            "fact_edges_to_remove": [],
            "agent_state_patch": dict(self.agent_state_patch),
            "feedback_narration": self.feedback_hint,
        }


def judge_agent_body_state_update(
    *,
    agent_name: str,
    action_proposal: str,
    world_action_event: dict | None = None,
    agent_node: dict,
    current_agent_fact_edges: list[dict],
    context_facts: list[str],
    provider_name: str = "ollama",
    model: str | None = None,
    print_output: bool = True,
) -> AgentBodyStateUpdateResult:
    prompt = build_agent_body_state_update_prompt(
        agent_name=agent_name,
        action_proposal=action_proposal,
        world_action_event=world_action_event or {},
        agent_node=agent_node,
        current_agent_fact_edges=current_agent_fact_edges,
        context_facts=context_facts,
    )
    api = APIManager(provider_name=provider_name)
    raw = api.generate(prompt, model=model)
    check_agent_body_state_update_output(raw, enabled=print_output)
    parsed = json.loads(_extract_json_text(raw))
    return AgentBodyStateUpdateResult(
        agent_state_patch=_dict(parsed.get("agent_state_patch")),
        feedback_hint=str(parsed.get("feedback_hint", "") or ""),
        raw_response=raw,
    )


def check_agent_body_state_update_output(raw_output: str, *, enabled: bool = True) -> None:
    # TODO: Replace this print hook with a meta-check that ensures the body
    # update does not mutate external world facts or clear relations silently.
    if not enabled:
        return
    print("Agent body state update output:")
    print(raw_output)


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
    raise ValueError("No JSON object found in agent body state update response.")

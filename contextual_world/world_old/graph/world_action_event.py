from __future__ import annotations

import json
from dataclasses import dataclass, field

from llm.api_manager import APIManager
from contextual_world.world_old.prompts.world_action_event_prompt import build_world_action_event_prompt


@dataclass
class WorldActionEventResult:
    accepted: bool = True
    route: str = ""
    actual_event: str = ""
    event_effects: list[str] = field(default_factory=list)
    involved_node_ids: list[str] = field(default_factory=list)
    estimated_duration: str = ""
    reason: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "route": self.route,
            "actual_event": self.actual_event,
            "event_effects": list(self.event_effects),
            "involved_node_ids": list(self.involved_node_ids),
            "estimated_duration": self.estimated_duration,
            "reason": self.reason,
        }


def generate_world_action_event(
    *,
    agent_name: str,
    action_proposal: str,
    route: str,
    element_support_result: dict,
    agent_node: dict,
    relevant_nodes: list[dict],
    current_fact_edges: list[dict],
    element_recent_history: dict | None = None,
    provider_name: str = "ollama",
    model: str | None = None,
    print_output: bool = True,
) -> WorldActionEventResult:
    prompt = build_world_action_event_prompt(
        agent_name=agent_name,
        action_proposal=action_proposal,
        route=route,
        element_support_result=element_support_result,
        agent_node=agent_node,
        relevant_nodes=relevant_nodes,
        current_fact_edges=current_fact_edges,
        element_recent_history=element_recent_history,
    )
    raw = APIManager(provider_name=provider_name).generate(prompt, model=model)
    check_world_action_event_output(raw, enabled=print_output)
    parsed = json.loads(_extract_json_text(raw))
    return WorldActionEventResult(
        accepted=bool(parsed.get("accepted", True)),
        route=str(parsed.get("route", "") or route),
        actual_event=str(parsed.get("actual_event", "") or ""),
        event_effects=[
            str(item).strip()
            for item in (parsed.get("event_effects", []) or [])
            if str(item).strip()
        ],
        involved_node_ids=[
            str(item).strip()
            for item in (parsed.get("involved_node_ids", []) or [])
            if str(item).strip()
        ],
        estimated_duration=str(parsed.get("estimated_duration", "") or ""),
        reason=str(parsed.get("reason", "") or ""),
        raw_response=raw,
    )


def check_world_action_event_output(raw_output: str, *, enabled: bool = True) -> None:
    # TODO: Replace this print hook with a meta-check that validates involved
    # node ids and ensures the event does not advance beyond one action step.
    if not enabled:
        return
    print("World action event output:")
    print(raw_output)


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
    raise ValueError("No JSON object found in world action event response.")

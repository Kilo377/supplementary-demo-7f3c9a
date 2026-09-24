from __future__ import annotations

import json

from llm.api_manager import APIManager


def build_belief_state_update_prompt(
    *,
    agent_name: str,
    state_fields: dict,
) -> str:
    return f"""You are updating the self-belief of a human agent.

The input consists of four fields after {agent_name} has just completed an action round:
- action_proposal: What {agent_name} just intended to do
- movement: Where {agent_name} moved from and to; if moved=false, they stayed in place
- interacted_elements: Interaction elements successfully bound to the scene schema by the system
- environment_feedback: What actually happened in the environment feedback

Your task:
Based on these four fields, summarize {agent_name}'s current belief about their own state.

Requirements:
- One short sentence in Chinese
- Do not propose next steps; only summarize.
- Do not repeat the full log
- If action_proposal and environment_feedback are inconsistent, prioritize environment_feedback and movement
- environment_feedback is the primary basis for {agent_name} to form self-belief
- If interacted_elements is empty, it only means this action was not bound to an explicit world element; it does not mean the action did not occur, nor that {agent_name} did not actually interact
- If environment_feedback describes objects in hand, food, packaging, tableware, or other low-surprise objects not explicitly modeled, acknowledge that the action actually occurred and include it in the self-belief
- Must cover three points: what was just intended; where from and to, or if stayed in place; what was actually done

Four fields:
{json.dumps(state_fields, ensure_ascii=False, indent=2)}

Output format:
{agent_name} just intended to...; {agent_name} moved from... to... / {agent_name} stayed in place; {agent_name} actually....
"""


def update_belief_state(
    *,
    agent_name: str,
    state_fields: dict,
    provider_name: str = "ollama",
    model: str | None = None,
) -> str:
    try:
        api = APIManager(
            provider_name=provider_name,
            task_name="agent.self_belief_update",
        )
        raw = api.generate(
            build_belief_state_update_prompt(
                agent_name=agent_name,
                state_fields=state_fields,
            ),
            model=model,
        )
        return clean_belief_state_update(raw, agent_name=agent_name)
    except Exception:
        return fallback_belief_state_update(agent_name=agent_name, state_fields=state_fields)


def clean_belief_state_update(text: str, *, agent_name: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1].replace("json", "", 1).strip()
        else:
            cleaned = cleaned.replace("```", "").strip()
    lines = [line.strip(" -") for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    cleaned = " ".join(lines[:3]).strip()
    if agent_name not in cleaned:
        cleaned = f"{agent_name} believes {cleaned}"
    return cleaned


def fallback_belief_state_update(*, agent_name: str, state_fields: dict) -> str:
    proposal = state_fields.get("action_proposal", "") or "Unclear action"
    feedback = state_fields.get("environment_feedback", "") or "The environment did not provide clear feedback"
    movement = state_fields.get("movement", {}) or {}
    from_text = movement.get("from", "")
    to_text = movement.get("to", "")
    moved = movement.get("moved", False)
    if moved:
        movement_text = f"Moved from {from_text} to {to_text}"
    else:
        movement_text = f"Stayed at {to_text or from_text or 'the original location'}"
    return f"{agent_name} just thought about {proposal}; {agent_name} {movement_text}; What {agent_name} actually experienced was: {feedback}"

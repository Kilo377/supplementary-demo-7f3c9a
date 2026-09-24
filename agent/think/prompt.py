from __future__ import annotations

import json

from agent.intent.state import IntentState
from agent.intuition import IntuitionResult
from agent.working_memory import WorkingMemoryFrame
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine

from .types import ThinkResult


def build_think_prompt(
    *,
    agent_name: str,
    intent: IntentState,
    working_memory: WorkingMemoryFrame,
    intuition: IntuitionResult,
) -> str:
    return f"""
You need to simulate {agent_name}'s slow-thinking process.

This is not a narrator's voice, not an action plan, nor an explanation to others. Put yourself in {agent_name}'s shoes and think briefly in the first person.
You must base your thoughts on {agent_name}'s biography, desire state, physical state, current location, current Intent, recent intuition, and what has already happened.
Different people should think differently: cautious people will weigh risks, stability, and consequences more heavily; impulsive people will trial-and-error faster; tired people will be more conservative.

The thinking content should help {agent_name} clarify the current issue, but do not write it directly as a multi-step action list.

Return only JSON, no explanation.

JSON schema:
{{
  "thought": "First-person Chinese inner thought",
  "conclusion": "A one-sentence summary of the inclination after thinking"
}}

Current Intent:
{intent.intent_text} [{intent.status}]

Recent Intuition:
[{intuition.route}] {intuition.thought}

Working Memory:
{working_memory.format_for_action_prompt()}
""".strip()


def build_post_think_route_prompt(
    *,
    agent_name: str,
    intent: IntentState,
    working_memory: WorkingMemoryFrame,
    think_result: ThinkResult,
    engine: PhysicsEngine,
) -> str:
    areas = [
        {"area_id": area.node_id, "area_name": area.name}
        for area in engine.home.areas
    ]
    areas_json = json.dumps(areas, ensure_ascii=False, indent=2)
    return f"""
{agent_name} just thought carefully about it:
{think_result.format_for_prompt()}

Based on this thinking result, the current Intent, and Working Memory, determine which route {agent_name} should enter next.
This is still a first-person inner reaction, not a complete plan.

The route can only be:
- action: wants to perform a specific action at the current location or nearby; will be refined by Action Proposal later.
- wait: decides to wait a moment, observe, or stay put for now. Must fill in wait_duration based on recent thinking, and naturally state how long they plan to wait in thought; e.g., if wanting to rest for over ten minutes, fill in 15min.
- walk: decides to go to another room or area. Try to fill in target_area_id or target_area_name as much as possible.
- chat: decides to talk to someone. Try to fill in chat_target.

Return only JSON, no explanation.

JSON schema:
{{
  "route": "action | wait | walk | chat",
  "thought": "A one-sentence first-person Chinese inner thought",
  "target_area_id": "Fill in when walking, empty string if none",
  "target_area_name": "Fill in when walking, empty string if none",
  "chat_target": "Fill in when chatting, empty string if none",
  "wait_duration": "Fill in when waiting: 30s | 1min | 3min | 5min | 10min | 15min | 20min | 30min; empty string for other routes"
}}

Current Intent:
{intent.intent_text} [{intent.status}]

Known rooms:
{areas_json}

Working Memory:
{working_memory.format_for_action_prompt()}
""".strip()

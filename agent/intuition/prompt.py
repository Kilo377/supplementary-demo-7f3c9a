from __future__ import annotations

import json

from agent.intent.state import IntentState
from agent.working_memory import WorkingMemoryFrame
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine


def build_intuition_prompt(
    *,
    agent_name: str,
    intent: IntentState,
    working_memory: WorkingMemoryFrame,
    engine: PhysicsEngine,
    action_sample_window: int = 1,
    include_think: bool = True,
) -> str:
    sample_window = min(12, max(1, int(action_sample_window)))
    areas = [
        {"area_id": area.node_id, "area_name": area.name}
        for area in engine.home.areas
    ]
    areas_json = json.dumps(areas, ensure_ascii=False, indent=2)
    route_options = "action | wait | walk | chat | think" if include_think else "action | wait | walk | chat"
    think_example = "\n- The storage room seems inaccessible; I need to figure out how to start without those tools first." if include_think else ""
    think_route = "\n- think: For complex, stuck, or unclear goals where a strategy needs to be formed first, the instinct is to slow down and think. Don't mistake ordinary hesitation for thinking." if include_think else ""
    if sample_window == 1:
        output_block = f"""Return only JSON, no explanation.

JSON schema:
{{
  "route": "{route_options}",
  "thought": "A first-person Chinese inner thought",
  "target_area_id": "Fill in for walk; empty string if not applicable",
  "target_area_name": "Fill in for walk; empty string if not applicable",
  "chat_target": "Fill in for chat; empty string if not applicable",
  "wait_duration": "Fill in for wait: 30s | 1min | 3min | 5min | 10min | 15min | 20min | 30min; empty string for other routes"
}}"""
    else:
        output_block = f"""Please provide {sample_window} possible Intuition replies.

Return only JSON, no explanations.

JSON schema:
{{
  "replies": [
    {{
      "route": "{route_options}",
      "thought": "A first-person Chinese inner thought",
      "target_area_id": "Fill in for 'walk'; leave empty if not applicable",
      "target_area_name": "Fill in for 'walk'; leave empty if not applicable",
      "chat_target": "Fill in for 'chat'; leave empty if not applicable",
      "wait_duration": "Fill in for 'wait': 30s | 1min | 3min | 5min | 10min | 15min | 20min | 30min; leave empty for other routes"
    }}
  ]
}}"""
    return f"""
You are to simulate {agent_name}'s immediate intuitive reaction after reading the Working Memory.

This is not a complete plan, nor is it post-hoc reflection. Put yourself in {agent_name}'s shoes and think of one sentence in the first person.
This sentence should feel like an inner reaction, for example:
- The microwave isn't done yet; I'd better wait.
- The computer is broken; I plan to check what's wrong.
- There's nothing here I want, so I'll go back to the bedroom to see.
- I want to talk to someone. {think_example}

You must consider the current Intent and not forget what {agent_name} was originally doing.
If the environment doesn't strongly interrupt, follow the current Intent forward in your reaction.
If you intend to act, state only the first reaction; do not elaborate on multi-step plans.

The route can only be:
- action: Want to perform a specific action at or near the current location; this will later be refined by Action Proposal.
- wait: The first reaction is to wait a moment, observe, or stay put for now. You must fill in wait_duration based on what you are waiting for, and naturally mention how long you plan to wait in the thought.
- walk: The first reaction is to go to another room or area. You must try to fill in target_area_id or target_area_name.
- chat: The first reaction is to talk to someone. Try to fill in chat_target. {think_route}

{output_block}

Current Intent:
{intent.intent_text} [{intent.status}]

Known Areas:
{areas_json}

Working Memory:
{working_memory.format_for_action_prompt()}
""".strip()

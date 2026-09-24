from __future__ import annotations

import json

from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine


def build_transition_iteration_intuition_prompt(
    *,
    agent_name: str,
    intent_text: str,
    transition_state_text: str,
    failure_context_text: str = "",
    action_sample_window: int = 1,
    engine: PhysicsEngine,
) -> str:
    areas = [
        {"area_id": area.node_id, "area_name": area.name}
        for area in engine.home.areas
    ]
    areas_json = json.dumps(areas, ensure_ascii=False, indent=2)
    sections = [
        f"You are {agent_name}.",
        f"What you currently want:\n{intent_text}",
        (
            "You just mentally simulated one step forward. Below is the state you believe you will be in next, based on your experience.\nTreat this as the current state you believe in during this continued simulation:\n\n"
            f"{transition_state_text}"
        ),
    ]
    failure = str(failure_context_text or "").strip()
    if failure:
        sections.append(
            "The previous simulation revealed an issue that needs to be addressed:\n\n" + failure
        )
    sample_count = max(1, min(12, int(action_sample_window)))
    if sample_count == 1:
        output_instruction = """
Return only JSON, no explanation.

JSON schema:
{
  "route": "action | wait | walk | chat",
  "thought": "A first-person Chinese inner thought",
  "target_area_id": "Fill in when route is 'walk'; empty string otherwise",
  "target_area_name": "Fill in when route is 'walk'; empty string otherwise",
  "chat_target": "Fill in when route is 'chat'; empty string otherwise",
  "wait_duration": "Fill in when route is 'wait': 30s | 1min | 3min | 5min | 10min | 15min | 20min | 30min; empty string for other routes"
}
""".strip()
    else:
        output_instruction = f"""
Provide {sample_count} different possible subsequent actions. The different responses should represent actual behavioral differences, not just rephrasings of the same sentence.

Return only JSON, no explanation.

JSON schema:
{{
  "replies": [
    {{
      "route": "action | wait | walk | chat",
      "thought": "A first-person Chinese inner thought",
      "target_area_id": "Fill in when route is 'walk'; empty string otherwise",
      "target_area_name": "Fill in when route is 'walk'; empty string otherwise",
      "chat_target": "Fill in when route is 'chat'; empty string otherwise",
      "wait_duration": "Fill in when route is 'wait'; empty string for other routes"
    }}
  ]
}}
""".strip()
    sections.extend([
        f"The rooms you know are:\n{areas_json}",
        f"""
Based on this new state, what does {agent_name} plan to do next to try to achieve your goal?


The route can only be:
- action: perform a specific action.
- wait: wait, observe, or remain temporarily still. Must fill in wait_duration, and naturally state how long you intend to wait in the thought.
- walk: go to another room or area. Try to fill in target_area_id or target_area_name as much as possible.
- chat: talk to someone. Try to fill in chat_target.

{output_instruction}
""".strip(),
    ])
    return "\n\n".join(section for section in sections if section.strip()).strip()

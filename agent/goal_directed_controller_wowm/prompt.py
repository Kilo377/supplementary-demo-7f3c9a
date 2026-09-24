from __future__ import annotations

from agent.action.prompt import (
    format_short_time_memory_for_action_prompt,
    format_spatial_belief_for_action_prompt,
)


def build_without_world_model_action_prompt(
    *,
    agent_name: str,
    intent,
    short_time_memory,
    spatial_belief,
    current_area_id: str,
    scene_name: str,
    action_sample_window: int = 1,
) -> str:
    sample_window = min(12, max(1, int(action_sample_window)))
    intent_text = str(getattr(intent, "intent_text", "") or "").strip()
    memory_text = format_short_time_memory_for_action_prompt(short_time_memory)
    spatial_text = format_spatial_belief_for_action_prompt(
        spatial_belief,
        current_area_id=current_area_id,
        scene_name=scene_name,
        agent_name=agent_name,
    )
    return f"""
You are generating the next action that {agent_name} will actually execute.

[Current Task Objective]
{intent_text or "(No clear Intent.)"}

[Recent Actions Taken]
{memory_text}

[Current Spatial Awareness]
{spatial_text}

[Generation Task]
Generate {sample_window} candidate actions that {agent_name} might actually execute next.
Candidate actions must serve the current task objective and be ordered from "most aligned with the goal and most suitable for execution now" to "relatively suboptimal."

[Action Requirements]
- Each candidate describes only one specific action.
- Actions must start with {agent_name}.
- These are actions to be directly executed by the world; do not explain reasons.
- Prioritize real furniture or items present in the environment.
- Do not repeat actions that have failed or produced no actual progress.
- Do not fabricate rooms, furniture, or tools not present in the Spatial Belief.
- Do not output thought processes or plans for subsequent steps.
- If the next step requires interacting with a real item in another room, directly describe the specific interaction with that item.

[Output Format]
Output only the following JSON; do not output Markdown or other text:
{{
  "actions": [
    {{"action": "A specific action executed by {agent_name}."}}
  ]
}}

The actions array must contain exactly {sample_window} items; each item describes only one action.
""".strip()

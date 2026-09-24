from __future__ import annotations

import json


def build_user_probe_prompt(
    user_command: str,
    *,
    actor_name: str,
    world_state: dict,
    focus_element_name: str = "",
) -> str:
    world_json = json.dumps(world_state, ensure_ascii=False, indent=2)
    focus_text = ""
    if focus_element_name:
        focus_text = f"\nCurrently, {actor_name} has arrived near the target. The current main interaction target is: {focus_element_name}\n"
    return f"""You are a home-scene world probe planner.

Your tasks:
1. Read the current world state
2. Understand the user's natural language command
3. Output strict JSON indicating which elements' states should be modified

Requirements:
- Only match based on areas and elements given in the world
- Do not fabricate non-existent element IDs
- If the user says "set the whole kitchen on fire," you can use area-level operations
- If the user says "make the stove catch fire," try to locate the most relevant elements, such as the burner, pot, etc.
- The user's command may also describe an interaction action already performed by the agent, e.g., "{actor_name} sat on the single sofa and adjusted the position of pillow 1"
- For such interaction actions, treat them as environmental changes that have already occurred, and change the related elements' physical_status to more appropriate short English status words
- Examples include: occupied, adjusted, held, opened, closed, regular, on_fire, leaking, withered, wet, broken, smoky
- If an action implies returning an anomaly to normal, such as "turn off the leaking showerhead" or "extinguish the fire on the stove," change the corresponding status back to regular, or to a more reasonable result state
- For interaction actions, default to modifying only 1 to 3 directly related elements; do not change the entire area at once
- Only allow area-level operations when the user explicitly mentions whole areas like "the entire kitchen" or "the entire living room"
- For interaction actions, prioritize element-level operations; do not change the status of unrelated furniture
- Use short English words for physical_status, e.g., on_fire, leaking, withered, wet, broken, smoky
- Only modify interaction_status in very clear contexts of character usage, such as in_use or idle
- summary must be a short Chinese sentence describing the result, like "{actor_name} has sat down and adjusted the pillow position."
- Do not include reasoning processes in summary; do not explain why
- Return only JSON; do not explain; do not use markdown
- The default normal state is regular

JSON schema:
{{
  "summary": "Chinese summary",
  "operations": [
    {{
      "scope": "element" | "area",
      "target_area_id": "optional",
      "target_element_ids": ["optional, prefer using IDs"],
      "target_element_names": ["optional, if unsure about ID, provide name"],
      "physical_status": "optional status word",
      "evolution_status": "optional changing or stable",
      "interaction_status": "optional status word",
      "reason": "why this match was made"
    }}
  ]
}}

Current world state:
{world_json}
{focus_text}

User command:
{user_command}
"""

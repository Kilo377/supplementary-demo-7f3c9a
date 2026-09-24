from __future__ import annotations

import json


def build_target_resolution_prompt(
    *,
    agent_name: str,
    action_text: str,
    current_area_id: str,
    current_area_name: str,
    scene_elements_text: str,
    known_world: dict,
    world_graph_context: dict | None = None,
) -> str:
    world_json = json.dumps(known_world, ensure_ascii=False, indent=2)
    graph_context_json = json.dumps(world_graph_context or {}, ensure_ascii=False, indent=2)
    graph_block = ""
    if world_graph_context:
        graph_block = f"""
Temporary objects and interaction facts that still hold in the recent real-world state:
{graph_context_json}

Note: Temporary objects, items in hand, and body actions themselves are not permanent elements of the scene. Only return target_element_id when an action requires approaching an existing piece of furniture or spatial object.
"""

    return f"""
You are the target resolver. You do not execute actions.
You need to separately determine the actual object being operated on by this action, and the navigation anchor point the body needs to approach to perform the action; also determine whether the action is completed upon reaching the anchor point.

Elements in the scene: {scene_elements_text}

Current area:
- id: {current_area_id}
- name: {current_area_name}

The actual action {agent_name} wants to perform is:
{action_text}

Rules:
- target_element_id is the object actually being operated on, picked up, organized, observed, or changed.
- navigation_anchor_element_id is the anchor point the body needs to approach to execute the action.
- For example, 'straighten the remote control': the target is the remote control, and the navigation anchor is the coffee table holding it.
- If the object being operated on is itself furniture or equipment, target and navigation anchor can be the same.
- Prioritize resolving elements in the current area, unless the action explicitly points to an element in another room or area.
- If the action is mainly about oneself, a temporary item in hand, something already held, or does not require approaching furniture, you may leave navigation_anchor_element_id empty.
- If the target mentioned in the action is not a scene element, do not create new elements; choose an existing element it actually attaches to or rests on, otherwise leave it empty.
- Do not return area. Only return element.
- arrival_completes_action=true means the action itself is just walking to, coming to, approaching, or standing next to the target; no further operation of the target occurs after arrival.
- arrival_completes_action=false means arrival is only a prerequisite, and the action also requires opening, picking up, cleaning, using, observing, or otherwise changing/operating on the target.
- Judge based on the overall meaning of action_text; do not change this field just because the target is currently close.
- Return only JSON, no explanation.

JSON schema:
{{
  "operation_target_element_id": "ID of the actual object being operated on, empty string if none",
  "operation_target_element_name": "Name of the actual object being operated on, empty string if none",
  "navigation_anchor_element_id": "ID of the anchor point the body needs to approach, empty string if none",
  "navigation_anchor_element_name": "Name of the anchor point, empty string if none",
  "secondary_target_element_id": "Empty string if none",
  "arrival_completes_action": false,
  "reason": "A brief Chinese explanation"
}}

Spatial memory index:
{world_json}

{graph_block}
"""

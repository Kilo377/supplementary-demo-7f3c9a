from __future__ import annotations

import json


def build_world_action_event_prompt(
    *,
    agent_name: str,
    action_proposal: str,
    route: str,
    element_support_result: dict,
    agent_node: dict,
    relevant_nodes: list[dict],
    current_fact_edges: list[dict],
    element_recent_history: dict | None = None,
) -> str:
    support_json = json.dumps(element_support_result, ensure_ascii=False, indent=2)
    agent_node_json = json.dumps(agent_node, ensure_ascii=False, indent=2)
    relevant_nodes_json = json.dumps(relevant_nodes, ensure_ascii=False, indent=2)
    current_fact_edges_json = json.dumps(current_fact_edges, ensure_ascii=False, indent=2)
    history_json = json.dumps(element_recent_history or {}, ensure_ascii=False, indent=2)
    return f"""You are the simulation environment world itself, responsible for generating the actual world event that occurs in this small step based on the current world state and the agent's action proposal.

The agent's action proposal is merely what they intend to do; it does not mean it has already happened.
The previous layer has provided the route and element_support_result, indicating which nodes are roughly involved in this action and whether a temporary element is needed.

Your task is to decide at the world level what actually happens in this small step.
You do not directly generate agent_state_patch, element_state_updates, or fact_edges.
You only generate an authoritative world_action_event; subsequent modules will project agent state, element state, and relationship changes based on this event.

element_recent_history is the local history of recent steps for relevant elements, used to help judge whether the current action follows the existing state of that element.
If an element has entered a clear phase, such as running, washing, heating, filling, or locked, world_action_event should maintain continuity with this phase.
If the action proposal does not form a complete causal chain with the current state, write actual_event as a statement of the true state at this moment.
For example, after the washing machine has closed its door, started, and is running, if the agent wants to put clothes in directly, actual_event can be written as "the washing machine is still running, clothes remain in their original position," with accepted=false.
If the action proposal intends to place an item into a container or device with a door/lid, and relevant_nodes shows door_state=closed, and the action proposal does not explicitly open the door/lid, then actual_event should be written as "the door remains closed, item stays in its original position," with accepted=false.
If the action proposal intends to adjust the program or contents inside a device that is running/washing/heating, and the action proposal does not explicitly pause, stop, or reopen the device, then actual_event should state that the device continues to run according to its current state, with accepted=false.

Requirements:
- actual_event must describe the event that actually occurred in this small step, using {agent_name} as the subject.
- event_effects should write the direct factual consequences of this event; short phrases are sufficient.
- involved_node_ids can only use node_ids that appear in relevant_nodes, agent_node, or current_fact_edges.
- If element_support_result has already created/referenced a temporary element, include it in the event understanding, but do not say "created out of nowhere."
- If the action proposal explicitly mentions a target object, such as a laundry basket, microwave, refrigerator, or bookshelf, do not replace the target with another similar object. If the target does not exist, you can use a temporary element based on element_support_result, or write the event as no change in current state.
- If agent_node shows that {agent_name} is already near or in the same area as the relevant element, do not write "walked to/arrived at/moved to the target" in actual_event.
- Navigation or pre-positioning are external steps before action execution and do not belong to this world_action_event; unless the action proposal itself is movement, actual_event only describes true interaction or physical actions.
- For ordinary, low-surprise actions with sufficient element support, default to letting the action succeed; do not fabricate failure out of nowhere.
- Only write the event as failed or invalid when the current world state or element_support_result explicitly shows it is broken, powered off, missing necessary targets, unreachable, or blocked.
- For example, "pressing the computer power button/opening the computer" should produce the direct consequence of "computer started/power on" when the computer exists and there are no fault facts; do not write "the computer remains off."
- Do not advance to the next step; only describe this small step of the action proposal.
- If the action cannot occur, set accepted=false and explain the reason; otherwise, set accepted=true.
- When accepted=false, still write actual_event and event_effects as world state statements, not as rule declarations.
- Avoid using adjudicative terms like "cannot," "unable to," "not allowed," or "usually unable" in actual_event, event_effects, or reason; rewrite them as state statements like "the door remains closed," "the device is still running," or "the item is still in hand."
- estimated_duration is the rough duration of this small step action; use 10s, 20s, 30s, or 1min, 5min, 10min.
- Do not output explanations outside of JSON.

Return format:
{{
  "accepted": true or false,
  "route": "{route}",
  "actual_event": "{agent_name} actually did what.",
  "event_effects": [
    "Direct factual consequence"
  ],
  "involved_node_ids": [
    "Existing node id"
  ],
  "estimated_duration": "10s | 20s | 30s | 1min | 5min | 10min",
  "reason": "If accepted=false, explain why; otherwise can be empty"
}}

route:
{route}

element_support_result:
{support_json}

agent_node:
{agent_node_json}

relevant_nodes:
{relevant_nodes_json}

current_fact_edges:
{current_fact_edges_json}

element_recent_history:
{history_json}

agent action proposal:
{action_proposal}
""".strip()

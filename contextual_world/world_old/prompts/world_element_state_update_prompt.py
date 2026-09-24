from __future__ import annotations

import json


def build_world_element_state_update_prompt(
    *,
    agent_name: str,
    action_proposal: str,
    world_action_event: dict | None = None,
    element_support_result: dict,
    relation_update_result: dict,
    relevant_nodes: list[dict],
    element_recent_history: dict | None = None,
) -> str:
    event_json = json.dumps(world_action_event or {}, ensure_ascii=False, indent=2)
    support_json = json.dumps(element_support_result, ensure_ascii=False, indent=2)
    relation_json = json.dumps(relation_update_result, ensure_ascii=False, indent=2)
    nodes_json = json.dumps(relevant_nodes, ensure_ascii=False, indent=2)
    history_json = json.dumps(element_recent_history or {}, ensure_ascii=False, indent=2)
    return f"""You are responsible for updating the state of element nodes themselves in the simulation environment.

The agent's action proposal is what {agent_name} just intended to do.
The world_action_event is the actual small-step event that has already occurred in the world; it is more authoritative than the action proposal.
The previous modules have determined which elements support this action and decided how the factual relationship between the agent and the elements changes.

Your task is not to judge whether the action is feasible, nor to create temporary elements, nor to update the agent's own physical state.
Your task is simply to determine, based on the world_action_event, whether the state fields of the relevant element nodes should change after this small step actually occurs.

element_recent_history is the local history of recent steps for the relevant element.
When updating the state, refer to the current status in relevant_nodes and element_recent_history; do not regress an element to an earlier stage without cause.
If a device has already entered a clear phase such as running, washing, heating, filling, or locked, the state update should maintain phase continuity.
Only change the key state or content of that device when world_action_event explicitly gives causal actions like pause, stop, open, remove, or close.
For example, after a washing machine has started running, subsequent vague actions usually keep power_state and contains as they are; if the event is merely a state statement, element_state_updates can be empty.

An element node's own state includes:
- physical_status: rough physical state, e.g., regular, clean, dirty, moved, open, closed.
- evolution_status: time-evolving state, e.g., stable, running, heating, cooling.
- interaction_status: current interaction state, e.g., idle, in_use, inspected.
- state_details: the element's own specific details, e.g., door_state=open, flow_state=on, surface_state=clean, power_state=running, contains=food.

Only update element states that will affect subsequent action judgments.
Do not repeatedly express the relationship between the agent and the element; relationships like holding, looking_at, sitting_on, placed_on are already handled by relation update.
Do not create or update element state for transient effects like foam, smell, water vapor, flying dust, or fragrance; if there is no stable impact, do not write them.
If world_action_event explicitly indicates that a device is opened, started, running, closed, or stopped, you must update the state_details of that device's element, e.g., power_state=on/off/running, and update evolution_status if needed.
Device internal states only need to be sufficient to support subsequent judgments; do not over-split them into multiple micro-program states; for example, a washing machine can be roughly recorded as running / contains=dirty_clothes, a microwave as heating, and a computer as power_state=on.
Do not advance subsequent steps; only handle the state changes directly caused by this small step of the action proposal.

element_id must come from existing permanent_element or temporary_element in relevant_nodes.
Prioritize updating the status of permanent_elements, e.g., refrigerator door, shower water flow, microwave operation, desktop cleanliness.
The lifecycle status of temporary_elements is usually already handled by the support layer; unless there is a clear change in the state details of the temporary_element itself, do not update it repeatedly.

The keys in state_details should be short, stable, and reusable, e.g., door_state, flow_state, power_state, surface_state, temperature, contains, recently_removed.
The values in state_details must be short states; do not write full sentences.
If the element already has a suitable state_details key, prioritize reusing it.
If there is no clear change in the element's own state, return an empty array for element_state_updates.

Return only JSON; do not explain the reasoning process.

Return format:
{{
  "element_state_updates": [
    {{
      "element_id": "existing element node id",
      "physical_status": null or "short state",
      "evolution_status": null or "short state",
      "interaction_status": null or "short state",
      "state_details": {{
        "short key": "short value"
      }}
    }}
  ]
}}

Example:

action proposal: {agent_name} opens the refrigerator.
Return:
{{
  "element_state_updates": [
    {{
      "element_id": "fridge_01",
      "physical_status": null,
      "evolution_status": null,
      "interaction_status": "in_use",
      "state_details": {{"door_state": "open"}}
    }}
  ]
}}

action proposal: {agent_name} takes a bottle of milk out of the refrigerator.
Return:
{{
  "element_state_updates": [
    {{
      "element_id": "fridge_01",
      "physical_status": null,
      "evolution_status": null,
      "interaction_status": "in_use",
      "state_details": {{"door_state": "open", "recently_removed": "milk"}}
    }}
  ]
}}

action proposal: {agent_name} wipes the office desk.
Return:
{{
  "element_state_updates": [
    {{
      "element_id": "desk_01",
      "physical_status": null,
      "evolution_status": null,
      "interaction_status": "in_use",
      "state_details": {{"surface_state": "clean"}}
    }}
  ]
}}

action proposal: {agent_name} turns on the computer.
Return:
{{
  "element_state_updates": [
    {{
      "element_id": "computer_01",
      "physical_status": null,
      "evolution_status": "running",
      "interaction_status": "in_use",
      "state_details": {{"power_state": "on"}}
    }}
  ]
}}

action proposal: {agent_name} turns off the computer.
Return:
{{
  "element_state_updates": [
    {{
      "element_id": "computer_01",
      "physical_status": null,
      "evolution_status": "stable",
      "interaction_status": "idle",
      "state_details": {{"power_state": "off"}}
    }}
  ]
}}

element_support_result:
{support_json}

world_action_event:
{event_json}

relation_update_result:
{relation_json}

relevant_nodes:
{nodes_json}

element_recent_history:
{history_json}

agent action proposal:
{action_proposal}
""".strip()

from __future__ import annotations

import json

from .types import WorldStateTransitionVariables


def build_world_state_transition_prompt(variables: WorldStateTransitionVariables) -> str:
    return f"""You are the world_state_transition module.

We are simulating a small step of action by a human agent in an environment. The environment is represented as a graph:
- human_agent, permanent_element, and temporary_element are all nodes.
- A fact edge represents currently valid factual relationships, such as agent holding milk or book placed_on desk.
- Node state represents the node's own status, such as refrigerator door_state=open, agent posture=sitting, milk status=held.

The action proposal you receive is a small step the agent intends to take.
It is not an already occurred fact.
Your task is to determine how the world graph should transition from st to st+1 based on the current subgraph state after this small step.

Crucially:
You must first decide the state changes in st+1, and then summarize the action outcome based on those state changes.
Do not write the natural language outcome first and then reverse-engineer the state.
If the state change does not fulfill the core goal of the action proposal, accepted=false.
If the state change fulfills the core goal, accepted=true.

You will receive two types of input:

1. Engineering-layer assembled natural language states:
- subgraph_state_text
- recent_state_history_text
- transition_constraints_text
- support_context_facts

These are the primary content you need to read; they have already assembled nodes, states, and relationships into a human-understandable form.
support_context_facts contains background information determined by the previous module that should not be modeled as independent nodes, such as emails, web pages, office software, or screen content.

2. Raw structured data:
- node_reference
- focused_nodes
- focused_fact_edges
- interaction_frame

These are used to verify node_id, node types, existing factual relationships, and action relationship hints.
If the natural language state conflicts with the raw data, prioritize the raw structured data.
interaction_frame is merely a hint for action relationships; it is not a currently established fact, nor is it the result of a state update.

Meaning of accepted=true:
The next_state_delta makes the core goal of the action proposal valid in st+1.
For example, when opening a refrigerator, the refrigerator's door_state becomes open; when eating food, the food's status becomes consumed/visible=false, and the agent no longer holds the food.
If the food/milk/yogurt is held or visible=true in the current st, and the action proposal is to eat/drink it, then st+1 can change it to consumed/visible=false. This indicates that this small step has been successfully completed.
In this case, actual_event should state that the agent ate/drank it, not "discovered it was already eaten/drunk."

Meaning of accepted=false:
The current state does not support the fulfillment of the core goal.
You may still update the agent's attempted action, gaze, and posture, but do not change external objects to a completed state.
Only when the current st explicitly shows that the target object has already been consumed, disposed, discarded, visible=false, or key objects are missing, can you reject it because "it is already finished/does not exist."
When accepted=false, do not change temporary objects to consumed/disposed/discarded, do not change doors to open, do not change computers to logged_in, and do not change any external object to the completed state of the action proposal.

node_updates:
- human_agent can update posture, interaction_elements, interaction_method, gaze_target, worn_items, body_surface, text_to_motion_description.
- permanent_element can update physical_status, evolution_status, interaction_status, state_details.
- temporary_element can update status, visible, lifecycle, temperature, cooking_state, anchor_element_id, and other short-term states.

Do not write fact relations into node state.
For example, holding, inside, placed_on, sitting_on should be written in fact_edges_to_add/remove, not in state_patch.

text_to_motion_description must be an English body action description. Do not include person names, and do not use or/maybe/possibly.
Example: a person reaches forward, grasps the refrigerator handle, pulls the door open, and looks inside.

estimated_duration should be estimated based on the complete execution_result: if the actual_event and event_effects were to occur in the real world, how long would it roughly take?
Here you are estimating the entire event described by actual_event, not just the last hand movement or the duration of a single game animation.
You must choose from the following real-life benchmarks:
- Glancing, turning around, pressing a button once, picking up or putting down an item already at hand: 10s to 30s.
- Opening a cabinet door, checking a small area, handling a simple item: 1min to 3min.
- Tidying a small desktop, organizing a few miscellaneous items, simply wiping a surface: 3min to 5min.
- Organizing a complete piece of furniture, bed, sofa, carpet, or drawers with multiple items for classification and cleaning: 10min to 20min.
- Organizing bookshelves, wardrobes, large amounts of miscellaneous items, completely cleaning a local area: 15min to 30min.
- Cleaning an entire room or completing multiple consecutive household chores: 30min to 60min.
- Waiting: strictly follow the duration required in the action proposal.
If actual_event uses completed expressions like "tidied up, cleaned thoroughly, cleaning finished, classification complete," you must include all the real-world time required to achieve that result, not just estimate the final movement.
Output only the rough tiers provided below; do not output decimal seconds.

Return only JSON; do not explain your reasoning process.

Return format:
{{
  "accepted": true,
  "next_state_delta": {{
    "node_updates": [
      {{
        "node_id": "existing node_id",
        "node_type": "human_agent | permanent_element | temporary_element",
        "state_patch": {{}}
      }}
    ],
    "fact_edges_to_add": [
      {{
        "subject_id": "existing node_id",
        "relation": "english_snake_case_relation",
        "object_id": "existing node_id",
        "reason": "optional"
      }}
    ],
    "fact_edges_to_remove": [
      {{
        "subject_id": "existing node_id",
        "relation": "existing relation from focused_fact_edges",
        "object_id": "existing node_id",
        "reason": "optional"
      }}
    ],
    "state_reasoning_summary": "A one-sentence Chinese debug summary, explaining only why these state changes are reasonable"
  }},
  "execution_result": {{
    "actual_event": "Summarize what actually happened based on st -> st+1",
    "event_effects": ["Direct factual consequences"],
    "estimated_duration": "10s | 20s | 30s | 1min | 3min | 5min | 10min | 15min | 20min | 30min | 45min | 60min",
    "rejection": {{
      "reason": "",
      "state_statement": ""
    }}
  }}
}}

agent_name:
{variables.agent_name}

action_proposal:
{variables.action_proposal}

subgraph_state_text:
{variables.subgraph_state_text}

recent_state_history_text:
{variables.recent_state_history_text}

transition_constraints_text:
{variables.transition_constraints_text}

support_context_facts:
{json.dumps(variables.support_context_facts, ensure_ascii=False, indent=2)}

node_reference:
{json.dumps(variables.node_reference, ensure_ascii=False, indent=2)}

focused_nodes:
{json.dumps(variables.focused_nodes, ensure_ascii=False, indent=2)}

focused_fact_edges:
{json.dumps(variables.focused_fact_edges, ensure_ascii=False, indent=2)}

interaction_frame:
{json.dumps(variables.interaction_frame, ensure_ascii=False, indent=2)}
""".strip()

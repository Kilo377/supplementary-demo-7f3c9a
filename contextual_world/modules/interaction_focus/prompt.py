from __future__ import annotations

import json

from .types import WorldInteractionFocusVariables


def build_world_interaction_focus_prompt(variables: WorldInteractionFocusVariables) -> str:
    return f"""You are the world_interaction_focus module.

We are simulating a small step of action by a human agent in an environment. The environment is a graph, and human_agent, permanent_element, and temporary_element are all nodes.

Your task is to select the minimal set of nodes that might participate in interaction or undergo change in this step, based on the action proposal, node directory, and current factual relationships, and provide the interaction_frame as a hint for the action relationship.

Please prioritize reading graph_state_text, which is the human-readable input assembled by the engineering layer from the node directory and factual relationships.
If you need to verify node_id, node_type, or existing fact edges, you can refer to the raw JSON.
If there is a conflict between natural language and raw JSON, follow the raw JSON.

You do not judge whether the action was successful, nor do you judge how node states change.
You only select the nodes that need to be viewed and potentially updated in the subsequent state transition.
Do not infer state changes such as whether a door is open, food is finished, computer is on, or body is wet.

focused_node_ids must include the human_agent node.
focused_node_ids can include multiple nodes; for example, when the agent sits on a sofa and hugs a pillow, it should include the agent, sofa, and pillow.
If the action proposal does not explicitly state an object, but current factual relationships indicate that the agent is holding, sitting_on, looking_at, or using an object, you may also include that object in focused_node_ids.
interaction_frame is a relational hint for the action intent, not an established fact. The actual factual relationships are determined by the subsequent state transition.
Each relationship in interaction_frame must have exactly one subject_id and one object_id, both of which must be single node_id strings, not arrays.
If an action involves multiple objects, split it into multiple interaction_frames; do not put multiple node_ids into the same field.
relation_hint must also be a single English snake_case string, not an array.
Do not select the entire graph; only select the nodes most likely relevant to this step.

Return only JSON, without explaining the reasoning process.

Return format:
{{
  "focused_node_ids": ["existing node_id"],
  "interaction_frame": [
    {{
      "subject_id": "existing node_id",
      "relation_hint": "english_snake_case_hint",
      "object_id": "existing node_id",
      "reason": "Why this action relationship is relevant to this round's action"
    }}
  ],
  "focus_reason": "A Chinese sentence explaining why these nodes were selected"
}}

action_proposal:
{variables.action_proposal}

support_result:
{json.dumps(variables.support_result, ensure_ascii=False, indent=2)}

graph_state_text:
{variables.graph_state_text}

node_reference:
{json.dumps(variables.node_reference, ensure_ascii=False, indent=2)}

raw_nodes:
{json.dumps(variables.nodes, ensure_ascii=False, indent=2)}

raw_fact_edges:
{json.dumps(variables.fact_edges, ensure_ascii=False, indent=2)}
""".strip()

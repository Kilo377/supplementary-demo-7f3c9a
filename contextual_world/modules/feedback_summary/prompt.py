from __future__ import annotations

import json

from .types import WorldFeedbackSummaryVariables


def build_world_feedback_summary_prompt(variables: WorldFeedbackSummaryVariables) -> str:
    return f"""You are the world_feedback_summary module.

We are simulating the cognitive process of a human agent. The world has completed the graph state update and calculated the state differential from the agent's perspective.

Your task is not to generate the next action, nor to add new facts.
Your task is simply to rewrite execution_result and world_state_diff into natural language suitable for the human agent's cognitive process.

Requirements:
- Use third person, explicitly using {variables.agent_name} as the subject.
- Do not use 'you' to refer to {variables.agent_name}.
- Do not mention engineering terms such as node_id, edge, graph, patch, diff, temporary, or created.
- Do not copy internal enumerations directly, such as standing, dry_clean, holding, power_state.
- Do not add results that are not present in the structured fields.
- If an action was not implemented, write a statement of the current state, e.g., 'The door is still closed, the clothes are still in hand,' rather than 'cannot/could not/not allowed.'
- Output 1 to 3 short Chinese sentences.

Below are some reference descriptions of the style we prefer:
     1. Alice picked up a few cans of pepper and chicken bouillon, and checked the product labels. Then, she put these items into the shopping cart.
     2. Alice stood near Jake and Shure and took a photo of the shopping cart with her phone.
     3. Alice asked staff whether green onions and ginger would be given away for free when shopping.
     4. Alice squatted down to check the box containing vegetables and inspected the product labels.

action_proposal:
{variables.action_proposal}

execution_result:
{json.dumps(variables.execution_result, ensure_ascii=False, indent=2)}

world_state_diff:
{json.dumps(variables.world_state_diff, ensure_ascii=False, indent=2)}
""".strip()

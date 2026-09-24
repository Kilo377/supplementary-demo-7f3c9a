from __future__ import annotations

import json


def build_world_feedback_summary_prompt(
    *,
    agent_name: str,
    action_proposal: str,
    route: str,
    world_state_diff: dict | None = None,
) -> str:
    world_state_diff_json = json.dumps(world_state_diff or {}, ensure_ascii=False, indent=2)
    return f"""You are responsible for rewriting structured environment feedback into natural language for the human agent's cognitive process in the simulation environment.

The action proposal here is merely what {agent_name} just intended to do.
The state after it actually occurs should only be based on agent_centered_world_state_diff.

Your task is not to generate the next step's action, nor to supplement new facts.
Your task is simply to summarize these structured fields into a short paragraph of natural language, allowing {agent_name} to perceive this small step like a human:
- How their own body and action status are, e.g., standing/sitting, wearing clothes or not, whether the body is wet/dirty/foamy;
- The relationship between themselves and objects, e.g., what they are holding in hand, what they are looking at, what they are touching;
- What objects they have obtained or noticed, but do not say "appeared out of nowhere" or "was created";
- What objects were eaten, drunk, thrown away, consumed, or are no longer available;
- Which external objects were affected by themselves, and what their current states are, e.g., the microwave is working, the desktop is clean, the shower is flowing.

Do not output JSON.
Do not list bullets.
Use the third person, explicitly using {agent_name} as the subject.
Do not use "you" to refer to {agent_name}.
Do not mention engineering terms like node_id, edge, graph, route, patch, diff, temporary, created.
Do not directly copy internal enumerations like standing, dry_clean, temporary_created, holding; convert them into natural expressions.
Do not add results that are not in the structured fields.
Do not mechanically cover all fields; only write information directly related to this round's action, state changes, and next-step cognition.
Do not mention unchanged default body states, e.g., still standing, still wearing original clothes, body still clean, unless the action proposal or self_state_changes directly involve these fields.
If agent_centered_world_state_diff contains self_state_after, you can use it to confirm the current state; but do not write out all default fields.
If acquired_or_noticed_elements come from low-surprise completion, only express that {agent_name} obtained, saw, touched, or noticed the object; do not express that the world generated it.
If the action did not materialize, write it as a current state statement, e.g., "the door is still closed, the item is still in hand," not as "cannot/could not/allowed not/usually cannot."
Feedback for devices like washing machines, microwaves, and computers should be expressed roughly according to experience, e.g., "the washing machine started washing," "the microwave started heating," "the computer is already on"; do not list power_state, door_state, program, contains item by item.
If a certain type of information is empty, do not write that type.
Output 1 to 3 short Chinese sentences.

route:
{route}

action proposal:
{action_proposal}

agent_centered_world_state_diff:
{world_state_diff_json}
""".strip()

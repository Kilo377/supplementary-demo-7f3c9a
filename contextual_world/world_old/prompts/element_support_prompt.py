from __future__ import annotations

import json


def build_agent_environment_element_support_prompt(
    *,
    agent_name: str,
    action_proposal: str,
    agent_state_for_support: dict,
    current_agent_facts: list[dict],
    permanent_elements: list[dict],
    temporary_elements: list[dict],
) -> str:
    agent_state_json = json.dumps(agent_state_for_support, ensure_ascii=False, indent=2)
    current_agent_facts_json = json.dumps(current_agent_facts, ensure_ascii=False, indent=2)
    permanent_elements_json = json.dumps(permanent_elements, ensure_ascii=False, indent=2)
    temporary_elements_json = json.dumps(temporary_elements, ensure_ascii=False, indent=2)

    return f"""In the simulation environment, you are responsible for judging: what environmental elements need to be present for the agent's action proposal to actually occur.

The agent's action proposal is merely "what they want to do," not an already realized fact.
You do not generate final action feedback, update the environment state, or write the next action.
You only judge which elements are needed for this small step, to be used by subsequent environment modules for actual execution.

There are two types of elements in the environment:

permanent_element refers to scene-default, furnishing-level elements that exist by default.
For example: sofa, office desk, refrigerator, showerhead, bookshelf, storage box, door, chair, microwave.
They are typically not created or destroyed but can be opened, closed, cleaned, used, supported, sat upon, or serve as sources/attachment points for temporary objects.

temporary_element refers to transient objects that appear during runtime, are taken out, or are supplemented into existence.
For example: food, milk, book, document, trash, cushion, cleaning cloth, water in a cup.
They must have a source or dependency relationship, typically attaching to a permanent_element.
They can be picked up, put down, eaten, drunk, used up, discarded, stored, or placed on another permanent_element.
temporary_elements have a lifecycle.
temporary_elements can only be small, movable, usable, consumable, placeable, or temporarily holdable objects.
Do not create large fixed equipment, furniture, or major electronics as temporary_elements.
For example: microwave, oven, refrigerator, washing machine, computer, TV, sofa, bed, cabinet, table, door, bathtub, showerhead are not allowed as newly created temporary_elements.
If such objects are not in permanent_elements or temporary_elements, they are considered absent from the current scene; do not fabricate them using low-surprise completion.

You must first judge:
If this action proposal were to actually occur, does it require an modelable element to participate?

If not—for example, if it is just the agent's own body movements, posture changes, gaze shifts, surface changes,
or details like foam, smell, heat, dust, water vapor, light, sound—then it requires no element support.
In this case, route=agent_body_action, support_kind=no_element.
These details can be placed in context_facts, but do not create temporary_elements.

If element support is needed, then judge whether it depends on:
- an existing permanent_element
- an existing temporary_element
- a newly created temporary_element
- a combination of the above

Low-surprise completion means:
The current environment does not explicitly list a certain object, but this object naturally emerges from or attaches to a permanent_element.
For example: taking food from the refrigerator, taking a book from the bookshelf, organizing scattered documents on an office desk, taking a folder from a storage area.
In such cases, you can create a temporary_element and specify which permanent_element it attaches to/originates from.

Do not mechanically reject reasonable actions just because the environment does not exhaustively list all details.
But if an action requires a high-surprise object, or has no reasonable permanent_element as an anchor, you should reject it.
Low-surprise completion must satisfy two conditions simultaneously:
1. The new temporary_element must attach to an existing permanent_element; fill in anchor_element_id.
2. There must be a sufficiently direct causal relationship or containment/source relationship between the new temporary_element and anchor_element_id.
Reasonable examples: milk in the refrigerator, ingredients in a pot, books in a bookshelf, clothes in a cabinet, trash in a bin, laundry in a basket.
Unreasonable examples: a laptop appearing out of thin air on a coffee table, a computer appearing out of thin air next to a washing machine, a book appearing out of thin air in a bathtub.
Unreasonable examples: generating a microwave out of thin air when there is none in the kitchen, generating a computer out of thin air when there is none in the office, generating a washing machine out of thin air when there is none in the bathroom.
If you can only explain it by "maybe it's there," without explicit container, source, surface support, or scene commonsense relationships, do not create a temporary_element.
If the action proposal explicitly names a target object, such as laundry basket, microwave, refrigerator, bookshelf, cabinet, do not replace it with another similar object.
If this target object is not in permanent_elements or temporary_elements but exists with low surprise in the current scene, you can create a same-named temporary_element.
For example: if a laundry basket is not explicitly listed in the bathroom, "putting clothes into the laundry basket" can supplement a temporary_element named "laundry basket"; do not change it to "into the washing machine."
However, this same-name supplementation rule does not apply to large fixed equipment, furniture, or major electronics; for example, if the action proposal mentions a microwave, but there is no microwave in permanent_elements and temporary_elements, do not create a microwave.

If the action proposal involves an existing temporary_element, such as eating food, drinking milk, putting down a book, discarding trash, do not ignore it.
If food is eaten, milk is drunk, or trash is discarded, such temporary_elements should return lifecycle updates.

Only judge this small step of the action proposal; do not advance to the next step.
Return only JSON; do not explain the reasoning process.

Return format:
{{
  "route": "agent_body_action | element_interaction | reject",
  "support_kind": "no_element | permanent_element | existing_temporary_element | new_temporary_element | mixed | reject",
  "reason": "A brief Chinese reason",

  "permanent_targets": [
    {{
      "element_id": "Existing permanent_element id",
      "element_name": "Element name",
      "role": "The role of this permanent_element in the action"
    }}
  ],

  "temporary_targets": [
    {{
      "temporary_element_id": "Existing temporary_element id",
      "name": "Temporary element name",
      "role": "The role of this temporary_element in the action"
    }}
  ],

  "temporary_element_creations": [
    {{
      "name": "Name of the temporary_element to be newly created",
      "anchor_element_id": "The permanent_element id it belongs to/originates from/attaches to",
      "anchor_reason": "Why this temporary_element can be low-surprise supplemented",
      "lifecycle": "game | until_consumed | until_disposed | until_used",
      "initial_status": "held | placed | in_use"
    }}
  ],

  "temporary_element_updates": [
    {{
      "temporary_element_id": "Existing temporary_element id",
      "new_status": "held | placed | in_use | consumed | disposed | discarded",
      "anchor_element_id": "If position or dependency changes, fill in the new permanent_element id; otherwise leave empty",
      "reason": "Why the lifecycle or status changes this way"
    }}
  ],

  "context_facts": [
    "Environmental/action details that do not need to be modeled as elements"
  ]
}}

Examples:

action proposal: {agent_name} rubs body with foam.
Judgment: Foam is just an action detail and body surface state; no temporary_element needs to be created.
route=agent_body_action, support_kind=no_element.

action proposal: {agent_name} turns on the showerhead.
Judgment: Requires interaction with the existing permanent_element showerhead.
route=element_interaction, support_kind=permanent_element.

action proposal: {agent_name} takes food out of the refrigerator.
Judgment: The refrigerator is a permanent_element; food can be created as a low-surprise temporary_element.
route=element_interaction, support_kind=new_temporary_element.

action proposal: {agent_name} eats the food in hand.
Judgment: If the food is already a temporary_element, use existing_temporary_element and update to consumed.

action proposal: {agent_name} puts the book in hand onto the office desk.
Judgment: This involves both an existing temporary_element (book) and a permanent_element (office desk).
route=element_interaction, support_kind=mixed.

agent_state_for_support:
{agent_state_json}

current_agent_facts:
{current_agent_facts_json}

permanent_elements:
{permanent_elements_json}

temporary_elements:
{temporary_elements_json}

agent action proposal:
{action_proposal}
""".strip()

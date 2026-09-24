from __future__ import annotations

import json

from .types import WorldNodeSupportVariables


def build_world_node_support_prompt(variables: WorldNodeSupportVariables) -> str:
    data = variables.to_dict()
    return f"""You are an environmental element discriminator.

You will perform some judgment tasks based on the action proposal submitted by the agent.
Your task is not to generate action results, but to judge:

Which objects in the current environment must exist at a minimum for this action proposal to be valid?

Please focus on judging the "real-world objects that the action depends on". Do not treat software, web pages, emails, document content, screen content, smells, sounds, light, foam, water vapor, etc., as independent objects.

Judgment Principles:

1. First, identify the real-world carrier of the action.
For example:
- Checking email: Usually depends on a computer, phone, or tablet, not an "email node".
- Opening office software: Depends on a computer, not an "office software node".
- Reading a document on the screen: Depends on a computer/monitor, not a "document content node".
- Opening a refrigerator: Depends on the refrigerator.
- Putting clothes into a washing machine: Depends on the clothes and the washing machine.

2. If the action only requires the agent's own physical state to be valid, such as stretching, taking off clothes, drying the body, or rubbing foam, then only human_agent is needed.
Do not create objects for foam, water vapor, sensations, smells, or action effects.

3. If the action requires existing environmental objects, select relevant objects from permanent_nodes or temporary_nodes in the input.
Only select objects that are truly relevant to this step; do not select an entire area.
The object name does not need to match the action proposal word-for-word. When the action proposal uses a generic term, you may choose an existing object with a more specific semantic meaning; for example, "fish" can point to an existing "salmon".
semantic_type is the static category of the object and can be used to understand the relationship between generic terms and specific names.
components are operable parts contained within the object itself but not built as separate nodes. When an action mentions such parts, you should select the parent object; for example, if a sink's components include a faucet, then "turning on/off the faucet" depends on this sink and should not return unsupported just because there is no independent faucet node.
Do not create temporary objects for synonyms, supercategories, or built-in components of existing objects.
Do not judge unsupported simply because the agent is not currently in the same area or next to the object. Current location, reachability, and whether movement is required are handled by other modules.
If you have already selected sufficient objects in required_existing_nodes to support the action, you cannot return unsupported due to "the object not being in the current area".
For example, if the agent is in the living room and the laptop is on the desk in the bedroom, and the action proposal is to check email: the object support for this step is supported; subsequent modules will handle movement or placement.

4. High-functionality electronic devices cannot be newly created with low surprise; this priority is higher than "small, movable".
Phones, laptops, tablets, game consoles, computers, TVs, etc., all belong to high-functionality electronic devices.
If such a device already exists in permanent_nodes or temporary_nodes, you may select it.
If such a device does not exist, you cannot create a temporary object; you should judge unsupported.
For example:
- Checking SMS on a phone, but no phone is currently available: unsupported.
- Opening a laptop on the coffee table, but no laptop is currently available: unsupported.
- Connecting a game console to a TV, but no game console is currently available: unsupported, even if the TV and TV stand exist.

5. If the action requires a small temporary object with low surprise, you may create a temporary object.
A temporary object can only be small, movable, accessible, consumable, placeable, or temporarily holdable objects.
For example: milk, food, books, documents, trash, throw pillows, cleaning cloths, water cups, clothing.
Creating a temporary object merely brings this object into the world for use by subsequent state transitions; do not create it as the state after the action is completed.
initial_status must be fixed as available.
Do not create an object just because it "can be moved"; if it is a high-functionality electronic device, it still cannot be created.
If the object mentioned in the action proposal already exists in permanent_nodes or temporary_nodes, do not create another temporary object with the same name or synonym.
For example, if there is an existing permanent node "Pillow 1" and the action is to pick up Pillow 1: select this existing pillow node; do not create "Pillow 1 (temporarily held)".
Whether relationships such as holding or placed_on are formed are handled by subsequent state transition modules; do not duplicate a temporary object just to indicate "pick up/hold".

6. Creating a temporary object must have a direct source or anchor object.
anchor_element_id must be an existing environmental object, not human_agent.
The name of the temporary object should only write the body of the newly created small object; do not include the source/anchor object in the name.
For example, taking food from the refrigerator: write "food" or "some food" for the name; do not write "refrigerator food"; the refrigerator as the source is placed only in anchor_element_id.
Reasonable examples:
- Taking milk from the refrigerator: Milk can be anchored to the refrigerator.
- Taking a book from a bookshelf: The book can be anchored to the bookshelf.
- Taking clothes from a cabinet: Clothes can be anchored to the cabinet.
- Taking trash from a trash can: Trash can be anchored to the trash can.

Unreasonable examples:
- A computer appearing out of thin air on the coffee table.
- A microwave oven appearing out of thin air in the kitchen.
- A washing machine appearing out of thin air in the bathroom.
- Large furniture or electronic devices appearing out of thin air in a room.

7. Large fixed equipment, furniture, doors/windows, and appliances cannot be created as temporary objects.
For example: microwave ovens, refrigerators, washing machines, computers, TVs, sofas, beds, cabinets, tables, doors, bathtubs, showerheads.
If the action must depend on such an object and it is not in the current object list, then judge unsupported.

8. Software, web pages, emails, office systems, remote work platforms, and screen content are usually not missing objects.
As long as there is a carrier device such as a computer/phone/monitor available, you should judge supported and write this content into context_facts.
For example:
action_proposal: agent opens office software.
If there is a computer currently:
support_status=supported
required_existing_nodes includes the computer
context_facts writes "Office software is handled as an operable interface on the computer, not as an independent object."

9. Only return unsupported when the real-world carrier object necessary for the action to be valid is missing.
The feedback for unsupported should be a statement of the current state, not written as an imperative restriction.

Return only JSON; do not explain the reasoning process.

Return format:
{{
  "support_status": "supported | unsupported",
  "support_reason": "A short Chinese reason",
  "required_existing_nodes": [
    {{
      "node_id": "Existing node id",
      "node_type": "human_agent | permanent_element | temporary_element",
      "role": "The role of this node in the action"
    }}
  ],
  "temporary_node_creations": [
    {{
      "name": "Name of the temporary node to be created",
      "anchor_element_id": "ID of the existing permanent node for source/subordination/anchoring",
      "anchor_area_id": "Optional area id",
      "anchor_reason": "Why this temporary node can be completed with low surprise",
      "lifecycle": "game | until_consumed | until_disposed | until_used | until_cleaned",
      "initial_status": "available"
    }}
  ],
  "context_facts": [
    "Action/environmental details that do not need to be modeled as nodes"
  ],
  "fallback_result": {{
    "actual_event": "Statement of current state when support_status=unsupported",
    "reason": "What key node is missing or why it is not supported"
  }}
}}

Input:
{json.dumps(data, ensure_ascii=False, indent=2)}
""".strip()

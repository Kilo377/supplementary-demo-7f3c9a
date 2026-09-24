from __future__ import annotations

import json

from agent.habitual_controller.cue_memory import CueMemoryRecord

from .types import DemonstratedActionObservation


def build_demonstrated_habit_learning_prompt(
    *,
    agent_name: str,
    intent_text: str,
    observations: list[DemonstratedActionObservation],
    existing_persona_habits: list[CueMemoryRecord] | None = None,
    existing_demonstrated_habits: list[CueMemoryRecord] | None = None,
) -> str:
    payload = [observation.to_prompt_dict() for observation in observations]
    existing = [
        {
            "response_key": record.response_key,
            "response_text": record.response_text,
            "required_cue_ids": list(record.required_cue_ids),
        }
        for record in list(existing_persona_habits or [])
    ]
    demonstrated = [
        {
            "response_key": record.response_key,
            "response_text": record.response_text,
            "required_cue_ids": list(record.required_cue_ids),
            "source_intent": record.metadata.get("source_intent", ""),
        }
        for record in list(existing_demonstrated_habits or [])
    ]
    return f"""
There is a person named {agent_name}. To achieve the following goal, {agent_name} originally engaged in a series of conscious Goal-Directed behaviors:

{intent_text}

Below are the actions that were successfully executed. Each action is accompanied by the Context Features that actually existed before the action occurred:

{json.dumps(payload, ensure_ascii=False, indent=2)}

This character originally had the following Persona Habits:

{json.dumps(existing, ensure_ascii=False, indent=2)}

The character has learned the following Demonstrated Habits from past controlled tasks:

{json.dumps(demonstrated, ensure_ascii=False, indent=2)}

Please judge which of these actions are suitable to serve as habit responses quickly prepared by context in the same or similar stable tasks.

Judgment Principles:
- Select behaviors that are repeatable, executable, and can be stably cued by the pre-action context.
- The previous successful action can be an important sequential Cue.
- Location, visible objects, object states, and body states can also serve as Cues.
- Do not learn an action simply because it occurred once; reject one-off trade-offs, accidental remedies, and actions overly dependent on specific phrasing.
- selected_cue_ids must literally select only the feature_id that actually exists in the available_context_features of that observation.
- Typically, select one to three most informative Cues; do not treat all Features as necessary conditions.
- Do not select time_period Cues unless time itself is essential for the behavior.
- These behaviors were originally conscious Goal-Directed behaviors, so the resulting responses are all conscious.
- canonical_action_signature uses short, stable English snake_case to express the action and core object, e.g., place_salmon_on_counter.
- habitual_action is a single action that can be executed directly even after detaching from this task narrative, starting with {agent_name}; do not include "then continue", goal descriptions, or next-step plans.
- If an action is substantially identical to the response of a Persona Habit, canonical_action_signature must reuse its existing response_key to help the retrieval phase identify duplicates.
- If an action under the current goal is substantially identical to a Demonstrated Habit, you must reuse its response_key; this reinforces old associations rather than creating new behaviors.

Each observation must return a judgment. Return only JSON:

{{
  "judgments": [
    {{
      "observation_id": "original observation_id",
      "should_learn": true,
      "reason": "brief judgment reason",
      "selected_cue_ids": ["actual feature_id"],
      "canonical_action_signature": "snake_case",
      "habitual_action": "{agent_name} performs a single specific action",
      "context_text": "Describe the context formed by these Cues in one natural Chinese sentence"
    }}
  ]
}}
""".strip()

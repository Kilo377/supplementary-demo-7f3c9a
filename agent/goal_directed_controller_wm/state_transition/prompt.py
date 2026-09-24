from __future__ import annotations

from .types import StateTransitionPromptContext


def build_state_transition_prompt(
    *,
    agent_name: str,
    action_text: str,
    intent_text: str = "",
    spatial_belief_text: str = "",
    time_text: str = "",
    physical_state_text: str = "",
    internal_state_text: str = "",
    recent_experience_text: str = "",
) -> str:
    name = str(agent_name or "Agent").strip()
    action = str(action_text or "").strip()
    sections = [f"You are {name}."]
    _append_section(sections, "What you currently want to do:", intent_text)
    sections.append(f"You plan to do: {action}")
    _append_section(sections, "You remember your home is:", spatial_belief_text)
    _append_section(sections, "It is now:", time_text)
    _append_section(sections, f"{name}'s current physical state is:", physical_state_text)
    _append_section(sections, f"{name}'s current internal state is:", internal_state_text)
    _append_section(sections, f"{name} recently experienced:", recent_experience_text)
    sections.append(_transition_instructions(name))
    return "\n\n".join(sections).strip()


def build_state_transition_prompt_from_context(
    context: StateTransitionPromptContext,
) -> str:
    return build_state_transition_prompt(
        agent_name=context.agent_name,
        action_text=context.action_text,
        intent_text=context.intent_text,
        spatial_belief_text=context.spatial_belief_text,
        time_text=context.time_text,
        physical_state_text=context.physical_state_text,
        internal_state_text=context.internal_state_text,
        recent_experience_text=context.recent_experience_text,
    )


def _append_section(sections: list[str], heading: str, content: str) -> None:
    cleaned = str(content or "").strip()
    if cleaned:
        sections.append(f"{heading}\n\n{cleaned}")


def _transition_instructions(name: str) -> str:
    return f"""
Predict what your state will be in the next moment after completing this action.

Consider:

1. Infer which room {name} will be in after the action, and which objects are nearby, held, gazed at, or being interacted with.
2. Infer possible changes in posture, orientation, body movements, and interaction status.
3. Infer whether fatigue, stress, tension, hunger, thirst, and hygiene states change immediately.
4. Infer which elements in the environment might change.
5. Estimate the time elapsed based on the complete action process in the real world.
6. Only predict the immediate next state caused directly by this action; do not plan subsequent actions for {name}.
7. Judge whether the state after the action satisfies, approximately satisfies, or still does not satisfy the current Intent. This judgment can be lenient: if practically the goal is basically achieved, there is no need to exhaustively list all details.

Please strictly output the following JSON:

{{
  "transition_outcome": {{
    "status": "success | partial | failed",
    "description": "The most likely direct result of the action",
    "reason": "The main basis for this prediction"
  }},
  "intent_satisfaction": {{
    "status": "satisfied | approximately_satisfied | not_satisfied",
    "reason": "Why the state after the action satisfies, approximately satisfies, or still does not satisfy the current Intent"
  }},
  "elapsed_seconds": 0,
  "next_self_state": {{
    "area": "The room where {name} is after the action",
    "near_element": "The element near {name} after the action, empty if none",
    "posture": "{name}'s posture after the action",
    "facing_or_gaze": "{name}'s orientation or gaze target after the action",
    "holding": ["Items held by {name} after the action"],
    "interacting_with": ["Elements {name} is interacting with after the action"],
    "worn_items_change": "Changes in worn items, empty if none",
    "body_surface_change": "Changes on the body surface, empty if none"
  }},
  "internal_state_changes": {{
    "hunger": 0,
    "thirst": 0,
    "hygiene": 0,
    "stress": 0,
    "tension": 0,
    "fatigue": 0,
    "mental_change": "Immediate subjective psychological change, empty if none"
  }},
  "spatial_belief_updates": [
    {{
      "element": "The element that changed",
      "state_change": "The state change"
    }}
  ],
  "expected_feedback": "The result {name} expects to experience immediately after",
  "uncertainty": "Parts of the prediction that cannot be confirmed, empty if none"
}}
""".strip()

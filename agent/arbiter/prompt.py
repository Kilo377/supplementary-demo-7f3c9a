from __future__ import annotations

from agent.habitual_controller import PreparedHabitualResponse

from .conscious_habit_prompt import build_conscious_habit_arbiter_prompt
from .goal_conflict_prompt import build_goal_conflict_arbiter_prompt


def build_arbiter_prompt(
    *,
    agent_name: str,
    personality: str,
    intent_text: str,
    trigger_context_text: str,
    current_state_text: str,
    intuition_text: str,
    think_text: str,
    goal_directed_action: str,
    habitual_response: PreparedHabitualResponse,
    entry_reason: str,
) -> str:
    builder = (
        build_goal_conflict_arbiter_prompt
        if entry_reason == "goal_conflict"
        else build_conscious_habit_arbiter_prompt
    )
    return builder(
        agent_name=agent_name,
        personality=personality,
        trigger_context_text=trigger_context_text,
        habitual_response=habitual_response,
        intent_text=intent_text,
        intuition_text=intuition_text,
        think_text=think_text,
        goal_directed_action=goal_directed_action,
        current_state_text=current_state_text,
    )

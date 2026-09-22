from __future__ import annotations

from .types import WorldFeedbackSummaryVariables


def build_world_feedback_summary_variables(
    *,
    agent_name: str,
    action_proposal: str,
    execution_result: dict,
    world_state_diff: dict,
) -> WorldFeedbackSummaryVariables:
    return WorldFeedbackSummaryVariables(
        agent_name=agent_name,
        action_proposal=action_proposal,
        execution_result=dict(execution_result or {}),
        world_state_diff=dict(world_state_diff or {}),
    )

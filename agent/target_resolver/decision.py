from __future__ import annotations

from agent.action.types import ActionProposalResult
from core.action_types import AgentDecision

from .types import TargetResolution


def decision_from_target_resolution(
    action_proposal: ActionProposalResult,
    target: TargetResolution | None,
) -> AgentDecision:
    operation_target_id = (target.target_element_id if target is not None else "") or ""
    navigation_anchor_id = (
        (target.navigation_anchor_element_id if target is not None else "")
        or operation_target_id
    )
    secondary_target_id = (
        (target.secondary_target_element_id if target is not None else "") or ""
    )
    trace = target_resolution_trace(target)
    if target is not None and target.arrival_completes_action:
        return AgentDecision(
            action_type="move",
            reason=action_proposal.action_text,
            target_element_id=operation_target_id or navigation_anchor_id,
            navigation_target_element_id=navigation_anchor_id or operation_target_id,
            secondary_target_element_id=secondary_target_id,
            action_proposal_text=action_proposal.action_text,
            target_resolution_trace=trace,
        )

    # An operation remains an action regardless of distance. World owns any
    # positioning precondition and preserves this action while the actor moves.
    return AgentDecision(
        action_type="action",
        reason=action_proposal.action_text,
        target_element_id=operation_target_id or None,
        navigation_target_element_id=navigation_anchor_id or None,
        secondary_target_element_id=secondary_target_id or None,
        action_proposal_text=action_proposal.action_text,
        target_resolution_trace=trace,
    )


def target_resolution_trace(target: TargetResolution | None) -> dict:
    if target is None:
        return {}
    return {
        "operation_target_element_id": target.target_element_id,
        "operation_target_element_name": target.target_element_name,
        "navigation_anchor_element_id": target.navigation_anchor_element_id,
        "navigation_anchor_element_name": target.navigation_anchor_element_name,
        # Keep the old keys in traces consumed by existing debugging tools.
        "target_element_id": target.target_element_id,
        "target_element_name": target.target_element_name,
        "secondary_target_element_id": target.secondary_target_element_id,
        "arrival_completes_action": target.arrival_completes_action,
        "reason": target.reason,
        "prompt": target.prompt,
        "raw_output": target.raw_response,
        "provider_name": target.provider_name,
        "model": target.model,
        "duration_seconds": target.duration_seconds,
        "error": target.error,
    }

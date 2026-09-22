from __future__ import annotations

from dataclasses import dataclass, field

from .world_graph import WorldGraph


@dataclass
class WorldStateTransitionCheckIssue:
    issue_type: str
    node_id: str = ""
    field: str = ""
    previous_value: str = ""
    planned_value: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "issue_type": self.issue_type,
            "node_id": self.node_id,
            "field": self.field,
            "previous_value": self.previous_value,
            "planned_value": self.planned_value,
            "reason": self.reason,
        }


@dataclass
class WorldStateTransitionCheckResult:
    verdict: str = "accept"
    issues: list[WorldStateTransitionCheckIssue] = field(default_factory=list)
    repaired_transition: dict = field(default_factory=dict)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "issues": [issue.to_dict() for issue in self.issues],
            "repaired_transition": dict(self.repaired_transition),
            "raw_response": self.raw_response,
        }


def check_world_state_transition(
    *,
    previous_graph: WorldGraph,
    action_proposal: str,
    support_result: dict,
    planned_transition: dict,
    recent_transition_history: list[dict] | None = None,
) -> WorldStateTransitionCheckResult:
    """Check whether a planned world transition is temporally reasonable.

    This is intentionally a no-op scaffold for now. The final checker should
    inspect previous_graph, recent_transition_history, action_proposal, and the
    planned transition to catch unsupported state regressions such as a computer
    going from power_state=on to power_state=off without a shutdown, outage, or
    failure event.
    """
    _ = previous_graph
    _ = action_proposal
    _ = support_result
    _ = planned_transition
    _ = recent_transition_history
    # TODO: Implement temporal consistency checks before applying transition:
    # - Unsupported element state regression, e.g. power_state on -> off.
    # - Temporary element disappearing or moving without a causal action.
    # - Agent posture/holding/relation changes unsupported by the action.
    # - Contradiction between actual_event and state/relation patches.
    # - Optional LLM meta-check for subtle inconsistencies.
    return WorldStateTransitionCheckResult(verdict="accept")

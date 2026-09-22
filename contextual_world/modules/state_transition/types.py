from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldStateTransitionVariables:
    agent_name: str
    action_proposal: str
    subgraph_state_text: str
    recent_state_history_text: str
    transition_constraints_text: str
    support_context_facts: list[str]
    node_reference: dict[str, dict]
    focused_nodes: list[dict] = field(default_factory=list)
    focused_fact_edges: list[dict] = field(default_factory=list)
    interaction_frame: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "action_proposal": self.action_proposal,
            "subgraph_state_text": self.subgraph_state_text,
            "recent_state_history_text": self.recent_state_history_text,
            "transition_constraints_text": self.transition_constraints_text,
            "support_context_facts": list(self.support_context_facts),
            "node_reference": self.node_reference,
            "focused_nodes": list(self.focused_nodes),
            "focused_fact_edges": list(self.focused_fact_edges),
            "interaction_frame": list(self.interaction_frame),
        }


@dataclass
class WorldStateTransitionResult:
    accepted: bool = False
    next_state_delta: dict[str, Any] = field(default_factory=dict)
    execution_result: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "next_state_delta": dict(self.next_state_delta),
            "execution_result": dict(self.execution_result),
        }

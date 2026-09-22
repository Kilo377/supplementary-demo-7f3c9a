from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldInteractionFocusVariables:
    agent_name: str
    action_proposal: str
    support_result: dict[str, Any]
    graph_state_text: str
    node_reference: dict[str, dict]
    nodes: list[dict] = field(default_factory=list)
    fact_edges: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "action_proposal": self.action_proposal,
            "support_result": self.support_result,
            "graph_state_text": self.graph_state_text,
            "node_reference": self.node_reference,
            "nodes": list(self.nodes),
            "fact_edges": list(self.fact_edges),
        }


@dataclass
class WorldInteractionFocusResult:
    focused_node_ids: list[str] = field(default_factory=list)
    interaction_frame: list[dict] = field(default_factory=list)
    focus_reason: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "focused_node_ids": list(self.focused_node_ids),
            "interaction_frame": list(self.interaction_frame),
            "focus_reason": self.focus_reason,
        }

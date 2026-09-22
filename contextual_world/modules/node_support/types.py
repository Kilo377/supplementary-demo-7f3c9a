from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldNodeSupportVariables:
    agent_name: str
    action_proposal: str
    agent_node: dict[str, Any]
    agent_related_facts: list[dict] = field(default_factory=list)
    permanent_nodes: list[dict] = field(default_factory=list)
    temporary_nodes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "action_proposal": self.action_proposal,
            "agent_node": self.agent_node,
            "agent_related_facts": list(self.agent_related_facts),
            "permanent_nodes": list(self.permanent_nodes),
            "temporary_nodes": list(self.temporary_nodes),
        }


@dataclass
class WorldNodeSupportResult:
    support_status: str = "supported"
    support_reason: str = ""
    required_existing_nodes: list[dict] = field(default_factory=list)
    temporary_node_creations: list[dict] = field(default_factory=list)
    context_facts: list[str] = field(default_factory=list)
    fallback_result: dict = field(default_factory=dict)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "support_status": self.support_status,
            "support_reason": self.support_reason,
            "required_existing_nodes": list(self.required_existing_nodes),
            "temporary_node_creations": list(self.temporary_node_creations),
            "context_facts": list(self.context_facts),
            "fallback_result": dict(self.fallback_result),
        }

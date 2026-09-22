from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldFeedbackSummaryVariables:
    agent_name: str
    action_proposal: str
    execution_result: dict[str, Any]
    world_state_diff: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "action_proposal": self.action_proposal,
            "execution_result": dict(self.execution_result),
            "world_state_diff": dict(self.world_state_diff),
        }


@dataclass
class WorldFeedbackSummaryResult:
    perception_summary: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "perception_summary": self.perception_summary,
        }

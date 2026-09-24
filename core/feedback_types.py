from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvironmentFeedback:
    route: str = ""
    agent_state: dict = field(default_factory=dict)
    environment_changes: list[dict] = field(default_factory=list)
    current_elements: list[dict] = field(default_factory=list)
    perception_summary: str = ""
    agent_state_patch: dict = field(default_factory=dict)
    graph_transition_report: dict = field(default_factory=dict)
    world_state_diff: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "agent_state": dict(self.agent_state),
            "environment_changes": list(self.environment_changes),
            "current_elements": list(self.current_elements),
            "perception_summary": self.perception_summary,
            "agent_state_patch": dict(self.agent_state_patch),
            "graph_transition_report": dict(self.graph_transition_report),
            "world_state_diff": dict(self.world_state_diff),
        }

    def adapter_text(self) -> str:
        if self.perception_summary.strip():
            return self.perception_summary.strip()
        action_text = self.agent_state.get("text_to_motion_description", "")
        if isinstance(action_text, str) and action_text.strip():
            return action_text.strip()
        method = self.agent_state.get("interaction_method", "")
        if isinstance(method, str) and method.strip():
            return method.strip()
        if self.environment_changes:
            return "The environment state has been updated."
        return "This round of actions is over."

    def learned_fact_texts(self) -> list[str]:
        facts = []
        if self.agent_state:
            facts.append(f"agent_state: {self.agent_state}")
        for change in self.environment_changes:
            if isinstance(change, dict):
                facts.append(f"environment_change: {change}")
        for element in self.current_elements:
            if isinstance(element, dict):
                facts.append(f"current_element: {element}")
        return facts

from __future__ import annotations

from dataclasses import dataclass, field

from .feedback_types import EnvironmentFeedback


@dataclass
class AgentDecision:
    action_type: str
    reason: str
    target_area_id: str | None = None
    target_element_id: str | None = None
    navigation_target_element_id: str | None = None
    secondary_target_element_id: str | None = None
    action_proposal_text: str = ""
    intuition_route: str = ""
    intuition_thought: str = ""
    think_thought: str = ""
    think_conclusion: str = ""
    wait_duration: str = ""
    shadow_intuition_route: str = ""
    shadow_intuition_thought: str = ""
    habitual_response_key: str = ""
    habitual_gate_decision: str = ""
    habitual_gate_probability: float = 0.0
    habitual_gate_sample: float = 0.0
    habitual_retrieval_status: str = ""
    habitual_association_id: str = ""
    habitual_response_text: str = ""
    habitual_dimension: str = ""
    habitual_activation: float = 0.0
    habitual_awareness: str = ""
    habitual_conflict: bool = False
    habitual_conflict_reason: str = ""
    habitual_awareness_reason: str = ""
    arbiter_mode: str = ""
    arbiter_reason: str = ""
    computation_architecture: str = "intuition"
    world_model_trace: dict = field(default_factory=dict)
    emprical_gate_trace: dict = field(default_factory=dict)
    action_generation_trace: dict = field(default_factory=dict)
    target_resolution_trace: dict = field(default_factory=dict)
    cognitive_step_id: int | None = None
    intent_action_index: int = 0
    previous_execution_result: str = ""


@dataclass
class ActionResult:
    summary: str
    execution_kind: str = "action"
    execution_narration: str = ""
    resolved_action: str = ""
    error: str = ""
    estimated_duration: str = ""
    temporary_element_changes: list[dict] = field(default_factory=list)
    actor_state_update: dict = field(default_factory=dict)
    environment_feedback: EnvironmentFeedback | None = None
    execution_debug: dict = field(default_factory=dict)
    counts_as_intent_action: bool = True

    def feedback_text(self) -> str:
        if self.environment_feedback is not None:
            return self.environment_feedback.adapter_text()
        return self.execution_narration or self.summary

    def cognitive_feedback_text(self) -> str:
        if self.environment_feedback is not None:
            return self.environment_feedback.adapter_text()
        return self.feedback_text()

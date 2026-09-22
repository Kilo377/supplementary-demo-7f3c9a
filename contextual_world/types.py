from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.feedback_types import EnvironmentFeedback
from contextual_world.world_old.graph.world_graph_transition import WorldGraphTransitionReport


@dataclass
class ContextualWorldTraceStep:
    module_name: str
    variables: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    raw_output: str = ""
    parsed_output: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    provider_name: str = ""
    model: str = ""
    started_at: str = ""
    duration_seconds: float | None = None
    phase: str = ""
    error_type: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "module_name": self.module_name,
            "variables": self.variables,
            "prompt": self.prompt,
            "raw_output": self.raw_output,
            "parsed_output": self.parsed_output,
            "error": self.error,
            "provider_name": self.provider_name,
            "model": self.model,
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "phase": self.phase,
            "error_type": self.error_type,
            "attempts": list(self.attempts),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ContextualWorldTraceStep":
        return cls(
            module_name=str(value.get("module_name", "") or ""),
            variables=dict(value.get("variables", {}) or {}),
            prompt=str(value.get("prompt", "") or ""),
            raw_output=str(value.get("raw_output", "") or ""),
            parsed_output=dict(value.get("parsed_output", {}) or {}),
            error=str(value.get("error", "") or ""),
            provider_name=str(value.get("provider_name", "") or ""),
            model=str(value.get("model", "") or ""),
            started_at=str(value.get("started_at", "") or ""),
            duration_seconds=value.get("duration_seconds"),
            phase=str(value.get("phase", "") or ""),
            error_type=str(value.get("error_type", "") or ""),
            attempts=list(value.get("attempts", []) or []),
        )


class ContextualWorldModuleError(RuntimeError):
    def __init__(self, message: str, *, trace_step: ContextualWorldTraceStep) -> None:
        super().__init__(message)
        self.trace_step = trace_step


@dataclass
class ContextualWorldActionResult:
    action_proposal: str
    accepted: bool = False
    actual_event: str = ""
    estimated_duration: str = ""
    support_result: dict[str, Any] = field(default_factory=dict)
    focus_result: dict[str, Any] = field(default_factory=dict)
    transition_result: dict[str, Any] = field(default_factory=dict)
    transition_check: dict[str, Any] = field(default_factory=dict)
    transition_report: WorldGraphTransitionReport = field(default_factory=WorldGraphTransitionReport)
    world_state_diff: dict[str, Any] = field(default_factory=dict)
    feedback: EnvironmentFeedback | None = None
    trace: list[ContextualWorldTraceStep] = field(default_factory=list)

    def feedback_text(self) -> str:
        if self.feedback is not None:
            return self.feedback.adapter_text()
        return self.actual_event

    def to_dict(self) -> dict:
        return {
            "action_proposal": self.action_proposal,
            "accepted": self.accepted,
            "actual_event": self.actual_event,
            "estimated_duration": self.estimated_duration,
            "support_result": self.support_result,
            "focus_result": self.focus_result,
            "transition_result": self.transition_result,
            "transition_check": self.transition_check,
            "transition_report": self.transition_report.to_dict(),
            "world_state_diff": self.world_state_diff,
            "feedback": self.feedback.to_dict() if self.feedback is not None else {},
            "trace": [step.to_dict() for step in self.trace],
        }

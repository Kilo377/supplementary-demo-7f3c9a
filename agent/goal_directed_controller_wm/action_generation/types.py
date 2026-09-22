from __future__ import annotations

from dataclasses import dataclass, field

from agent.intuition import IntuitionResult


@dataclass(frozen=True)
class HabitualActionSupport:
    association_id: str
    response_key: str
    habit_strength: float
    activation: float
    awareness_type: str

    def to_dict(self) -> dict:
        return {
            "association_id": self.association_id,
            "response_key": self.response_key,
            "habit_strength": self.habit_strength,
            "activation": self.activation,
            "awareness_type": self.awareness_type,
        }


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    intuition_reply: IntuitionResult
    generated_by_goal: bool = False
    habitual_support: tuple[HabitualActionSupport, ...] = ()

    @property
    def action_text(self) -> str:
        return self.intuition_reply.thought

    @property
    def route(self) -> str:
        return self.intuition_reply.route

    @property
    def source(self) -> str:
        if self.generated_by_goal and self.habitual_support:
            return "goal_directed+habitual"
        if self.habitual_support:
            return "habitual"
        return "goal_directed"

    @property
    def maximum_habit_strength(self) -> float:
        return max((item.habit_strength for item in self.habitual_support), default=0.0)

    @property
    def maximum_habit_activation(self) -> float:
        return max((item.activation for item in self.habitual_support), default=0.0)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "action_text": self.action_text,
            "intuition_reply": {
                "route": self.intuition_reply.route,
                "thought": self.intuition_reply.thought,
                "target_area_id": self.intuition_reply.target_area_id,
                "target_area_name": self.intuition_reply.target_area_name,
                "chat_target": self.intuition_reply.chat_target,
                "wait_duration": self.intuition_reply.wait_duration,
            },
            "source": self.source,
            "generated_by_goal": self.generated_by_goal,
            "habitual_support": [item.to_dict() for item in self.habitual_support],
        }


@dataclass
class ActionGenerationResult:
    action_space: list[ActionCandidate] = field(default_factory=list)
    sampled_intuition_replies: list[IntuitionResult] = field(default_factory=list)
    provider_name: str = "ollama"
    model: str | None = None
    prompt: str = ""
    raw_response: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "action_space": [candidate.to_dict() for candidate in self.action_space],
            "sampled_intuition_replies": [
                {
                    "route": reply.route,
                    "thought": reply.thought,
                    "target_area_id": reply.target_area_id,
                    "target_area_name": reply.target_area_name,
                    "chat_target": reply.chat_target,
                    "wait_duration": reply.wait_duration,
                }
                for reply in self.sampled_intuition_replies
            ],
            "provider_name": self.provider_name,
            "model": self.model,
            "prompt": self.prompt,
            "error": self.error,
        }

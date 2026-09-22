from __future__ import annotations

from dataclasses import dataclass, field

from agent.habitual_controller.cue_memory import CueMemoryRecord


@dataclass(frozen=True)
class DemonstratedCueFeature:
    feature_id: str
    text: str
    feature_type: str = ""
    temporal_state: str = "current"

    def to_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "text": self.text,
            "feature_type": self.feature_type,
            "temporal_state": self.temporal_state,
        }


@dataclass(frozen=True)
class DemonstratedActionObservation:
    observation_id: str
    step_id: int
    action_text: str
    result_text: str
    area_name: str
    estimated_duration: str
    previous_action_text: str = ""
    cue_features: tuple[DemonstratedCueFeature, ...] = ()
    source_run_id: str = ""

    @property
    def available_cue_ids(self) -> set[str]:
        return {feature.feature_id for feature in self.cue_features}

    def to_prompt_dict(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "action": self.action_text,
            "world_result": self.result_text,
            "area": self.area_name,
            "previous_successful_action": self.previous_action_text,
            "available_context_features": [
                feature.to_dict() for feature in self.cue_features
            ],
        }


@dataclass(frozen=True)
class DemonstratedHabitRejection:
    observation_id: str
    action_text: str
    reason: str


@dataclass
class DemonstratedHabitLearningResult:
    records: list[CueMemoryRecord] = field(default_factory=list)
    rejections: list[DemonstratedHabitRejection] = field(default_factory=list)
    created_count: int = 0
    reinforced_count: int = 0
    prompt: str = ""
    raw_response: str = ""
    error: str = ""

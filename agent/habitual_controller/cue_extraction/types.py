from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ContextCue:
    cue_id: str
    cue_type: str
    value: str
    source: str
    text: str
    confidence: float = 1.0
    persistence: str = "current"
    occurred_at: datetime | None = None
    ttl_seconds: int = 0

    def to_dict(self) -> dict:
        return {
            "cue_id": self.cue_id,
            "cue_type": self.cue_type,
            "value": self.value,
            "source": self.source,
            "text": self.text,
            "confidence": self.confidence,
            "persistence": self.persistence,
            "occurred_at": self.occurred_at.isoformat() if self.occurred_at else None,
            "ttl_seconds": self.ttl_seconds,
        }


@dataclass(frozen=True)
class ContextFeature:
    feature_id: str
    feature_key: str
    feature_type: str
    modality: str
    value: str
    source: str
    text: str
    temporal_state: str
    first_observed_at: datetime
    observed_at: datetime
    duration_seconds: int = 0
    previous_value: str = ""
    confidence: float = 1.0
    activation: float = 1.0

    def to_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "feature_key": self.feature_key,
            "feature_type": self.feature_type,
            "modality": self.modality,
            "value": self.value,
            "source": self.source,
            "text": self.text,
            "temporal_state": self.temporal_state,
            "first_observed_at": self.first_observed_at.isoformat(),
            "observed_at": self.observed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "previous_value": self.previous_value,
            "confidence": self.confidence,
            "activation": round(self.activation, 4),
        }


@dataclass(frozen=True)
class PreviousExecutionInput:
    response_key: str
    response_text: str
    result_text: str
    succeeded: bool
    area_before: str = ""
    area_after: str = ""
    interacted_element_ids: tuple[str, ...] = ()
    interacted_element_names: tuple[str, ...] = ()


@dataclass
class ContextCueFrame:
    agent_name: str
    timestamp: datetime
    state_cues: list[ContextCue] = field(default_factory=list)
    event_cues: list[ContextCue] = field(default_factory=list)
    preceding_responses: list[str] = field(default_factory=list)

    @property
    def cues(self) -> list[ContextCue]:
        return [*self.state_cues, *self.event_cues]

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "state_cues": [cue.to_dict() for cue in self.state_cues],
            "event_cues": [cue.to_dict() for cue in self.event_cues],
            "preceding_responses": list(self.preceding_responses),
        }


@dataclass
class BufferedCue:
    cue: ContextCue
    activation: float
    age_seconds: int

    def to_dict(self) -> dict:
        return {
            "cue": self.cue.to_dict(),
            "activation": round(self.activation, 4),
            "age_seconds": self.age_seconds,
        }


@dataclass
class ContextCueBuffer:
    event_cues: dict[str, ContextCue] = field(default_factory=dict)
    current_state_cues: dict[str, ContextCue] = field(default_factory=dict)
    preceding_responses: list[str] = field(default_factory=list)
    max_preceding_responses: int = 5
    state_started_at: dict[str, datetime] = field(default_factory=dict)
    latest_features: list[ContextFeature] = field(default_factory=list)

    def update(self, frame: ContextCueFrame) -> list[BufferedCue]:
        previous_state_cues = dict(self.current_state_cues)
        previous_by_key = {
            _feature_key(cue): cue
            for cue in previous_state_cues.values()
        }
        state_features: list[ContextFeature] = []
        next_started_at: dict[str, datetime] = {}
        current_feature_keys = {_feature_key(cue) for cue in frame.state_cues}
        for cue in frame.state_cues:
            key = _feature_key(cue)
            previous = previous_by_key.get(key)
            if cue.cue_id in previous_state_cues:
                temporal_state = "maintained"
                first_observed_at = self.state_started_at.get(cue.cue_id, cue.occurred_at or frame.timestamp)
                previous_value = ""
            elif previous is not None:
                temporal_state = "changed"
                first_observed_at = cue.occurred_at or frame.timestamp
                previous_value = previous.value
            else:
                temporal_state = "onset"
                first_observed_at = cue.occurred_at or frame.timestamp
                previous_value = ""
            next_started_at[cue.cue_id] = first_observed_at
            state_features.append(
                _context_feature(
                    cue,
                    now=frame.timestamp,
                    temporal_state=temporal_state,
                    first_observed_at=first_observed_at,
                    previous_value=previous_value,
                    activation=cue.confidence,
                )
            )
        for previous in previous_state_cues.values():
            previous_key = _feature_key(previous)
            if previous_key in current_feature_keys:
                continue
            first_observed_at = self.state_started_at.get(
                previous.cue_id,
                previous.occurred_at or frame.timestamp,
            )
            state_features.append(
                _offset_feature(
                    previous,
                    now=frame.timestamp,
                    first_observed_at=first_observed_at,
                )
            )

        self.current_state_cues = {cue.cue_id: cue for cue in frame.state_cues}
        self.state_started_at = next_started_at
        incoming_event_ids = {cue.cue_id for cue in frame.event_cues}
        for cue in frame.event_cues:
            self.event_cues[cue.cue_id] = cue
        for response in frame.preceding_responses:
            if response and (not self.preceding_responses or self.preceding_responses[-1] != response):
                self.preceding_responses.append(response)
        self.preceding_responses = self.preceding_responses[-self.max_preceding_responses :]
        self._remove_expired(frame.timestamp)
        active = self.active(frame.timestamp)
        event_features = [
            _context_feature(
                item.cue,
                now=frame.timestamp,
                temporal_state="onset" if item.cue.cue_id in incoming_event_ids else "recent",
                first_observed_at=item.cue.occurred_at or frame.timestamp,
                previous_value="",
                activation=item.activation,
            )
            for item in active
            if item.cue.cue_id in self.event_cues
        ]
        self.latest_features = [*state_features, *event_features]
        return active

    def active(self, now: datetime) -> list[BufferedCue]:
        active = [BufferedCue(cue=cue, activation=cue.confidence, age_seconds=0) for cue in self.current_state_cues.values()]
        for cue in self.event_cues.values():
            age_seconds = max(0, int((now - (cue.occurred_at or now)).total_seconds()))
            ttl = max(1, cue.ttl_seconds)
            activation = cue.confidence * max(0.0, 1.0 - age_seconds / ttl)
            if activation > 0:
                active.append(BufferedCue(cue=cue, activation=activation, age_seconds=age_seconds))
        return active

    def clear(self) -> None:
        self.event_cues.clear()
        self.current_state_cues.clear()
        self.preceding_responses.clear()
        self.state_started_at.clear()
        self.latest_features.clear()

    def _remove_expired(self, now: datetime) -> None:
        self.event_cues = {
            cue_id: cue
            for cue_id, cue in self.event_cues.items()
            if int((now - (cue.occurred_at or now)).total_seconds()) < max(1, cue.ttl_seconds)
        }


@dataclass
class CueExtractionResult:
    frame: ContextCueFrame
    active_cues: list[BufferedCue]
    prompt_text: str
    features: list[ContextFeature] = field(default_factory=list)

    @property
    def structured_feature_ids(self) -> list[str]:
        return [feature.feature_id for feature in self.features]

    @property
    def natural_context_text(self) -> str:
        return self.prompt_text

    def to_dict(self) -> dict:
        return {
            "frame": self.frame.to_dict(),
            "active_cues": [cue.to_dict() for cue in self.active_cues],
            "prompt_text": self.prompt_text,
            "natural_context_text": self.natural_context_text,
            "features": [feature.to_dict() for feature in self.features],
        }


def _context_feature(
    cue: ContextCue,
    *,
    now: datetime,
    temporal_state: str,
    first_observed_at: datetime,
    previous_value: str,
    activation: float,
) -> ContextFeature:
    return ContextFeature(
        feature_id=cue.cue_id,
        feature_key=_feature_key(cue),
        feature_type=cue.cue_type,
        modality=_feature_modality(cue),
        value=cue.value,
        source=cue.source,
        text=cue.text,
        temporal_state=temporal_state,
        first_observed_at=first_observed_at,
        observed_at=now,
        duration_seconds=max(0, int((now - first_observed_at).total_seconds())),
        previous_value=previous_value,
        confidence=cue.confidence,
        activation=activation,
    )


def _offset_feature(
    cue: ContextCue,
    *,
    now: datetime,
    first_observed_at: datetime,
) -> ContextFeature:
    return ContextFeature(
        feature_id=f"offset:{cue.cue_id}",
        feature_key=f"offset:{_feature_key(cue)}",
        feature_type=f"{cue.cue_type}_offset",
        modality=_feature_modality(cue),
        value=cue.value,
        source=cue.source,
        text=f"不再感知到：{cue.text or cue.value}",
        temporal_state="offset",
        first_observed_at=first_observed_at,
        observed_at=now,
        duration_seconds=max(0, int((now - first_observed_at).total_seconds())),
        previous_value=cue.value,
        confidence=cue.confidence,
        activation=cue.confidence,
    )


def _feature_key(cue: ContextCue) -> str:
    if cue.cue_type in {"location", "time_period", "posture"}:
        return cue.cue_type
    if cue.cue_type == "feeling":
        return f"feeling:{cue.source}"
    if cue.cue_type in {"visual_state", "visual_type_state"}:
        return cue.cue_id.rsplit(":", 1)[0]
    return cue.cue_id


def _feature_modality(cue: ContextCue) -> str:
    if cue.cue_type.startswith("visual"):
        return "exteroceptive_visual"
    if cue.cue_type == "feeling":
        return "interoceptive" if cue.source.startswith(("physiological_state:", "internal_state:")) else "affective"
    if cue.cue_type in {"posture", "wearing"}:
        return "proprioceptive"
    if cue.cue_type in {"completed", "failed", "entered_area"}:
        return "event"
    if cue.cue_type == "time_period":
        return "temporal"
    if cue.cue_type == "location":
        return "spatial"
    return "contextual"

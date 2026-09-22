from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import re


DEFAULT_HABIT_STRENGTH_THRESHOLD = 0.70


@dataclass
class CueMemoryRecord:
    association_id: str
    cue_ids: list[str]
    required_cue_ids: list[str]
    cue_prefixes: list[str]
    response_key: str
    response_text: str
    habit_strength: float
    context_text: str = ""
    source_dimension: str = "external_context"
    awareness_type: str = "context_dependent"
    visual_element_ids: list[str] = field(default_factory=list)
    visual_semantic_keys: list[str] = field(default_factory=list)
    estimated_duration: str = ""
    required_body_resources: list[str] = field(default_factory=list)
    cooldown_seconds: int = 0
    reinforcement_count: int = 1
    last_reinforced_at: datetime | None = None
    context_embedding: list[float] = field(default_factory=list)
    dimension_embeddings: dict[str, list[float]] = field(default_factory=dict)
    dimension_embedding_texts: dict[str, str] = field(default_factory=dict)
    embedding_model: str = ""
    embedding_dimensions: int = 0
    embedding_version: str = "cue-memory-v1"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.habit_strength = min(1.0, max(0.0, float(self.habit_strength)))

    def is_established(self, threshold: float) -> bool:
        return self.habit_strength >= threshold

    def canonical_association_key(self) -> str:
        configured = str(self.metadata.get("canonical_association_key", "") or "").strip()
        if configured:
            return configured
        cue_signature = "|".join(sorted({
            *[str(item).strip() for item in self.required_cue_ids if str(item).strip()],
            *[f"prefix:{str(item).strip()}" for item in self.cue_prefixes if str(item).strip()],
        }))
        response_signature = _canonical_text(self.response_key or self.response_text)
        return hashlib.sha256(
            f"{cue_signature}=>{response_signature}".encode("utf-8")
        ).hexdigest()[:24]

    def to_dict(self) -> dict:
        return {
            "association_id": self.association_id,
            "cue_ids": list(self.cue_ids),
            "required_cue_ids": list(self.required_cue_ids),
            "cue_prefixes": list(self.cue_prefixes),
            "context_text": self.context_text,
            "response_key": self.response_key,
            "response_text": self.response_text,
            "habit_strength": self.habit_strength,
            "source_dimension": self.source_dimension,
            "awareness_type": self.awareness_type,
            "visual_element_ids": list(self.visual_element_ids),
            "visual_semantic_keys": list(self.visual_semantic_keys),
            "estimated_duration": self.estimated_duration,
            "required_body_resources": list(self.required_body_resources),
            "cooldown_seconds": self.cooldown_seconds,
            "reinforcement_count": self.reinforcement_count,
            "last_reinforced_at": self.last_reinforced_at.isoformat() if self.last_reinforced_at else None,
            "context_embedding": list(self.context_embedding),
            "dimension_embeddings": {
                str(key): list(value)
                for key, value in self.dimension_embeddings.items()
            },
            "dimension_embedding_texts": dict(self.dimension_embedding_texts),
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_version": self.embedding_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict, *, agent_name: str = "Agent") -> "CueMemoryRecord":
        metadata = dict(data.get("metadata", {}) or {})
        required_cue_ids = _string_list(data.get("required_cue_ids"))
        cue_ids = _string_list(data.get("cue_ids")) or list(required_cue_ids)
        return cls(
            association_id=str(data.get("association_id", "") or "").strip(),
            cue_ids=cue_ids,
            required_cue_ids=required_cue_ids,
            cue_prefixes=_string_list(data.get("cue_prefixes")),
            context_text=str(data.get("context_text", "") or "").format(agent_name=agent_name).strip(),
            response_key=str(data.get("response_key", "") or "").strip(),
            response_text=str(data.get("response_text", "") or "").format(agent_name=agent_name).strip(),
            habit_strength=float(data.get("habit_strength", 0.0) or 0.0),
            source_dimension=str(data.get("source_dimension", metadata.get("source_dimension", "external_context")) or "external_context").strip(),
            awareness_type=_awareness_type(data.get("awareness_type", metadata.get("awareness_type"))),
            visual_element_ids=_string_list(data.get("visual_element_ids")),
            visual_semantic_keys=_string_list(data.get("visual_semantic_keys")),
            estimated_duration=str(data.get("estimated_duration", metadata.get("estimated_duration", "")) or "").strip(),
            required_body_resources=_string_list(data.get("required_body_resources", metadata.get("required_body_resources"))),
            cooldown_seconds=max(0, int(data.get("cooldown_seconds", metadata.get("cooldown_seconds", 0)) or 0)),
            reinforcement_count=max(0, int(data.get("reinforcement_count", 1) or 0)),
            last_reinforced_at=_datetime_value(data.get("last_reinforced_at")),
            context_embedding=[float(item) for item in (data.get("context_embedding") or [])],
            dimension_embeddings={
                str(key): [float(item) for item in value]
                for key, value in (data.get("dimension_embeddings") or {}).items()
                if isinstance(value, list)
            },
            dimension_embedding_texts={
                str(key): str(value)
                for key, value in (data.get("dimension_embedding_texts") or {}).items()
            },
            embedding_model=str(data.get("embedding_model", "") or ""),
            embedding_dimensions=int(data.get("embedding_dimensions", 0) or 0),
            embedding_version=str(data.get("embedding_version", "cue-memory-v1") or "cue-memory-v1"),
            metadata=metadata,
        )


@dataclass
class CueMemory:
    records: list[CueMemoryRecord] = field(default_factory=list)
    habit_strength_threshold: float = DEFAULT_HABIT_STRENGTH_THRESHOLD

    def __post_init__(self) -> None:
        self.habit_strength_threshold = min(1.0, max(0.0, float(self.habit_strength_threshold)))

    @property
    def established_habits(self) -> list[CueMemoryRecord]:
        return [
            record
            for record in self.records
            if record.is_established(self.habit_strength_threshold)
        ]

    @property
    def candidates(self) -> list[CueMemoryRecord]:
        return [
            record
            for record in self.records
            if not record.is_established(self.habit_strength_threshold)
        ]

    def remember(self, record: CueMemoryRecord) -> None:
        self.records.append(record)

    def upsert_established(self, record: CueMemoryRecord) -> tuple[CueMemoryRecord, bool]:
        key = record.canonical_association_key()
        for existing in self.records:
            if existing.canonical_association_key() != key:
                continue
            existing.habit_strength = max(existing.habit_strength, record.habit_strength)
            existing.reinforcement_count = max(1, existing.reinforcement_count) + 1
            existing.last_reinforced_at = record.last_reinforced_at or datetime.now()
            source_runs = list(existing.metadata.get("source_runs", []) or [])
            for run_id in list(record.metadata.get("source_runs", []) or []):
                if run_id and run_id not in source_runs:
                    source_runs.append(run_id)
            existing.metadata["source_runs"] = source_runs
            return existing, False
        self.records.append(record)
        return record, True

    def clear(self) -> None:
        self.records.clear()

    # TODO: Habit learning updates candidate strength from repeated S-R episodes.


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _awareness_type(value) -> str:
    cleaned = str(value or "context_dependent").strip().lower()
    if cleaned in {"unconscious", "conscious", "context_dependent"}:
        return cleaned
    return "context_dependent"


def _datetime_value(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    return datetime.fromisoformat(text) if text else None


def _canonical_text(value: str) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)

from __future__ import annotations

from dataclasses import dataclass, field

from agent.habitual_controller.cue_memory import CueMemoryRecord


@dataclass(frozen=True)
class CueRetrievalCandidate:
    record: CueMemoryRecord
    retrieval_method: str
    matching_cue_ids: tuple[str, ...]
    similarity: float
    retrieval_score: float
    familiarity_status: str = "unchecked"
    evidence: dict = field(default_factory=dict)


@dataclass
class CueRetrievalResult:
    candidates: list[CueRetrievalCandidate] = field(default_factory=list)
    retrieval_method: str = "none"
    query_embedding: list[float] = field(default_factory=list)
    similarity_threshold: float = 0.0
    habit_strength_threshold: float = 0.0
    status: str = "no_match"

    @property
    def has_candidates(self) -> bool:
        return bool(self.candidates)

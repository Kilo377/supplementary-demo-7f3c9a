from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class MemoryEncodingResult:
    recallable: bool
    probability: float
    sample: float | None


def decide_short_time_memory_encoding(
    *,
    unconscious_habit: bool,
    probability: float = 0.2,
    sample: float | None = None,
) -> MemoryEncodingResult:
    if not unconscious_habit:
        return MemoryEncodingResult(recallable=True, probability=1.0, sample=None)
    probability = min(1.0, max(0.0, float(probability)))
    sampled = random.random() if sample is None else min(1.0, max(0.0, float(sample)))
    return MemoryEncodingResult(
        recallable=sampled < probability,
        probability=probability,
        sample=sampled,
    )

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class HabitualRetrievalGateResult:
    should_retrieve: bool
    probability: float
    sample: float


def decide_habitual_retrieval(
    probability: float = 0.9,
    *,
    sample: float | None = None,
) -> HabitualRetrievalGateResult:
    normalized = min(1.0, max(0.0, float(probability)))
    if normalized <= 0.0:
        value = 1.0
    elif normalized >= 1.0:
        value = 0.0
    else:
        value = random.random() if sample is None else min(1.0, max(0.0, float(sample)))
    return HabitualRetrievalGateResult(
        should_retrieve=value < normalized,
        probability=normalized,
        sample=value,
    )

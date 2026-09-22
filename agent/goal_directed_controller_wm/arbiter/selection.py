from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .types import ActionSelectionResult, ArbitrationEntry


SELECTION_METHODS = ("greedy", "softmax", "epsilon_greedy", "random")


def select_action(
    entries: Sequence[ArbitrationEntry],
    *,
    method: str = "greedy",
    temperature: float = 0.2,
    epsilon: float = 0.1,
    rng: random.Random | None = None,
) -> ActionSelectionResult:
    if not entries:
        raise ValueError("Cannot select an action from an empty arbitration set.")
    normalized_method = str(method or "greedy").strip().lower()
    if normalized_method not in SELECTION_METHODS:
        raise ValueError(
            f"Unknown selection method: {method}. "
            f"Expected one of {', '.join(SELECTION_METHODS)}."
        )
    random_source = rng or random.Random()

    if normalized_method == "greedy":
        selected = _greedy_entry(entries)
        probabilities = {
            entry.action_id: 1.0 if entry.action_id == selected.action_id else 0.0
            for entry in entries
        }
    elif normalized_method == "softmax":
        probabilities = _softmax_probabilities(entries, temperature=temperature)
        selected = _sample_entry(entries, probabilities, rng=random_source)
    elif normalized_method == "epsilon_greedy":
        probabilities = _epsilon_greedy_probabilities(entries, epsilon=epsilon)
        selected = _sample_entry(entries, probabilities, rng=random_source)
    else:
        probability = 1.0 / len(entries)
        probabilities = {entry.action_id: probability for entry in entries}
        selected = _sample_entry(entries, probabilities, rng=random_source)

    return ActionSelectionResult(
        selected_action_id=selected.action_id,
        method=normalized_method,
        probabilities=probabilities,
    )


def _greedy_entry(entries: Sequence[ArbitrationEntry]) -> ArbitrationEntry:
    return max(
        entries,
        key=lambda entry: (
            entry.decision_value,
            entry.normalized_reward,
            entry.habit_strength,
            _stable_id_tiebreak(entry.action_id),
        ),
    )


def _softmax_probabilities(
    entries: Sequence[ArbitrationEntry],
    *,
    temperature: float,
) -> dict[str, float]:
    safe_temperature = max(1e-6, float(temperature))
    maximum = max(entry.decision_value for entry in entries)
    weights = [
        math.exp((entry.decision_value - maximum) / safe_temperature)
        for entry in entries
    ]
    total = sum(weights)
    return {
        entry.action_id: weight / total
        for entry, weight in zip(entries, weights)
    }


def _epsilon_greedy_probabilities(
    entries: Sequence[ArbitrationEntry],
    *,
    epsilon: float,
) -> dict[str, float]:
    safe_epsilon = min(1.0, max(0.0, float(epsilon)))
    greedy = _greedy_entry(entries)
    random_probability = safe_epsilon / len(entries)
    return {
        entry.action_id: random_probability
        + (1.0 - safe_epsilon if entry.action_id == greedy.action_id else 0.0)
        for entry in entries
    }


def _sample_entry(
    entries: Sequence[ArbitrationEntry],
    probabilities: dict[str, float],
    *,
    rng: random.Random,
) -> ArbitrationEntry:
    sample = rng.random()
    cumulative = 0.0
    for entry in entries:
        cumulative += probabilities.get(entry.action_id, 0.0)
        if sample <= cumulative:
            return entry
    return entries[-1]


def _stable_id_tiebreak(action_id: str) -> float:
    try:
        return -float(action_id)
    except ValueError:
        return 0.0

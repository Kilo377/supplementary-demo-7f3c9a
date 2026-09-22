from __future__ import annotations

from dataclasses import asdict, dataclass
import random


BETAS = {"stress": 0.45, "depletion": 0.38, "cognitive_load": 0.40}


@dataclass(frozen=True)
class GateResult:
    states: dict[str, float]
    factor_suppression: dict[str, float]
    suppression_probability: float
    sample: float | None
    route: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_gate(internal_state, *, has_habit: bool, conscious: bool,
                  enabled: bool = True, overrides: dict | None = None,
                  rng: random.Random | None = None) -> GateResult:
    states = {
        key: min(10.0, max(0.0, float((overrides or {}).get(
            key, getattr(internal_state, key, 3)))))
        for key in BETAS
    }
    factors = {key: min(1.0, max(0.0, beta * (states[key] - 3) / 4))
               for key, beta in BETAS.items()}
    probability = max(factors.values())
    sample = None
    if not has_habit:
        route, reason = "goal_directed", "no_habit"
    elif not conscious:
        route, reason = "habitual_direct", "unconscious"
    elif not enabled:
        route, reason = "competition", "disabled"
    else:
        sample = (rng or random).random()
        suppressed = sample < probability
        route = "habitual_direct" if suppressed else "competition"
        reason = "suppressed" if suppressed else "passed"
    return GateResult(states, factors, probability, sample, route, reason)

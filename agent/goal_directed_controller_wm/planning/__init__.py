from .beam import beam_search_state_transitions
from .greedy import greedy_search_state_transitions
from .mcts import mcts_state_transitions

PLANNING_STRATEGIES = ("greedy", "beam", "mcts")


def normalize_planning_strategy(value: str) -> str:
    normalized = str(value or "greedy").strip().lower()
    if normalized not in PLANNING_STRATEGIES:
        raise ValueError(
            f"Unknown planning strategy: {value}. "
            f"Expected one of {', '.join(PLANNING_STRATEGIES)}."
        )
    return normalized


__all__ = [
    "PLANNING_STRATEGIES",
    "beam_search_state_transitions",
    "greedy_search_state_transitions",
    "mcts_state_transitions",
    "normalize_planning_strategy",
]

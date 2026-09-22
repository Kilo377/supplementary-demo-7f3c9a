from __future__ import annotations

import random
from collections.abc import Sequence

from agent.goal_directed_controller_wm.action_generation import ActionCandidate
from agent.goal_directed_controller_wm.reward_evaluation import RewardEvaluationResult

from .selection import select_action
from .types import ArbitrationEntry, ArbitrationResult


ARBITER_MODES = ("weighted", "random_controller")
DEFAULT_ARBITER_MODE = "weighted"
RANDOM_CONTROLLER_HABIT_PROBABILITY = 0.5


def arbitrate_actions(
    action_candidates: Sequence[ActionCandidate],
    reward_evaluation: RewardEvaluationResult,
    *,
    degree_of_model_based_control: float,
    selection_method: str = "greedy",
    temperature: float = 0.2,
    epsilon: float = 0.1,
    rng: random.Random | None = None,
    arbiter_mode: str = DEFAULT_ARBITER_MODE,
) -> ArbitrationResult:
    if reward_evaluation.error:
        raise ValueError(
            "Cannot arbitrate actions because Reward Evaluation failed: "
            f"{reward_evaluation.error}"
        )
    degree = min(1.0, max(0.0, float(degree_of_model_based_control)))
    has_active_habit = any(
        candidate.maximum_habit_strength > 0.0
        for candidate in action_candidates
    )
    mode = "model_based_habitual" if has_active_habit else "model_based_only"
    entries = tuple(
        _build_entry(
            action_id=str(index),
            candidate=candidate,
            reward_evaluation=reward_evaluation,
            degree_of_model_based_control=degree,
            use_habitual_weight=has_active_habit,
        )
        for index, candidate in enumerate(action_candidates, start=1)
    )
    normalized_mode = normalize_arbiter_mode(arbiter_mode)
    if normalized_mode == "random_controller" and has_active_habit:
        selection, selected_controller = _select_random_controller(
            action_candidates,
            entries,
            rng=rng,
        )
        mode = f"random_controller_{selected_controller}"
    else:
        selection = select_action(
            entries,
            method=selection_method,
            temperature=temperature,
            epsilon=epsilon,
            rng=rng,
        )
    return ArbitrationResult(
        degree_of_model_based_control=degree,
        mode=mode,
        entries=entries,
        selection=selection,
    )


def normalize_arbiter_mode(value: str) -> str:
    normalized = str(value or DEFAULT_ARBITER_MODE).strip().lower()
    if normalized not in ARBITER_MODES:
        raise ValueError(
            f"Unknown arbiter mode: {value}. Expected one of {', '.join(ARBITER_MODES)}."
        )
    return normalized


def _select_random_controller(
    action_candidates: Sequence[ActionCandidate],
    entries: tuple[ArbitrationEntry, ...],
    *,
    rng: random.Random | None,
):
    from .types import ActionSelectionResult

    random_source = rng or random.Random()
    paired = list(zip(action_candidates, entries))
    goal_choices = [pair for pair in paired if pair[0].generated_by_goal]
    habitual_choices = [pair for pair in paired if pair[0].habitual_support]
    best_goal = max(
        goal_choices,
        key=lambda pair: (pair[1].normalized_reward, -int(pair[1].action_id)),
        default=None,
    )
    best_habit = max(
        habitual_choices,
        key=lambda pair: (
            pair[0].maximum_habit_strength,
            pair[0].maximum_habit_activation,
            -int(pair[1].action_id),
        ),
        default=None,
    )
    if best_goal is None and best_habit is None:
        raise ValueError("Random-controller arbitration has no selectable action.")
    if best_goal is None:
        selected = best_habit
        selected_controller = "habitual"
    elif best_habit is None:
        selected = best_goal
        selected_controller = "goal_directed"
    elif random_source.random() < RANDOM_CONTROLLER_HABIT_PROBABILITY:
        selected = best_habit
        selected_controller = "habitual"
    else:
        selected = best_goal
        selected_controller = "goal_directed"

    goal_id = best_goal[1].action_id if best_goal is not None else ""
    habit_id = best_habit[1].action_id if best_habit is not None else ""
    probabilities = {entry.action_id: 0.0 for entry in entries}
    if goal_id and habit_id and goal_id != habit_id:
        probabilities[goal_id] = 1.0 - RANDOM_CONTROLLER_HABIT_PROBABILITY
        probabilities[habit_id] = RANDOM_CONTROLLER_HABIT_PROBABILITY
    else:
        probabilities[selected[1].action_id] = 1.0
    return (
        ActionSelectionResult(
            selected_action_id=selected[1].action_id,
            method="random_controller",
            probabilities=probabilities,
        ),
        selected_controller,
    )


def _build_entry(
    *,
    action_id: str,
    candidate: ActionCandidate,
    reward_evaluation: RewardEvaluationResult,
    degree_of_model_based_control: float,
    use_habitual_weight: bool,
) -> ArbitrationEntry:
    reward = reward_evaluation.evaluation_by_id(action_id)
    if reward is None:
        raise ValueError(f"Missing Reward Evaluation for action {action_id}.")
    normalized_reward = min(1.0, max(0.0, reward.reward / 10.0))
    habit_strength = min(1.0, max(0.0, candidate.maximum_habit_strength))
    if use_habitual_weight:
        model_based_contribution = degree_of_model_based_control * normalized_reward
        habitual_contribution = (1.0 - degree_of_model_based_control) * habit_strength
    else:
        model_based_contribution = normalized_reward
        habitual_contribution = 0.0
    decision_value = model_based_contribution + habitual_contribution
    return ArbitrationEntry(
        action_id=action_id,
        action_text=candidate.action_text,
        reward=reward.reward,
        normalized_reward=normalized_reward,
        habit_strength=habit_strength,
        model_based_contribution=model_based_contribution,
        habitual_contribution=habitual_contribution,
        decision_value=decision_value,
    )

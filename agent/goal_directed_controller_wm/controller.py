from __future__ import annotations

from dataclasses import dataclass

from agent.habitual_controller import PreparedHabitualResponse
from agent.perceive import PerceiveResult
from agent.working_memory import build_working_memory
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from contextual_world.world_old.action_execution import WorldActionExecutor

from .action_generation import ActionCandidate, ActionGenerationResult, generate_action_space
from .arbiter import ArbitrationResult, DEFAULT_ARBITER_MODE, arbitrate_actions
from .planning import (
    beam_search_state_transitions,
    greedy_search_state_transitions,
    mcts_state_transitions,
    normalize_planning_strategy,
)
from .reward_evaluation import RewardEvaluationResult, evaluate_action_rewards
from .state_transition import (
    DEFAULT_WORLD_MODEL_MODE,
    StateTransitionRolloutResult,
    normalize_world_model_mode,
    rollout_state_transitions,
)


@dataclass
class WorldModelControllerResult:
    selected_candidate: ActionCandidate | None = None
    generation: ActionGenerationResult | None = None
    rollout: StateTransitionRolloutResult | None = None
    reward_evaluation: RewardEvaluationResult | None = None
    arbitration: ArbitrationResult | None = None
    error: str = ""
    world_model_mode: str = DEFAULT_WORLD_MODEL_MODE
    planning_strategy: str = "greedy"

    def to_trace_dict(self) -> dict:
        candidates = self.generation.action_space if self.generation is not None else []
        return {
            "world_model_mode": self.world_model_mode,
            "planning_strategy": self.planning_strategy,
            "error": self.error,
            "selected_candidate_id": (
                self.selected_candidate.candidate_id if self.selected_candidate is not None else ""
            ),
            "selected_action": (
                self.selected_candidate.action_text if self.selected_candidate is not None else ""
            ),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "rollout": _rollout_trace(self.rollout),
            "reward_evaluation": (
                self.reward_evaluation.to_dict() if self.reward_evaluation is not None else {}
            ),
            "arbitration": self.arbitration.to_dict() if self.arbitration is not None else {},
        }


def run_world_model_controller(
    agent,
    engine: PhysicsEngine,
    perception: PerceiveResult,
    *,
    habitual_responses: list[PreparedHabitualResponse] | None = None,
    action_sample_window: int = 3,
    iteration_number: int = 3,
    recent_experience_count: int = 8,
    world_model_mode: str = DEFAULT_WORLD_MODEL_MODE,
    action_executor: WorldActionExecutor | None = None,
    previous_execution_result: str = "",
    selection_method: str = "greedy",
    selection_temperature: float = 0.2,
    selection_epsilon: float = 0.1,
    planning_strategy: str = "greedy",
    beam_width: int = 3,
    planning_branching_factor: int = 3,
    mcts_simulations: int = 12,
    mcts_exploration_constant: float = 2 ** 0.5,
    arbiter_mode: str = DEFAULT_ARBITER_MODE,
    arbiter_rng=None,
    provider_name: str = "ollama",
    model: str | None = None,
) -> WorldModelControllerResult:
    mode = normalize_world_model_mode(world_model_mode)
    strategy = normalize_planning_strategy(planning_strategy)
    intent = getattr(agent, "active_intent", None)
    if intent is None or intent.status != "active":
        return WorldModelControllerResult(
            error="World Model requires an active Intent.",
            world_model_mode=mode,
            planning_strategy=strategy,
        )

    working_memory = build_working_memory(agent, engine, perception, intent=intent)
    generation = generate_action_space(
        agent_name=agent.name,
        intent=intent,
        working_memory=working_memory,
        engine=engine,
        habitual_responses=habitual_responses or [],
        action_sample_window=action_sample_window,
        provider_name=provider_name,
        model=model,
    )
    result = WorldModelControllerResult(
        generation=generation,
        world_model_mode=mode,
        planning_strategy=strategy,
    )
    if generation.error:
        result.error = generation.error
        return result
    if not generation.action_space:
        result.error = "World Model action generation produced no candidates."
        return result

    rollout_function = {
        "beam": beam_search_state_transitions,
        "greedy": greedy_search_state_transitions,
        "mcts": mcts_state_transitions,
    }[strategy]
    rollout_kwargs = {
        "branching_factor": planning_branching_factor,
    }
    if strategy == "beam":
        rollout_kwargs = {
            "beam_width": beam_width,
            "branching_factor": planning_branching_factor,
        }
    elif strategy == "mcts":
        rollout_kwargs = {
            "branching_factor": planning_branching_factor,
            "simulation_budget": mcts_simulations,
            "exploration_constant": mcts_exploration_constant,
        }
    rollout = rollout_function(
        agent,
        intent=intent,
        engine=engine,
        initial_action_space=generation.action_space,
        iteration_number=iteration_number,
        recent_experience_count=recent_experience_count,
        world_model_mode=mode,
        action_executor=action_executor,
        previous_execution_result=previous_execution_result,
        provider_name=provider_name,
        model=model,
        **rollout_kwargs,
    )
    result.rollout = rollout
    reward = evaluate_action_rewards(
        agent,
        intent=intent,
        rollout=rollout,
        provider_name=provider_name,
        model=model,
    )
    result.reward_evaluation = reward
    if reward.error:
        result.error = reward.error
        return result

    try:
        arbitration = arbitrate_actions(
            generation.action_space,
            reward,
            degree_of_model_based_control=agent.degree_of_model_based_control,
            selection_method=selection_method,
            temperature=selection_temperature,
            epsilon=selection_epsilon,
            arbiter_mode=arbiter_mode,
            rng=arbiter_rng,
        )
    except Exception as error:
        result.error = f"World Model arbitration failed: {error}"
        return result

    result.arbitration = arbitration
    try:
        selected_index = int(arbitration.selected_action_id) - 1
        result.selected_candidate = generation.action_space[selected_index]
    except (TypeError, ValueError, IndexError):
        result.error = (
            "World Model selected an unknown action: "
            f"{arbitration.selected_action_id}"
        )
    return result


def _rollout_trace(rollout: StateTransitionRolloutResult | None) -> list[dict]:
    if rollout is None:
        return []
    entries = []
    for node in rollout.iter_nodes():
        transition = node.transition
        entries.append(
            {
                "path": node.path_text,
                "depth": node.depth,
                "action": node.candidate.action_text,
                "predicted_outcome": (
                    transition.outcome.description if transition is not None else ""
                ),
                "intent_satisfaction": (
                    transition.intent_satisfaction.status if transition is not None else ""
                ),
                "stop_reason": node.stop_reason,
                "error": node.transition_result.error or node.expansion_error,
                "world_model_mode": node.transition_result.world_model_mode,
                "failure_context": (
                    {
                        "action_text": node.transition_result.failure_context.action_text,
                        "failed_module": node.transition_result.failure_context.failed_module,
                        "error": node.transition_result.failure_context.error,
                        "world_feedback": node.transition_result.failure_context.world_feedback,
                    }
                    if node.transition_result.failure_context is not None
                    else {}
                ),
            }
        )
    return entries

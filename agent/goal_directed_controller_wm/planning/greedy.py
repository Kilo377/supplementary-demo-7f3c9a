from __future__ import annotations

from agent.goal_directed_controller_wm.action_generation import ActionCandidate
from agent.goal_directed_controller_wm.state_transition.backend import (
    normalize_world_model_mode,
)
from agent.goal_directed_controller_wm.state_transition.rollout import (
    StateTransitionRolloutResult,
    StateTransitionRolloutState,
    rollout_state_transitions,
)
from agent.intent.state import IntentState
from contextual_world.world_old.action_execution import WorldActionExecutor
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine

from .beam import _build_chain, _expand_path, _score_paths, _start_path


def greedy_search_state_transitions(
    agent,
    *,
    intent: IntentState,
    engine: PhysicsEngine,
    initial_action_space: list[ActionCandidate],
    iteration_number: int = 3,
    recent_experience_count: int = 8,
    branching_factor: int = 3,
    world_model_mode: str = "simple",
    action_executor: WorldActionExecutor | None = None,
    previous_execution_result: str = "",
    provider_name: str = "ollama",
    model: str | None = None,
) -> StateTransitionRolloutResult:
    """Roll out each root by retaining its highest-valued child at every depth."""
    mode = normalize_world_model_mode(world_model_mode)
    if mode != "simple":
        return rollout_state_transitions(
            agent,
            intent=intent,
            engine=engine,
            initial_action_space=initial_action_space,
            iteration_number=iteration_number,
            recent_experience_count=recent_experience_count,
            world_model_mode=mode,
            action_executor=action_executor,
            previous_execution_result=previous_execution_result,
            provider_name=provider_name,
            model=model,
        )

    depth_limit = max(1, int(iteration_number))
    branches = max(1, min(12, int(branching_factor)))
    initial_state = StateTransitionRolloutState.from_agent(
        agent,
        intent_text=intent.intent_text,
        recent_experience_count=recent_experience_count,
    )
    paths = [
        _start_path(
            root_index=index,
            candidate=candidate,
            state=initial_state,
            depth_limit=depth_limit,
            provider_name=provider_name,
            model=model,
        )
        for index, candidate in enumerate(initial_action_space, start=1)
    ]

    for depth in range(2, depth_limit + 1):
        expanded = []
        retained = []
        for path in paths:
            if path.stopped:
                retained.append(path)
                continue
            children = _expand_path(
                path,
                depth=depth,
                depth_limit=depth_limit,
                branching_factor=branches,
                intent=intent,
                engine=engine,
                provider_name=provider_name,
                model=model,
            )
            if children:
                expanded.extend(children)
            else:
                retained.append(path)

        _score_paths(
            agent,
            intent=intent,
            paths=expanded,
            provider_name=provider_name,
            model=model,
        )
        best_by_root = {path.root_index: path for path in retained}
        for path in expanded:
            previous = best_by_root.get(path.root_index)
            if previous is None or path.score > previous.score:
                best_by_root[path.root_index] = path
        paths = [best_by_root[index] for index in sorted(best_by_root)]

    roots = [
        _build_chain(path, action_id=index)
        for index, path in enumerate(paths, start=1)
    ]
    return StateTransitionRolloutResult(
        roots=roots,
        iteration_number=depth_limit,
        world_model_mode=mode,
    )

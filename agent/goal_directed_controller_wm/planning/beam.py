from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from agent.goal_directed_controller_wm.action_generation import ActionCandidate
from agent.goal_directed_controller_wm.reward_evaluation import evaluate_action_rewards
from agent.goal_directed_controller_wm.state_transition.backend import (
    normalize_world_model_mode,
)
from agent.goal_directed_controller_wm.state_transition.iteration_action import (
    generate_transition_iteration_action,
)
from agent.goal_directed_controller_wm.state_transition.rollout import (
    StateTransitionRolloutNode,
    StateTransitionRolloutResult,
    StateTransitionRolloutState,
    rollout_state_transitions,
)
from agent.goal_directed_controller_wm.state_transition.transition import (
    run_state_transition,
)
from agent.intent.state import IntentState
from contextual_world.world_old.action_execution import WorldActionExecutor
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine


@dataclass
class _BeamPath:
    root_index: int
    nodes: tuple[StateTransitionRolloutNode, ...]
    state: StateTransitionRolloutState
    score: int = 0
    stopped: bool = False


def beam_search_state_transitions(
    agent,
    *,
    intent: IntentState,
    engine: PhysicsEngine,
    initial_action_space: list[ActionCandidate],
    iteration_number: int = 3,
    recent_experience_count: int = 8,
    beam_width: int = 3,
    branching_factor: int = 3,
    world_model_mode: str = "simple",
    action_executor: WorldActionExecutor | None = None,
    previous_execution_result: str = "",
    provider_name: str = "ollama",
    model: str | None = None,
) -> StateTransitionRolloutResult:
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
    width = max(1, int(beam_width))
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
    _score_paths(agent, intent=intent, paths=paths, provider_name=provider_name, model=model)
    best_by_root = {path.root_index: path for path in paths}
    frontier = _best_active(paths, width=width)

    for depth in range(2, depth_limit + 1):
        expanded: list[_BeamPath] = []
        for path in frontier:
            expanded.extend(
                _expand_path(
                    path,
                    depth=depth,
                    depth_limit=depth_limit,
                    branching_factor=branches,
                    intent=intent,
                    engine=engine,
                    provider_name=provider_name,
                    model=model,
                )
            )
        if not expanded:
            break
        _score_paths(
            agent,
            intent=intent,
            paths=expanded,
            provider_name=provider_name,
            model=model,
        )
        for path in expanded:
            previous = best_by_root[path.root_index]
            if (path.score, len(path.nodes)) > (previous.score, len(previous.nodes)):
                best_by_root[path.root_index] = path
        frontier = _best_active(expanded, width=width)
        if not frontier:
            break

    roots = [
        _build_chain(best_by_root[index], action_id=index)
        for index in range(1, len(initial_action_space) + 1)
    ]
    return StateTransitionRolloutResult(
        roots=roots,
        iteration_number=depth_limit,
        world_model_mode=mode,
    )


def _start_path(
    *,
    root_index: int,
    candidate: ActionCandidate,
    state: StateTransitionRolloutState,
    depth_limit: int,
    provider_name: str,
    model: str | None,
) -> _BeamPath:
    result = run_state_transition(
        state.to_prompt_context(action_text=candidate.action_text),
        provider_name=provider_name,
        model=model,
    )
    node = StateTransitionRolloutNode(
        path=(root_index,),
        depth=1,
        candidate=candidate,
        transition_result=result,
    )
    transition = result.transition
    stopped = transition is None or transition.intent_satisfaction.is_satisfied or depth_limit <= 1
    node.stop_reason = _stop_reason(transition, at_limit=depth_limit <= 1)
    next_state = (
        state.after_transition(action_text=candidate.action_text, transition=transition)
        if transition is not None
        else state
    )
    return _BeamPath(root_index, (node,), next_state, stopped=stopped)


def _expand_path(
    path: _BeamPath,
    *,
    depth: int,
    depth_limit: int,
    branching_factor: int,
    intent: IntentState,
    engine: PhysicsEngine,
    provider_name: str,
    model: str | None,
) -> list[_BeamPath]:
    generation = generate_transition_iteration_action(
        agent_name=path.state.agent_name,
        intent_text=intent.intent_text,
        transition_state_text=path.state.format_for_action_prompt(),
        action_sample_window=branching_factor,
        engine=engine,
        provider_name=provider_name,
        model=model,
    )
    if generation.error or not generation.action_space:
        path.nodes[-1].expansion_error = generation.error
        path.nodes[-1].stop_reason = "action_generation_error" if generation.error else "no_next_action"
        path.stopped = True
        return []

    children = []
    for child_index, candidate in enumerate(generation.action_space, start=1):
        result = run_state_transition(
            path.state.to_prompt_context(action_text=candidate.action_text),
            provider_name=provider_name,
            model=model,
        )
        node = StateTransitionRolloutNode(
            path=(*path.nodes[-1].path, child_index),
            depth=depth,
            candidate=candidate,
            transition_result=result,
            iteration_intuition_prompt=generation.prompt,
        )
        transition = result.transition
        stopped = transition is None or transition.intent_satisfaction.is_satisfied or depth >= depth_limit
        node.stop_reason = _stop_reason(transition, at_limit=depth >= depth_limit)
        next_state = (
            path.state.after_transition(action_text=candidate.action_text, transition=transition)
            if transition is not None
            else path.state
        )
        children.append(
            _BeamPath(
                root_index=path.root_index,
                nodes=(*path.nodes, node),
                state=next_state,
                stopped=stopped,
            )
        )
    return children


def _score_paths(agent, *, intent, paths, provider_name, model) -> None:
    if not paths:
        return
    rollout = StateTransitionRolloutResult(
        roots=[_build_chain(path, action_id=index) for index, path in enumerate(paths, 1)],
        iteration_number=max(len(path.nodes) for path in paths),
    )
    result = evaluate_action_rewards(
        agent,
        intent=intent,
        rollout=rollout,
        provider_name=provider_name,
        model=model,
    )
    if result.error:
        return
    for index, path in enumerate(paths, 1):
        evaluation = result.evaluation_by_id(str(index))
        if evaluation is not None:
            path.score = evaluation.reward


def _best_active(paths: list[_BeamPath], *, width: int) -> list[_BeamPath]:
    active = [path for path in paths if not path.stopped]
    return sorted(active, key=lambda path: (path.score, -path.root_index), reverse=True)[:width]


def _build_chain(path: _BeamPath, *, action_id: int) -> StateTransitionRolloutNode:
    nodes = [deepcopy(node) for node in path.nodes]
    for depth, node in enumerate(nodes, start=1):
        node.path = (action_id, *([1] * (depth - 1)))
        node.children = [nodes[depth]] if depth < len(nodes) else []
    return nodes[0]


def _stop_reason(transition, *, at_limit: bool) -> str:
    if transition is None:
        return "transition_error"
    if transition.intent_satisfaction.is_satisfied:
        return "intent_satisfied"
    if at_limit:
        return "iteration_limit"
    return ""

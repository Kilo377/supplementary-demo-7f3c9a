from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

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

from .beam import _BeamPath, _build_chain, _expand_path, _score_paths, _start_path

REWARD_SCALE = 10.0


@dataclass
class _MCTSNode:
    path: _BeamPath
    parent: "_MCTSNode | None" = None
    children: list["_MCTSNode"] = field(default_factory=list)
    expanded: bool = False
    visits: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0


def mcts_state_transitions(
    agent,
    *,
    intent: IntentState,
    engine: PhysicsEngine,
    initial_action_space: list[ActionCandidate],
    iteration_number: int = 3,
    recent_experience_count: int = 8,
    branching_factor: int = 3,
    simulation_budget: int = 12,
    exploration_constant: float = math.sqrt(2.0),
    world_model_mode: str = "simple",
    action_executor: WorldActionExecutor | None = None,
    previous_execution_result: str = "",
    provider_name: str = "ollama",
    model: str | None = None,
    rng: random.Random | None = None,
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
    branches = max(1, min(12, int(branching_factor)))
    budget = max(1, int(simulation_budget))
    exploration = max(0.0, float(exploration_constant))
    random_source = rng or random.Random()
    initial_state = StateTransitionRolloutState.from_agent(
        agent,
        intent_text=intent.intent_text,
        recent_experience_count=recent_experience_count,
    )
    root_paths = [
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
    _score_paths(
        agent,
        intent=intent,
        paths=root_paths,
        provider_name=provider_name,
        model=model,
    )
    roots = [
        _MCTSNode(path=path, visits=1, total_reward=float(path.score))
        for path in root_paths
    ]

    for _ in range(budget):
        node = _select_root(roots, exploration=exploration)
        node = _select_and_expand(
            node,
            depth_limit=depth_limit,
            branching_factor=branches,
            exploration=exploration,
            intent=intent,
            engine=engine,
            provider_name=provider_name,
            model=model,
            rng=random_source,
        )
        _score_paths(
            agent,
            intent=intent,
            paths=[node.path],
            provider_name=provider_name,
            model=model,
        )
        _backpropagate(node, float(node.path.score))

    selected_paths = [_most_visited_path(root) for root in roots]
    return StateTransitionRolloutResult(
        roots=[
            _build_chain(path, action_id=index)
            for index, path in enumerate(selected_paths, start=1)
        ],
        iteration_number=depth_limit,
        world_model_mode=mode,
    )


def _select_root(roots: list[_MCTSNode], *, exploration: float) -> _MCTSNode:
    total_visits = max(1, sum(root.visits for root in roots))
    return max(
        roots,
        key=lambda node: _ucb_score(
            node,
            parent_visits=total_visits,
            exploration=exploration,
        ),
    )


def _select_and_expand(
    node: _MCTSNode,
    *,
    depth_limit: int,
    branching_factor: int,
    exploration: float,
    intent: IntentState,
    engine: PhysicsEngine,
    provider_name: str,
    model: str | None,
    rng: random.Random,
) -> _MCTSNode:
    while True:
        if node.path.stopped or len(node.path.nodes) >= depth_limit:
            return node
        if not node.expanded:
            paths = _expand_path(
                node.path,
                depth=len(node.path.nodes) + 1,
                depth_limit=depth_limit,
                branching_factor=branching_factor,
                intent=intent,
                engine=engine,
                provider_name=provider_name,
                model=model,
            )
            node.children = [_MCTSNode(path=path, parent=node) for path in paths]
            node.expanded = True
            if not node.children:
                return node
        unvisited = [child for child in node.children if child.visits == 0]
        if unvisited:
            return rng.choice(unvisited)
        node = max(
            node.children,
            key=lambda child: _ucb_score(
                child,
                parent_visits=node.visits,
                exploration=exploration,
            ),
        )


def _ucb_score(
    node: _MCTSNode,
    *,
    parent_visits: int,
    exploration: float,
) -> float:
    if node.visits == 0:
        return math.inf
    normalized_mean_reward = node.mean_reward / REWARD_SCALE
    return normalized_mean_reward + exploration * math.sqrt(
        math.log(max(1, parent_visits)) / node.visits
    )


def _backpropagate(node: _MCTSNode, reward: float) -> None:
    current = node
    while current is not None:
        current.visits += 1
        current.total_reward += reward
        current = current.parent


def _most_visited_path(root: _MCTSNode) -> _BeamPath:
    node = root
    while True:
        visited = [child for child in node.children if child.visits > 0]
        if not visited:
            return node.path
        node = max(
            visited,
            key=lambda child: (
                child.visits,
                child.mean_reward,
                len(child.path.nodes),
            ),
        )

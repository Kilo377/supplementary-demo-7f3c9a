from __future__ import annotations

import atexit
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
import json
from typing import Any

from agent.action import ActionProposalResult
from agent.belief.time_belief import parse_duration_seconds
from agent.goal_directed_controller_wm.action_generation import ActionCandidate
from agent.target_resolver import (
    decision_from_target_resolution,
    resolve_target_fallback,
    resolve_target_with_llm,
)
from contextual_world.runtime_adapter import ContextualWorldActionPipe
from contextual_world.world_old.action_execution import WorldActionExecutor
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from core.action_types import ActionResult, AgentDecision

from .intent_satisfaction import evaluate_transition_intent_satisfaction
from .types import (
    StateTransition,
    StateTransitionResult,
    TransitionFailureContext,
    TransitionInternalStateChanges,
    TransitionOutcome,
    TransitionSelfState,
    TransitionSpatialBeliefUpdate,
)


@dataclass
class ComplexWorldSnapshot:
    agent: Any
    engine: PhysicsEngine
    executor: WorldActionExecutor
    previous_execution_result: str = ""


class _SimulationFailureSession:
    """Discard failures produced by counterfactual World execution."""

    def observe_action_attempt(self) -> None:
        return None

    def record_failure(self, record: dict) -> None:
        return None

    def record_result(self, result: object) -> None:
        return None

    def close(self) -> None:
        return None


def fork_complex_world(
    agent,
    engine: PhysicsEngine,
    source_executor: WorldActionExecutor | None,
    *,
    provider_name: str,
    model: str | None,
    previous_execution_result: str = "",
) -> ComplexWorldSnapshot:
    cloned = deepcopy(
        {
            "agent": agent,
            "home": engine.home,
            "graph": (
                source_executor.graph_pipe.graph
                if source_executor is not None
                and source_executor.graph_pipe is not None
                else None
            ),
        }
    )
    cloned_agent = cloned["agent"]
    cloned_home = cloned["home"]
    cloned_engine = PhysicsEngine(cloned_home)
    cloned_executor = WorldActionExecutor(
        provider_name=provider_name,
        model=model,
    )
    source_pipe = (
        source_executor.graph_pipe
        if source_executor is not None and source_executor.graph_pipe is not None
        else None
    )
    cloned_pipe = ContextualWorldActionPipe(
        home=cloned_home,
        provider_name=provider_name,
        model=model,
        agent_node_id=cloned_agent.node_id,
        use_llm_summary=(source_pipe.use_llm_summary if source_pipe is not None else True),
        graph=cloned["graph"],
    )
    atexit.unregister(cloned_pipe.failure_session.close)
    cloned_pipe.failure_session = _SimulationFailureSession()
    cloned_executor.graph_pipe = cloned_pipe
    return ComplexWorldSnapshot(
        agent=cloned_agent,
        engine=cloned_engine,
        executor=cloned_executor,
        previous_execution_result=str(
            previous_execution_result
            or getattr(agent, "last_world_feedback_summary", "")
            or ""
        ).strip(),
    )


def run_complex_state_transition(
    snapshot: ComplexWorldSnapshot,
    candidate: ActionCandidate,
    *,
    intent_text: str,
    provider_name: str = "ollama",
    model: str | None = None,
) -> StateTransitionResult:
    before_world = _world_snapshot(snapshot.engine)
    before_worn = tuple(snapshot.agent.worn_items or [])
    before_surface = str(snapshot.agent.body_surface or "")
    try:
        decision = _decision_for_candidate(
            snapshot,
            candidate,
            provider_name=provider_name,
            model=model,
        )
        results = _execute_complete_semantic_action(snapshot, decision)
    except Exception as error:
        return StateTransitionResult(
            transition=None,
            prompt="",
            provider_name=provider_name,
            model=model,
            world_model_mode="complex",
            error=f"Complex state transition failed: {error}",
        )

    feedback = " ".join(
        result.feedback_text().strip()
        for result in results
        if result.feedback_text().strip()
    )
    snapshot.previous_execution_result = feedback
    snapshot.agent.last_world_feedback_summary = feedback
    elapsed_seconds = sum(
        max(0, parse_duration_seconds(result.estimated_duration))
        for result in results
    )
    if elapsed_seconds:
        snapshot.agent.time_belief.current_datetime = (
            snapshot.agent.time_belief.current_datetime
            + timedelta(seconds=elapsed_seconds)
        )
    snapshot.agent.perceive(snapshot.engine)

    after_world = _world_snapshot(snapshot.engine)
    updates = _merge_updates(
        _world_updates(before_world, after_world),
        _action_result_updates(results),
    )
    final_result = results[-1]
    failed = (
        bool(final_result.error)
        or final_result.execution_kind == "action_error"
        or bool(final_result.execution_debug.get("failed_module"))
        or bool(final_result.execution_debug.get("error"))
    )
    outcome_status = (
        "failed"
        if failed
        else ("partial" if final_result.execution_kind == "chat_todo" else "success")
    )
    failure_context = (
        TransitionFailureContext(
            action_text=candidate.action_text,
            failed_module=str(
                final_result.execution_debug.get("failed_module", "")
                or final_result.execution_kind
            ).strip(),
            error=str(
                final_result.error
                or final_result.execution_debug.get("error", "")
                or ""
            ).strip(),
            world_feedback=feedback,
        )
        if failed
        else None
    )
    next_state = _next_self_state(
        snapshot.agent,
        snapshot.engine,
        before_worn=before_worn,
        before_surface=before_surface,
        decision=decision,
    )
    next_state_text = _next_state_text(next_state, updates)
    satisfaction, prompt, raw_response, satisfaction_error = (
        evaluate_transition_intent_satisfaction(
            agent_name=snapshot.agent.name,
            intent_text=intent_text,
            action_text=candidate.action_text,
            world_feedback=feedback,
            next_state_text=next_state_text,
            action_failed=failed,
            provider_name=provider_name,
            model=model,
        )
    )
    transition = StateTransition(
        outcome=TransitionOutcome(
            status=outcome_status,
            description=feedback,
            reason=(
                final_result.error
                or "This result comes from actual action execution in the Contextual World replica."
            ),
        ),
        intent_satisfaction=satisfaction,
        elapsed_seconds=elapsed_seconds,
        next_self_state=next_state,
        internal_state_changes=TransitionInternalStateChanges(),
        spatial_belief_updates=updates,
        expected_feedback=feedback,
        uncertainty=satisfaction_error,
    )
    return StateTransitionResult(
        transition=transition,
        prompt=prompt,
        raw_response=raw_response,
        provider_name=provider_name,
        model=model,
        world_model_mode="complex",
        failure_context=failure_context,
    )


def _decision_for_candidate(
    snapshot: ComplexWorldSnapshot,
    candidate: ActionCandidate,
    *,
    provider_name: str,
    model: str | None,
) -> AgentDecision:
    intuition = candidate.intuition_reply
    if intuition.route == "wait":
        return AgentDecision(
            action_type="wait",
            reason=intuition.thought,
            action_proposal_text=intuition.thought,
            intuition_route="wait",
            intuition_thought=intuition.thought,
            wait_duration=intuition.wait_duration,
        )
    if intuition.route == "chat":
        return AgentDecision(
            action_type="chat",
            reason=intuition.thought,
            action_proposal_text=intuition.thought,
            intuition_route="chat",
            intuition_thought=intuition.thought,
        )
    if intuition.route == "walk":
        area_id = _resolve_area_id(snapshot.engine, intuition)
        if area_id:
            return AgentDecision(
                action_type="move_to_area",
                target_area_id=area_id,
                reason=intuition.thought,
                action_proposal_text=intuition.thought,
                intuition_route="walk",
                intuition_thought=intuition.thought,
            )
        return AgentDecision(
            action_type="wait",
            reason=intuition.thought,
            action_proposal_text=intuition.thought,
            intuition_route="walk",
            intuition_thought=intuition.thought,
        )

    perception = snapshot.agent.perceive(snapshot.engine)
    proposal = ActionProposalResult(
        action_text=candidate.action_text,
        provider_name=provider_name,
        model=model,
    )
    try:
        target = resolve_target_with_llm(
            snapshot.engine,
            agent_name=snapshot.agent.name,
            action_text=candidate.action_text,
            current_area_id=perception.area_id,
            current_area_name=perception.area_name,
            provider_name=provider_name,
            model=model,
            spatial_belief=snapshot.agent.belief,
            world_graph_context=snapshot.executor.target_resolver_context(
                snapshot.agent,
                snapshot.engine,
            ),
        )
    except Exception:
        target = resolve_target_fallback(
            snapshot.engine,
            action_text=candidate.action_text,
            current_area_id=perception.area_id,
            spatial_belief=snapshot.agent.belief,
        )
    return decision_from_target_resolution(proposal, target)


def _resolve_area_id(engine: PhysicsEngine, intuition) -> str:
    area_id = str(getattr(intuition, "target_area_id", "") or "").strip()
    if area_id and engine.get_area(area_id) is not None:
        return area_id
    area_name = str(getattr(intuition, "target_area_name", "") or "").strip()
    thought = str(getattr(intuition, "thought", "") or "")
    for area in engine.home.areas:
        if area_name and (
            area_name == area.name
            or area_name in area.name
            or area.name in area_name
        ):
            return area.node_id
        if area.name in thought or area.node_id in thought:
            return area.node_id
    return ""


def _execute_complete_semantic_action(
    snapshot: ComplexWorldSnapshot,
    decision: AgentDecision,
) -> list[ActionResult]:
    results = [
        snapshot.executor.execute(
            snapshot.agent,
            snapshot.engine,
            decision,
            previous_execution_result=snapshot.previous_execution_result,
        )
    ]
    for _ in range(2):
        if not snapshot.executor.has_pending_graph_action():
            break
        pending_text = snapshot.executor.pending_graph_action_text()
        pending_decision = AgentDecision(
            action_type="action",
            reason=pending_text,
            action_proposal_text=pending_text,
        )
        results.append(
            snapshot.executor.execute(
                snapshot.agent,
                snapshot.engine,
                pending_decision,
                previous_execution_result=results[-1].feedback_text(),
            )
        )
    return results


def _world_snapshot(engine: PhysicsEngine) -> dict[str, tuple]:
    snapshot = {}
    for area in engine.home.areas:
        for element in area.elements:
            snapshot[element.node_id] = (
                element.name,
                element.physical_status,
                element.evolution_status,
                element.interaction_status,
                deepcopy(element.state_details),
            )
    return snapshot


def _world_updates(before: dict, after: dict) -> tuple[TransitionSpatialBeliefUpdate, ...]:
    updates = []
    for element_id, current in after.items():
        previous = before.get(element_id)
        if previous is None or previous == current:
            continue
        changes = []
        labels = (
            "name",
            "physical_status",
            "evolution_status",
            "interaction_status",
            "state_details",
        )
        for index in range(1, len(current)):
            if previous[index] != current[index]:
                old = (
                    json.dumps(previous[index], ensure_ascii=False, sort_keys=True)
                    if isinstance(previous[index], dict)
                    else previous[index]
                )
                new = (
                    json.dumps(current[index], ensure_ascii=False, sort_keys=True)
                    if isinstance(current[index], dict)
                    else current[index]
                )
                changes.append(f"{labels[index]} changed from {old} to {new}")
        updates.append(
            TransitionSpatialBeliefUpdate(
                element=current[0],
                state_change="，".join(changes),
            )
        )
    return tuple(updates)


def _action_result_updates(
    results: list[ActionResult],
) -> tuple[TransitionSpatialBeliefUpdate, ...]:
    updates = []
    for result in results:
        for change in list(result.temporary_element_changes or []):
            if not isinstance(change, dict):
                continue
            element = str(
                change.get("name")
                or change.get("temporary_element_name")
                or change.get("temporary_element_id")
                or "Temporary object"
            ).strip()
            updates.append(
                TransitionSpatialBeliefUpdate(
                    element=element,
                    state_change=json.dumps(change, ensure_ascii=False, sort_keys=True),
                )
            )
        feedback = result.environment_feedback
        for change in list(getattr(feedback, "environment_changes", []) or []):
            if not isinstance(change, dict):
                continue
            element = str(
                change.get("element_name")
                or change.get("name")
                or change.get("element_id")
                or "Environment"
            ).strip()
            updates.append(
                TransitionSpatialBeliefUpdate(
                    element=element,
                    state_change=json.dumps(change, ensure_ascii=False, sort_keys=True),
                )
            )
    return tuple(updates)


def _merge_updates(
    *groups: tuple[TransitionSpatialBeliefUpdate, ...],
) -> tuple[TransitionSpatialBeliefUpdate, ...]:
    merged = []
    seen = set()
    for group in groups:
        for update in group:
            key = (update.element, update.state_change)
            if key in seen:
                continue
            seen.add(key)
            merged.append(update)
    return tuple(merged)


def _next_self_state(
    agent,
    engine: PhysicsEngine,
    *,
    before_worn: tuple[str, ...],
    before_surface: str,
    decision: AgentDecision,
) -> TransitionSelfState:
    area = engine.get_area_for_point(*agent.center)
    gaze_id = str(getattr(agent, "gaze_target", "") or "")
    gaze_element = engine.get_element(gaze_id)
    near_id = gaze_id or str(decision.navigation_target_element_id or decision.target_element_id or "")
    near_element = engine.get_element(near_id)
    interaction_names = tuple(
        name
        for item in list(getattr(agent, "interaction_elements", []) or [])
        if (name := _element_reference_name(item, engine))
    )
    worn = tuple(str(item) for item in list(getattr(agent, "worn_items", []) or []))
    surface = str(getattr(agent, "body_surface", "") or "")
    return TransitionSelfState(
        area=area.name if area is not None else "",
        near_element=near_element.name if near_element is not None else "",
        posture=str(getattr(agent, "posture", "") or ""),
        facing_or_gaze=(
            gaze_element.name
            if gaze_element is not None
            else (gaze_id or f"Facing {getattr(agent, 'facing', 0):.1f} degrees")
        ),
        holding=interaction_names,
        interacting_with=interaction_names,
        worn_items_change=(
            f"{list(before_worn)} -> {list(worn)}" if before_worn != worn else ""
        ),
        body_surface_change=(
            f"{before_surface} -> {surface}" if before_surface != surface else ""
        ),
    )


def _element_reference_name(item: object, engine: PhysicsEngine) -> str:
    if isinstance(item, dict):
        value = item.get("name") or item.get("element_id") or item.get("node_id") or ""
    else:
        value = str(item or "")
    element = engine.get_element(str(value))
    return element.name if element is not None else str(value).strip()


def _next_state_text(
    state: TransitionSelfState,
    updates: tuple[TransitionSpatialBeliefUpdate, ...],
) -> str:
    parts = [
        f"Current room: {state.area or 'Unknown'}",
        f"Nearby: {state.near_element or 'No clear object'}",
        f"Posture: {state.posture or 'Unknown'}",
        f"Gazing or facing: {state.facing_or_gaze or 'Unknown'}",
    ]
    if state.holding:
        parts.append(f"Holding or touching: {'、'.join(state.holding)}")
    if updates:
        parts.append(
            "Environmental changes:"
            + "；".join(f"{item.element}{item.state_change}" for item in updates)
        )
    return "\n".join(parts)

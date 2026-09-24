from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, hypot, radians, sin
from typing import Any

from contextual_world.runtime_adapter import ContextualWorldActionPipe
from core.action_types import ActionResult, AgentDecision
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine


@dataclass
class PendingContextualWorldAction:
    action_proposal_text: str
    support_result: dict
    anchor_element_id: str = ""
    execution_context: dict = field(default_factory=dict)


@dataclass
class WorldActionExecutor:
    provider_name: str = "ollama"
    model: str | None = None
    failure_output_path: str | None = None
    interaction_reach: float = 0.0
    disable_spatial_gating: bool = True
    graph_pipe: ContextualWorldActionPipe | None = None
    pending_graph_action: PendingContextualWorldAction | None = None

    def has_pending_graph_action(self) -> bool:
        return self.pending_graph_action is not None

    def pending_graph_action_text(self) -> str:
        if self.pending_graph_action is None:
            return ""
        return self.pending_graph_action.action_proposal_text

    def clear_pending_graph_action(self) -> None:
        self.pending_graph_action = None

    def target_resolver_context(
        self,
        actor: Any,
        engine: PhysicsEngine,
    ) -> dict:
        if self.graph_pipe is None or self.graph_pipe.graph is None:
            return {}
        self.graph_pipe.sync_actor_to_graph(actor, engine)
        graph = self.graph_pipe.graph
        agent_id = self._actor_node_id(actor)
        agent_node = graph.nodes.get(agent_id)
        fact_edges = []
        for edge in graph.edges_for_node(agent_id, edge_kind="fact"):
            other_id = edge.to_node_id if edge.from_node_id == agent_id else edge.from_node_id
            other = graph.nodes.get(other_id)
            fact_edges.append({
                "relation": edge.relation,
                "direction": "out" if edge.from_node_id == agent_id else "in",
                "other_node_id": other_id,
                "other_name": other.name if other is not None else other_id,
                "other_type": other.node_type if other is not None else "",
                "other_state": dict(other.state) if other is not None else {},
            })
        temporary_elements = []
        for node in graph.nodes.values():
            if node.node_type != "temporary_element":
                continue
            if node.state.get("visible") is False:
                continue
            related_facts = [
                {
                    "from": edge.from_node_id,
                    "relation": edge.relation,
                    "to": edge.to_node_id,
                }
                for edge in graph.edges_for_node(node.node_id, edge_kind="fact")
            ]
            temporary_elements.append({
                "temporary_element_id": node.node_id,
                "name": node.name,
                "state": dict(node.state),
                "related_facts": related_facts,
            })
        return {
            "agent_state": dict(agent_node.state) if agent_node is not None else {},
            "current_agent_facts": fact_edges,
            "visible_temporary_elements": temporary_elements,
        }

    def execute(
        self,
        actor: Any,
        engine: PhysicsEngine,
        decision: AgentDecision,
        *,
        previous_execution_result: str = "",
    ) -> ActionResult:
        # The agent-side cognition loop only decides whether this is movement,
        # waiting, or a world action. All self/element/temporary routing belongs
        # to contextual_world.
        if decision.action_type == "action":
            return self._contextual_world_action_interaction(
                actor,
                engine,
                decision,
            )

        if decision.action_type == "move":
            return self._move_to_element(
                actor,
                engine,
                decision,
            )

        if decision.action_type == "move_to_area" and decision.target_area_id is not None:
            return self._move_to_area(actor, engine, decision.target_area_id)

        if decision.action_type == "wait":
            return self._wait_through_contextual_world(
                actor,
                engine,
                decision,
                previous_execution_result=previous_execution_result,
            )

        # TODO: route chat to a real multi-agent dialogue pipeline.
        if decision.action_type == "chat":
            return ActionResult(
                summary=f"chat_route_todo: {decision.reason or f'{actor.name} wants to chat with someone.'}",
                execution_kind="chat_todo",
                resolved_action="",
            )

        raise RuntimeError(f"Unknown action_type: {decision.action_type}")

    def is_actor_near_element(
        self,
        actor: Any,
        engine: PhysicsEngine,
        element_id: str,
    ) -> bool:
        return self._is_near_element(actor, engine, element_id)

    def _wait_through_contextual_world(
        self,
        actor: Any,
        engine: PhysicsEngine,
        decision: AgentDecision,
        *,
        previous_execution_result: str = "",
    ) -> ActionResult:
        wait_text = self._wait_action_text(
            actor_name=actor.name,
            previous_execution_result=previous_execution_result,
            wait_duration=decision.wait_duration,
        )
        wait_decision = AgentDecision(
            action_type="action",
            reason=wait_text,
            action_proposal_text=wait_text,
            intuition_route=decision.intuition_route,
            intuition_thought=decision.intuition_thought,
            wait_duration=decision.wait_duration,
            shadow_intuition_route=decision.shadow_intuition_route,
            shadow_intuition_thought=decision.shadow_intuition_thought,
        )
        result = self._contextual_world_action_interaction(actor, engine, wait_decision)
        result.execution_kind = "wait"
        if decision.wait_duration:
            result.estimated_duration = decision.wait_duration
        if not result.summary:
            result.summary = f"{actor.name} waited for a moment."
        return result

    def _wait_action_text(
        self,
        *,
        actor_name: str,
        previous_execution_result: str = "",
        wait_duration: str = "",
    ) -> str:
        previous = str(previous_execution_result or "").strip()
        if not previous:
            previous = "No clear world feedback in the previous round"
        previous = previous.rstrip("。.!！?？；;，, ")
        duration_text = self._duration_text_for_world(wait_duration)
        return (
            f"The world feedback from the previous round was: {previous}, and {actor_name} intends to wait for {duration_text}.\n            Please end this waiting period and apply any state changes that would naturally occur during this time."
        )

    def _duration_text_for_world(self, duration: str) -> str:
        value = str(duration or "").strip()
        if value.endswith("min"):
            return f"{value[:-3]} minutes"
        if value.endswith("s"):
            return f"{value[:-1]} seconds"
        return value or "a while"

    def _move_to_element(
        self,
        actor: Any,
        engine: PhysicsEngine,
        decision: AgentDecision,
    ) -> ActionResult:
        element_id = decision.target_element_id or decision.navigation_target_element_id or ""
        if not element_id:
            return ActionResult(
                summary=f"{actor.name} wants to move, but no clear target element was provided.",
                execution_kind="action_error",
                error="move_target_missing",
                execution_debug={
                    "failed_module": "target_resolution",
                    "error": "move_target_missing",
                    "action_proposal": decision.action_proposal_text or decision.reason,
                },
            )

        element = engine.get_element(element_id)
        if element is None:
            message = f"{actor.name} wants to approach the target, but this target element could not be found in the current environment."
            self._record_world_executor_failure(
                actor,
                engine,
                failed_module="movement_target_resolution",
                error=f"unknown_target_element: {element_id}",
                action_proposal=decision.action_proposal_text or decision.reason,
                world_feedback=message,
                execution_context=self._world_execution_context(
                    actor,
                    decision,
                    pending_action=False,
                ),
                count_action_attempt=True,
            )
            return ActionResult(
                summary=message,
                execution_kind="move",
                resolved_action="",
            )

        movement_start = actor.center
        movement_summary = ""
        if not self._is_near_element(actor, engine, element_id):
            movement_summary = self._position_actor_for_element(actor, engine, element_id)
            if not movement_summary and not self._is_near_element(actor, engine, element_id):
                message = f"{actor.name} wants to approach {element.name}, but no suitable landing spot has been found yet."
                self._record_world_executor_failure(
                    actor,
                    engine,
                    failed_module="interaction_positioning",
                    error="interaction_position_unreachable",
                    action_proposal=decision.action_proposal_text or decision.reason,
                    world_feedback=message,
                    execution_context=self._world_execution_context(
                        actor,
                        decision,
                        pending_action=False,
                    ),
                    count_action_attempt=True,
                )
                return ActionResult(
                    summary=message,
                    execution_kind="move",
                    resolved_action="",
                )
        else:
            self._face_element(actor, element)
            if self.graph_pipe is not None and self.graph_pipe.graph is not None:
                self.graph_pipe.sync_actor_to_graph(actor, engine)

        if not movement_summary:
            area = engine.get_area_for_point(*actor.center)
            area_name = area.name if area is not None else "current area"
            movement_summary = f"{actor.name} has arrived at {area_name}, next to {element.name}."
        return ActionResult(
            summary=movement_summary,
            execution_kind="move",
            resolved_action=movement_summary,
            estimated_duration=self._walking_duration(movement_start, actor.center),
            execution_debug={
                "target_resolution": self._decision_context(decision),
                "world_pipeline_invoked": False,
            },
        )

    def _contextual_world_action_interaction(
        self,
        actor: Any,
        engine: PhysicsEngine,
        decision: AgentDecision,
    ) -> ActionResult:
        action_text = decision.action_proposal_text or decision.reason
        pending = self.pending_graph_action
        navigation_summary = ""
        if pending is not None:
            action_text = pending.action_proposal_text
            support_result = pending.support_result
            pending_execution_context = dict(pending.execution_context or {})
            pending_execution_context["pending_action"] = {
                "was_pending": True,
                "pending_action_text": action_text,
                "anchor_element_id": pending.anchor_element_id,
            }
            if self._pending_still_needs_positioning(actor, engine, pending):
                self.pending_graph_action = None
                message = f"{actor.name} is no longer in a position to interact with the target; the original action was not executed."
                self._record_world_executor_failure(
                    actor,
                    engine,
                    failed_module="interaction_positioning",
                    error="pending_interaction_position_lost",
                    action_proposal=action_text,
                    world_feedback=message,
                    execution_context=pending_execution_context,
                    support_result=support_result,
                )
                return ActionResult(
                    summary=message,
                    execution_kind="action_error",
                    error="pending_interaction_position_lost",
                    execution_debug={
                        "failed_module": "interaction_positioning",
                        "error": "pending_interaction_position_lost",
                        "action_proposal": action_text,
                    },
                )
            self.pending_graph_action = None
            pipe_result = self._execute_contextual_world_action_with_support(
                actor,
                engine,
                action_text,
                support_result,
                execution_context=pending_execution_context,
            )
        else:
            execution_context = self._world_execution_context(
                actor,
                decision,
                pending_action=False,
            )
            try:
                support_result = self._plan_contextual_world_support(
                    actor,
                    engine,
                    action_text,
                    execution_context=execution_context,
                )
            except Exception as error:
                return ActionResult(
                    summary=f"{actor.name} did not receive stable environmental support for this step, so the action is temporarily pending.",
                    execution_kind="action_error",
                    execution_narration=f"element_support_error: {error}",
                    resolved_action="",
                    error=f"element_support_error: {error}",
                    environment_feedback=None,
                    execution_debug={
                        "failed_module": "element_support",
                        "error": str(error),
                        "action_proposal": action_text,
                    },
                )
            movement_start = actor.center
            navigation_anchor_id = (
                decision.navigation_target_element_id
                or self._primary_spatial_anchor_element_id(support_result)
            )
            if navigation_anchor_id:
                navigation_summary = self._position_actor_for_element(
                    actor,
                    engine,
                    navigation_anchor_id,
                )
            pending = PendingContextualWorldAction(
                action_proposal_text=action_text,
                support_result=support_result,
                anchor_element_id=navigation_anchor_id,
                execution_context=execution_context,
            )
            if self._pending_still_needs_positioning(actor, engine, pending):
                message = f"{actor.name} failed to reach a position where they can interact with the target, so the original action is temporarily pending."
                self._record_world_executor_failure(
                    actor,
                    engine,
                    failed_module="interaction_positioning",
                    error="interaction_position_unreachable",
                    action_proposal=action_text,
                    world_feedback=message,
                    execution_context=execution_context,
                    support_result=support_result,
                )
                return ActionResult(
                    summary=message,
                    execution_kind="action_error",
                    execution_narration="interaction_position_unreachable",
                    resolved_action="",
                    error="interaction_position_unreachable",
                    execution_debug={
                        "failed_module": "interaction_positioning",
                        "error": "interaction_position_unreachable",
                        "action_proposal": action_text,
                    },
                )
            if navigation_summary:
                if actor.center == movement_start:
                    message = f"{actor.name} did not actually get close to the target, so the original action is temporarily pending."
                    self._record_world_executor_failure(
                        actor,
                        engine,
                        failed_module="interaction_positioning",
                        error="interaction_position_no_progress",
                        action_proposal=action_text,
                        world_feedback=message,
                        execution_context=execution_context,
                        support_result=support_result,
                    )
                    return ActionResult(
                        summary=message,
                        execution_kind="action_error",
                        error="interaction_position_no_progress",
                        execution_debug={
                            "failed_module": "interaction_positioning",
                            "error": "interaction_position_no_progress",
                            "action_proposal": action_text,
                        },
                    )
                self.pending_graph_action = pending
                return ActionResult(
                    summary=navigation_summary,
                    execution_kind="move_precondition",
                    resolved_action=navigation_summary,
                    estimated_duration=self._walking_duration(movement_start, actor.center),
                    counts_as_intent_action=False,
                    execution_debug={
                        "failed_module": "",
                        "error": "",
                        "pending_action_text": action_text,
                        "navigation_anchor_element_id": navigation_anchor_id,
                        "world_pipeline_invoked": False,
                    },
                )
            pipe_result = self._execute_contextual_world_action_with_support(
                actor,
                engine,
                action_text,
                support_result,
                execution_context=execution_context,
            )

        summary = pipe_result.feedback_text
        result = ActionResult(
            summary=summary,
            execution_kind="action",
            execution_narration=summary,
            resolved_action=pipe_result.resolved_action or summary,
            estimated_duration=pipe_result.estimated_duration,
            temporary_element_changes=pipe_result.temporary_element_changes,
            actor_state_update=pipe_result.agent_state_patch,
            environment_feedback=pipe_result.feedback,
            execution_debug={
                "failed_module": str(getattr(pipe_result, "failed_module", "") or ""),
                "error": str(getattr(pipe_result, "error", "") or ""),
                "support_result": dict(pipe_result.support_result or {}),
                "focus_result": dict(getattr(pipe_result, "focus_result", {}) or {}),
                "world_state_transition": dict(getattr(pipe_result, "transition_result", {}) or {}),
                "transition_check": dict(getattr(pipe_result, "transition_check", {}) or {}),
                "world_action_event": dict(pipe_result.world_action_event or {}),
                "world_state_diff": dict(getattr(pipe_result, "world_state_diff", {}) or {}),
                "transition_report": (
                    pipe_result.transition_report.to_dict()
                    if hasattr(pipe_result.transition_report, "to_dict")
                    else {}
                ),
                "contextual_world_trace": list(getattr(pipe_result, "trace", []) or []),
                "route": pipe_result.route,
                "execution_context": dict(getattr(pipe_result, "execution_context", {}) or {}),
            },
        )
        if navigation_summary:
            self._prepend_movement_summary(result, navigation_summary)
        return result

    def _pending_still_needs_positioning(
        self,
        actor: Any,
        engine: PhysicsEngine,
        pending: PendingContextualWorldAction,
    ) -> bool:
        element_id = pending.anchor_element_id or self._primary_spatial_anchor_element_id(pending.support_result)
        if not element_id:
            return False
        return not self._is_near_element(actor, engine, element_id)

    def _position_actor_for_graph_action(
        self,
        actor: Any,
        engine: PhysicsEngine,
        support_result: dict,
    ) -> str:
        element_id = self._primary_spatial_anchor_element_id(support_result)
        if not element_id:
            return ""
        element = engine.get_element(element_id)
        if element is None:
            return ""
        if self._is_near_element(actor, engine, element_id):
            self._face_element(actor, element)
            return ""
        target_point = self._nearest_reachable_element_point(actor, engine, element_id)
        if target_point is None and self.disable_spatial_gating:
            target_area = engine.get_area_for_element(element_id)
            if target_area is not None:
                target_point = self._find_area_anchor(engine, target_area.node_id, actor)
        if target_point is None:
            return ""
        path = actor.find_path(engine, target_point)
        if path:
            moved = actor.follow_path(engine, path)
        elif self.disable_spatial_gating:
            moved = actor.move_to(engine, target_point).success
        else:
            return ""
        if not moved:
            return ""
        self._face_element(actor, element)
        self._mark_actor_after_movement(actor, engine, focus_element=element)
        area = engine.get_area_for_point(*actor.center)
        area_name = area.name if area is not None else "Unknown area"
        return f"{actor.name} first walks to {area_name}, then comes next to {element.name}."

    def _position_actor_for_element(
        self,
        actor: Any,
        engine: PhysicsEngine,
        element_id: str,
    ) -> str:
        return self._position_actor_for_graph_action(
            actor,
            engine,
            {"permanent_targets": [{"element_id": element_id}]},
        )

    def _prepend_movement_summary(
        self,
        action_result: ActionResult,
        movement_summary: str,
    ) -> None:
        prefix = movement_summary.strip()
        if not prefix:
            return
        if action_result.summary:
            action_result.summary = f"{prefix} {action_result.summary}"
        else:
            action_result.summary = prefix
        if action_result.execution_narration:
            action_result.execution_narration = f"{prefix} {action_result.execution_narration}"
        if action_result.resolved_action:
            action_result.resolved_action = f"{prefix} {action_result.resolved_action}"
        feedback = action_result.environment_feedback
        if feedback is not None and feedback.perception_summary.strip():
            feedback.perception_summary = f"{prefix} {feedback.perception_summary.strip()}"

    def _primary_spatial_anchor_element_id(self, support_result: dict) -> str:
        for item in support_result.get("required_existing_nodes", []) or []:
            if not isinstance(item, dict):
                continue
            node_type = str(item.get("node_type", "") or "")
            if node_type == "permanent_element" and item.get("node_id"):
                return str(item["node_id"])
            if node_type == "temporary_element" and item.get("node_id") and self.graph_pipe is not None:
                graph = self.graph_pipe.graph
                node = graph.nodes.get(str(item["node_id"])) if graph is not None else None
                if node is not None and node.state.get("anchor_element_id"):
                    return str(node.state["anchor_element_id"])
        for item in support_result.get("temporary_node_creations", []) or []:
            if isinstance(item, dict) and item.get("anchor_element_id"):
                return str(item["anchor_element_id"])
        for item in support_result.get("temporary_node_updates", []) or []:
            if isinstance(item, dict) and item.get("anchor_element_id"):
                return str(item["anchor_element_id"])
            if isinstance(item, dict) and item.get("temporary_element_id") and self.graph_pipe is not None:
                graph = self.graph_pipe.graph
                node = graph.nodes.get(str(item["temporary_element_id"])) if graph is not None else None
                if node is not None and node.state.get("anchor_element_id"):
                    return str(node.state["anchor_element_id"])
        for item in support_result.get("permanent_targets", []) or []:
            if isinstance(item, dict) and item.get("element_id"):
                return str(item["element_id"])
        for item in support_result.get("temporary_element_creations", []) or []:
            if isinstance(item, dict) and item.get("anchor_element_id"):
                return str(item["anchor_element_id"])
        for item in support_result.get("temporary_element_updates", []) or []:
            if isinstance(item, dict) and item.get("anchor_element_id"):
                return str(item["anchor_element_id"])
        return ""

    def _is_near_element(
        self,
        actor: Any,
        engine: PhysicsEngine,
        element_id: str,
        *,
        threshold: float = 0.8,
    ) -> bool:
        element = engine.get_element(element_id)
        if element is None:
            return False
        base_threshold = threshold
        threshold = self._interaction_distance_threshold(element, threshold)
        element_area = engine.get_area_for_element(element_id)
        actor_area = engine.get_area_for_point(*actor.center)
        if element_area is not None and actor_area is not None and actor_area.node_id != element_area.node_id:
            return False
        if self.disable_spatial_gating:
            return True

        # Use the actual distance to the furniture boundary before consulting
        # discrete navigation points. Long counters can have their nearest
        # perimeter samples filtered out by adjacent fixtures even when the
        # actor is already within arm's reach.
        bounds = engine.get_element_bounds(element)
        dx = max(bounds.x_min - actor.center[0], 0.0, actor.center[0] - bounds.x_max)
        dy = max(bounds.y_min - actor.center[1], 0.0, actor.center[1] - bounds.y_max)
        actor_reach = max(actor.size) / 2.0 + 0.20
        if hypot(dx, dy) <= threshold + actor_reach + 1e-6:
            return True

        if self._uses_extended_near_points(element):
            points = self._reachable_candidate_points(actor, engine, element_id)
        else:
            points = actor.element_perimeter_points(engine, element_id)
        if element_area is not None:
            points = [
                point
                for point in points
                if (area := engine.get_area_for_point(*point)) is not None
                and area.node_id == element_area.node_id
            ]
        if points and min(self._distance(actor.center, point) for point in points) <= threshold:
            return True

        # Small non-blocking items often live on top of a table/counter. In
        # that case the body should stand near the supporting furniture and
        # reach with a hand, instead of trying to stand beside the tiny item.
        for anchor_id in self._supporting_spatial_anchor_ids(engine, element_id):
            if self._is_near_element(actor, engine, anchor_id, threshold=max(base_threshold, 1.1)):
                return True
        return False

    def _interaction_distance_threshold(self, element: Any, base_threshold: float) -> float:
        size = getattr(element, "size", (0.0, 0.0)) or (0.0, 0.0)
        blocks_movement = bool(getattr(element, "blocks_movement", False))
        movable = bool(getattr(element, "movable", False))
        if not blocks_movement and (movable or max(size) <= 0.6):
            base_threshold = max(base_threshold, 1.35)
        if self.interaction_reach > 0:
            base_threshold = max(base_threshold, float(self.interaction_reach))
        return base_threshold

    def _uses_extended_near_points(self, element: Any) -> bool:
        size = getattr(element, "size", (0.0, 0.0)) or (0.0, 0.0)
        return bool(getattr(element, "blocks_movement", False)) and max(size) <= 1.0

    def _nearest_reachable_element_point(
        self,
        actor: Any,
        engine: PhysicsEngine,
        element_id: str,
    ) -> tuple[float, float] | None:
        target_ids = [element_id, *self._supporting_spatial_anchor_ids(engine, element_id)]
        seen: set[str] = set()
        for target_id in target_ids:
            if target_id in seen:
                continue
            seen.add(target_id)
            candidates = self._reachable_candidate_points(actor, engine, target_id)
            target_area = engine.get_area_for_element(target_id)
            if target_area is not None:
                same_area_candidates = [
                    point
                    for point in candidates
                    if (area := engine.get_area_for_point(*point)) is not None
                    and area.node_id == target_area.node_id
                ]
                if same_area_candidates:
                    candidates = same_area_candidates
            candidates.sort(key=lambda point: self._distance(actor.center, point))
            for point in candidates:
                if not self._point_supports_element_interaction(
                    actor,
                    engine,
                    target_id,
                    point,
                ):
                    continue
                path = actor.find_path(engine, point)
                if path:
                    return point
        return None

    def _point_supports_element_interaction(
        self,
        actor: Any,
        engine: PhysicsEngine,
        element_id: str,
        point: tuple[float, float],
        *,
        threshold: float = 0.8,
    ) -> bool:
        element = engine.get_element(element_id)
        if element is None:
            return False
        element_area = engine.get_area_for_element(element_id)
        point_area = engine.get_area_for_point(*point)
        if (
            element_area is not None
            and point_area is not None
            and point_area.node_id != element_area.node_id
        ):
            return False
        bounds = engine.get_element_bounds(element)
        dx = max(bounds.x_min - point[0], 0.0, point[0] - bounds.x_max)
        dy = max(bounds.y_min - point[1], 0.0, point[1] - bounds.y_max)
        interaction_distance = self._interaction_distance_threshold(element, threshold)
        actor_reach = max(actor.size) / 2.0 + 0.20
        if hypot(dx, dy) <= interaction_distance + actor_reach + 1e-6:
            return True
        return any(
            self._point_supports_element_interaction(
                actor,
                engine,
                anchor_id,
                point,
                threshold=max(threshold, 1.1),
            )
            for anchor_id in self._supporting_spatial_anchor_ids(engine, element_id)
        )

    def _reachable_candidate_points(
        self,
        actor: Any,
        engine: PhysicsEngine,
        element_id: str,
    ) -> list[tuple[float, float]]:
        candidates: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        clearances = [0.20, 0.45, 0.70, 1.00]
        if self.interaction_reach > 1.0:
            clearances.extend(
                value
                for value in (1.50, 2.00, 3.00)
                if value <= min(3.0, self.interaction_reach)
            )
        for clearance in clearances:
            for point in actor.element_perimeter_points(engine, element_id, clearance=clearance):
                if point in seen:
                    continue
                seen.add(point)
                candidates.append(point)
        return candidates

    def _supporting_spatial_anchor_ids(
        self,
        engine: PhysicsEngine,
        element_id: str,
    ) -> list[str]:
        element = engine.get_element(element_id)
        element_area = engine.get_area_for_element(element_id)
        if element is None or element_area is None:
            return []

        anchor_ids: list[str] = []
        visited = {element_id}
        current = element
        while current is not None:
            support_id = str(current.state_details.get("support_element_id", "") or "").strip()
            if not support_id or support_id in visited:
                break
            support = engine.get_element(support_id)
            if support is None:
                break
            visited.add(support_id)
            anchor_ids.append(support_id)
            current = support

        element_bounds = engine.get_element_bounds(element)
        element_center = element_bounds.center
        element_area_size = element.size[0] * element.size[1]
        anchors: list[tuple[float, str]] = []
        for item in engine.get_blocking_elements(exclude_node_id=element_id):
            if item.area_id != element_area.node_id:
                continue
            bounds = engine.get_element_bounds(item.element)
            if not (
                bounds.x_min <= element_center[0] <= bounds.x_max
                and bounds.y_min <= element_center[1] <= bounds.y_max
            ):
                continue
            anchor_area_size = item.element.size[0] * item.element.size[1]
            if anchor_area_size <= element_area_size:
                continue
            anchors.append((self._distance(element.center, item.element.center), item.element.node_id))
        anchors.sort(key=lambda item: item[0])
        for _, node_id in anchors:
            if node_id not in visited:
                visited.add(node_id)
                anchor_ids.append(node_id)
        return anchor_ids

    def _face_element(self, actor: Any, element: Any) -> None:
        actor.update_facing(
            element.center[0] - actor.center[0],
            element.center[1] - actor.center[1],
        )

    def _move_to_area(self, actor: Any, engine: PhysicsEngine, area_id: str) -> ActionResult:
        movement_start = actor.center
        target_point = self._find_area_anchor(engine, area_id, actor)
        if target_point is None:
            return ActionResult(summary=f"{actor.name} wants to go to {self._area_name(engine, area_id)}, but no suitable landing spot has been found yet.", execution_kind="move", resolved_action="")

        path = actor.find_path(engine, target_point)
        if not path:
            if self.disable_spatial_gating and actor.move_to(engine, target_point).success:
                self._mark_actor_after_movement(actor, engine)
                report = self._position_report(actor, engine)
                return ActionResult(
                    summary=report,
                    execution_kind="move",
                    resolved_action=report,
                    estimated_duration=self._walking_duration(movement_start, actor.center),
                )
            return ActionResult(summary=f"{actor.name} wants to go to {self._area_name(engine, area_id)}, but no walkable path was found for this step.", execution_kind="move", resolved_action="")

        moved = actor.follow_path(engine, path)
        if moved:
            self._mark_actor_after_movement(actor, engine)
            report = self._position_report(actor, engine)
            return ActionResult(
                summary=report,
                execution_kind="move",
                resolved_action=report,
                estimated_duration=self._walking_duration(movement_start, actor.center),
            )
        return ActionResult(summary=f"{actor.name} tried to head to {self._area_name(engine, area_id)}, but was blocked halfway there.", execution_kind="move", resolved_action="")

    def _walking_duration(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> str:
        distance = self._distance(start, end)
        seconds = max(10, int(round(distance / 1.2 + 4)))
        return f"{seconds}s"

    def _position_report(
        self,
        actor: Any,
        engine: PhysicsEngine,
        *,
        focus_element=None,
    ) -> str:
        area = engine.get_area_for_point(*actor.center)
        area_name = area.name if area is not None else "Unknown area"
        front_element = focus_element or self._find_front_element(actor, engine)
        if front_element is not None:
            return f"{actor.name} has now arrived at {area_name}, with {front_element.name} directly in front of them."
        return f"{actor.name} has now arrived at {area_name}, but there is no clear target immediately in front of them yet."

    def _mark_actor_after_movement(
        self,
        actor: Any,
        engine: PhysicsEngine,
        *,
        focus_element: Any = None,
    ) -> None:
        held_ids = self._held_temporary_element_ids(actor)
        actor.posture = "standing"
        actor.interaction_elements = held_ids
        actor.interaction_method = "Move"
        actor.gaze_target = getattr(focus_element, "node_id", "") if focus_element is not None else ""
        actor.text_to_motion_description = "a person walks forward, shifts weight from one foot to the other, and stops standing upright"
        if self.graph_pipe is not None and self.graph_pipe.graph is not None:
            self._clear_stale_agent_motion_facts(actor)
            self.graph_pipe.sync_actor_to_graph(actor, engine)

    def _held_temporary_element_ids(self, actor: Any) -> list[str]:
        if self.graph_pipe is None or self.graph_pipe.graph is None:
            return [
                str(item)
                for item in (getattr(actor, "interaction_elements", []) or [])
                if str(item).startswith("temporary_")
            ]
        graph = self.graph_pipe.graph
        agent_id = self._actor_node_id(actor)
        held = []
        for edge in graph.edges_for_node(agent_id, edge_kind="fact"):
            if edge.from_node_id != agent_id or edge.relation != "holding":
                continue
            node = graph.nodes.get(edge.to_node_id)
            if node is not None and node.node_type == "temporary_element" and node.state.get("visible") is not False:
                held.append(node.node_id)
        return held

    def _clear_stale_agent_motion_facts(self, actor: Any) -> None:
        if self.graph_pipe is None or self.graph_pipe.graph is None:
            return
        graph = self.graph_pipe.graph
        agent_id = self._actor_node_id(actor)
        for edge in list(graph.edges_for_node(agent_id, edge_kind="fact")):
            if edge.relation in {"holding", "located_in"}:
                continue
            graph.edges.pop(edge.edge_id, None)

    def _ensure_graph_pipe(
        self,
        actor: Any,
        engine: PhysicsEngine | None,
    ) -> ContextualWorldActionPipe:
        if engine is None:
            raise RuntimeError("Contextual world action pipe requires a PhysicsEngine.")
        if self.graph_pipe is None or self.graph_pipe.home is not engine.home:
            self.graph_pipe = ContextualWorldActionPipe(
                engine.home,
                provider_name=self.provider_name,
                model=self.model,
                agent_node_id=self._actor_node_id(actor),
                use_llm_summary=True,
                failure_output_path=self.failure_output_path,
            )
        return self.graph_pipe

    def close_world_failure_session(self):
        if self.graph_pipe is None:
            return None
        return self.graph_pipe.failure_session.close()

    def _actor_node_id(self, actor: Any) -> str:
        return (
            str(getattr(actor, "node_id", "") or "").strip()
            or str(getattr(actor, "actor_id", "") or "").strip()
            or "agent_01"
        )

    def _plan_contextual_world_support(
        self,
        actor: Any,
        engine: PhysicsEngine | None,
        action_proposal: str,
        execution_context: dict | None = None,
    ) -> dict:
        graph_pipe = self._ensure_graph_pipe(actor, engine)
        return graph_pipe.plan_support(
            actor=actor,
            engine=engine,
            action_proposal=action_proposal,
            execution_context=execution_context,
        )

    def _execute_contextual_world_action_with_support(
        self,
        actor: Any,
        engine: PhysicsEngine | None,
        action_proposal: str,
        support_result: dict,
        execution_context: dict | None = None,
    ):
        graph_pipe = self._ensure_graph_pipe(actor, engine)
        return graph_pipe.execute_with_support(
            actor=actor,
            engine=engine,
            action_proposal=action_proposal,
            support_result=support_result,
            execution_context=execution_context,
        )

    def _world_execution_context(
        self,
        actor: Any,
        decision: AgentDecision,
        *,
        pending_action: bool,
    ) -> dict:
        active_intent = getattr(actor, "active_intent", None)
        return {
            "target_resolution": self._decision_context(decision),
            "pending_action": {
                "was_pending": pending_action,
                "pending_action_text": self.pending_graph_action_text() if pending_action else "",
            },
            "intent": {
                "text": str(getattr(active_intent, "intent_text", "") or ""),
                "status": str(getattr(active_intent, "status", "") or ""),
                "step_count": getattr(active_intent, "step_count", None),
            },
            "cognition": {
                "engine_step_id": decision.cognitive_step_id,
                "intent_action_index": decision.intent_action_index,
                "intuition_route": decision.intuition_route,
                "intuition_thought": decision.intuition_thought,
                "think_thought": decision.think_thought,
                "think_conclusion": decision.think_conclusion,
                "computation_architecture": decision.computation_architecture,
                "previous_execution_result": decision.previous_execution_result,
            },
        }

    def _decision_context(self, decision: AgentDecision) -> dict:
        return {
            "action_type": decision.action_type,
            "reason": decision.reason,
            "action_proposal_text": decision.action_proposal_text,
            "target_area_id": decision.target_area_id,
            "target_element_id": decision.target_element_id,
            "navigation_target_element_id": decision.navigation_target_element_id,
            "secondary_target_element_id": decision.secondary_target_element_id,
            "target_resolution_trace": dict(decision.target_resolution_trace or {}),
        }

    def _record_world_executor_failure(
        self,
        actor: Any,
        engine: PhysicsEngine,
        *,
        failed_module: str,
        error: str,
        action_proposal: str,
        world_feedback: str,
        execution_context: dict | None = None,
        support_result: dict | None = None,
        count_action_attempt: bool = False,
    ) -> None:
        graph_pipe = self._ensure_graph_pipe(actor, engine)
        graph_pipe.record_external_failure(
            actor=actor,
            failed_module=failed_module,
            error=error,
            action_proposal=action_proposal,
            world_feedback=world_feedback,
            execution_context=execution_context,
            support_result=support_result,
            count_action_attempt=count_action_attempt,
        )

    def _find_front_element(self, actor: Any, engine: PhysicsEngine):
        nearby = engine.get_nearby_elements(center=actor.center, radius=1.8)
        if not nearby:
            return None
        current_area = engine.get_area_for_point(*actor.center)
        if current_area is not None:
            same_area_nearby = [item for item in nearby if item.area_id == current_area.node_id]
            if same_area_nearby:
                nearby = same_area_nearby

        facing_rad = radians(actor.facing)
        facing_vector = (cos(facing_rad), sin(facing_rad))
        best = None
        best_score = None
        for item in nearby:
            dx = item.element.center[0] - actor.center[0]
            dy = item.element.center[1] - actor.center[1]
            distance = hypot(dx, dy)
            if distance == 0:
                continue
            dot = (dx * facing_vector[0] + dy * facing_vector[1]) / distance
            if dot <= 0:
                continue
            score = (dot, -distance)
            if best is None or score > best_score:
                best = item.element
                best_score = score

        if best is not None:
            return best

        nearby.sort(key=lambda item: self._distance(actor.center, item.element.center))
        return nearby[0].element

    def _find_area_anchor(
        self,
        engine: PhysicsEngine,
        area_id: str,
        actor: Any,
        *,
        resolution: float = 0.25,
    ) -> tuple[float, float] | None:
        area = engine.get_area(area_id)
        if area is None:
            return None

        x_min, y_min, x_max, y_max = area.bounds
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        samples = [
            (center_x, center_y),
            (center_x - 0.5, center_y),
            (center_x + 0.5, center_y),
            (center_x, center_y - 0.5),
            (center_x, center_y + 0.5),
            (x_min + 0.5, y_min + 0.5),
            (x_max - 0.5, y_min + 0.5),
            (x_min + 0.5, y_max - 0.5),
            (x_max - 0.5, y_max - 0.5),
        ]

        for sample in samples:
            snapped = actor._snap(sample, resolution)
            if engine.can_place_box(new_center=snapped, size=actor.size).success:
                return snapped
        return None

    def _area_name(self, engine: PhysicsEngine, area_id: str) -> str:
        area = engine.get_area(area_id)
        return area.name if area is not None else area_id

    def _distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return hypot(a[0] - b[0], a[1] - b[1])

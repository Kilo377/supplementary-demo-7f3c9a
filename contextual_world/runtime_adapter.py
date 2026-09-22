from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.feedback_types import EnvironmentFeedback
from contextual_world.modules.node_support import run_world_node_support
from contextual_world.modules.node_support.variables import build_world_node_support_variables
from contextual_world.failures import WorldFailureSession
from contextual_world.pipeline import (
    ContextualWorldPipeline,
    ContextualWorldPipelineFailure,
    ContextualWorldTransitionRejected,
    ContextualWorldUnsupportedAction,
)
from contextual_world.world_old.graph.world_graph import WorldGraph
from contextual_world.world_old.graph.world_graph_transition import WorldGraphTransitionReport
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from contextual_world.structure.scene_schema import Home
from contextual_world.types import ContextualWorldModuleError, ContextualWorldTraceStep


@dataclass
class ContextualWorldActionPipeResult:
    route: str = "contextual_world"
    resolved_action: str = ""
    failed_module: str = ""
    error: str = ""
    feedback: EnvironmentFeedback | None = None
    transition_report: WorldGraphTransitionReport = field(default_factory=WorldGraphTransitionReport)
    agent_state_patch: dict = field(default_factory=dict)
    temporary_element_changes: list[dict] = field(default_factory=list)
    support_result: dict = field(default_factory=dict)
    focus_result: dict = field(default_factory=dict)
    transition_result: dict = field(default_factory=dict)
    transition_check: dict = field(default_factory=dict)
    world_state_diff: dict = field(default_factory=dict)
    world_action_event: dict = field(default_factory=dict)
    estimated_duration: str = ""
    trace: list[dict] = field(default_factory=list)
    execution_context: dict = field(default_factory=dict)

    @property
    def feedback_text(self) -> str:
        if self.feedback is not None:
            return self.feedback.adapter_text()
        return self.resolved_action


@dataclass
class ContextualWorldActionPipe:
    home: Home
    provider_name: str = "ollama"
    model: str | None = None
    agent_node_id: str = "agent_01"
    use_llm_summary: bool = True
    graph: WorldGraph | None = None
    failure_output_path: str | None = None
    failure_session: WorldFailureSession = field(init=False, repr=False)
    _support_trace_by_action: dict[str, ContextualWorldTraceStep] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _execution_context_by_action: dict[str, dict] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        session_kwargs = dict(
            scene=self.home.node_id,
            provider=self.provider_name,
            model=self.model or "provider default",
        )
        if self.failure_output_path:
            session_kwargs["output_path"] = self.failure_output_path
        self.failure_session = WorldFailureSession(**session_kwargs)

    def ensure_graph(self, actor: Any, engine: PhysicsEngine) -> WorldGraph:
        if self.graph is None:
            self.graph = WorldGraph.from_scene(self.home, actor=actor)
        self.sync_actor_to_graph(actor, engine)
        return self.graph

    def reset(self) -> None:
        self.graph = None
        self._support_trace_by_action.clear()
        self._execution_context_by_action.clear()

    def plan_support(
        self,
        *,
        actor: Any,
        engine: PhysicsEngine,
        action_proposal: str,
        execution_context: dict | None = None,
    ) -> dict:
        graph = self.ensure_graph(actor, engine)
        self.failure_session.observe_action_attempt()
        action_key = str(action_proposal or "")
        self._execution_context_by_action[action_key] = dict(execution_context or {})
        variables = build_world_node_support_variables(
            graph=graph,
            agent_node_id=self.agent_node_id,
            action_proposal=action_proposal,
        )
        try:
            support, support_trace = run_world_node_support(
                variables,
                provider_name=self.provider_name,
                model=self.model,
            )
        except Exception as error:
            trace = (
                [error.trace_step.to_dict()]
                if isinstance(error, ContextualWorldModuleError)
                else []
            )
            self.failure_session.record_failure({
                "failure_kind": "pipeline_error",
                "failed_module": "world_node_support",
                "error": str(error),
                "world_route": "world_node_support",
                "action_proposal": action_proposal,
                "world_feedback": "",
                "world_judgments": {
                    "node_support": {},
                    "interaction_focus": {},
                    "state_transition": {},
                    "transition_check": {},
                    "world_action_event": {},
                    "warnings": [str(error)],
                },
                "contextual_world_trace": trace,
                "graph_transition_report": {},
                "actor_state": self._actor_state_for_failure(actor),
                "execution_context": dict(execution_context or {}),
            })
            self._execution_context_by_action.pop(action_key, None)
            raise
        self._support_trace_by_action[action_key] = support_trace
        return support.to_dict()

    def execute_with_support(
        self,
        *,
        actor: Any,
        engine: PhysicsEngine,
        action_proposal: str,
        support_result: dict,
        execution_context: dict | None = None,
    ) -> ContextualWorldActionPipeResult:
        graph = self.ensure_graph(actor, engine)
        action_key = str(action_proposal or "")
        was_planned = (
            action_key in self._support_trace_by_action
            or action_key in self._execution_context_by_action
        )
        if not was_planned:
            self.failure_session.observe_action_attempt()
        support_trace = self._support_trace_by_action.pop(action_key, None)
        stored_context = self._execution_context_by_action.pop(action_key, {})
        current_execution_context = dict(execution_context or stored_context or {})
        support_dict = dict(support_result or {})
        if support_dict.get("support_status") == "unsupported":
            return self._unsupported_result(
                actor=actor,
                action_proposal=action_proposal,
                support_result=support_dict,
                execution_context=current_execution_context,
            )
        pipeline = ContextualWorldPipeline(
            graph=graph,
            agent_node_id=self.agent_node_id,
            provider_name=self.provider_name,
            model=self.model,
            use_llm_feedback_summary=self.use_llm_summary,
            raise_unsupported=True,
            raise_check_rejected=True,
        )
        try:
            result = pipeline.process_action(
                action_proposal,
                support_result=support_dict,
                support_trace=support_trace,
            )
        except ContextualWorldUnsupportedAction as error:
            return self._error_result(
                actor=actor,
                route="world_node_support",
                action_proposal=action_proposal,
                error=error,
                support_result=support_dict,
                execution_context=current_execution_context,
            )
        except ContextualWorldTransitionRejected as error:
            return self._error_result(
                actor=actor,
                route="world_transition_check",
                action_proposal=action_proposal,
                error=error,
                support_result=support_dict,
                diagnostics=error.diagnostics(),
                execution_context=current_execution_context,
            )
        except ContextualWorldPipelineFailure as error:
            return self._error_result(
                actor=actor,
                route=error.failed_module or "contextual_world_error",
                action_proposal=action_proposal,
                error=error,
                support_result=support_dict,
                diagnostics=error.diagnostics(),
                execution_context=current_execution_context,
            )
        except Exception as error:
            return self._error_result(
                actor=actor,
                route="contextual_world_error",
                action_proposal=action_proposal,
                error=error,
                support_result=support_dict,
                execution_context=current_execution_context,
            )

        self._sync_updated_permanent_elements_to_scene(result.transition_report)
        self.sync_actor_from_graph(actor)
        self.sync_actor_to_graph(actor, engine)
        transition_report = result.transition_report
        world_action_event = dict(transition_report.world_action_event or {})
        return ContextualWorldActionPipeResult(
            route="contextual_world",
            resolved_action=result.actual_event or result.feedback_text(),
            feedback=result.feedback,
            transition_report=transition_report,
            agent_state_patch=self._agent_state_patch_from_transition(result.transition_result),
            temporary_element_changes=self._temporary_changes_from_report(transition_report),
            support_result=result.support_result,
            focus_result=result.focus_result,
            transition_result=result.transition_result,
            transition_check=result.transition_check,
            world_state_diff=result.world_state_diff,
            world_action_event=world_action_event,
            estimated_duration=result.estimated_duration,
            trace=[step.to_dict() for step in result.trace],
            execution_context=current_execution_context,
        )

    def process(
        self,
        *,
        actor: Any,
        engine: PhysicsEngine,
        action_proposal: str,
        execution_context: dict | None = None,
    ) -> ContextualWorldActionPipeResult:
        support = self.plan_support(
            actor=actor,
            engine=engine,
            action_proposal=action_proposal,
            execution_context=execution_context,
        )
        return self.execute_with_support(
            actor=actor,
            engine=engine,
            action_proposal=action_proposal,
            support_result=support,
            execution_context=execution_context,
        )

    def record_external_failure(
        self,
        *,
        actor: Any,
        failed_module: str,
        error: str,
        action_proposal: str,
        world_feedback: str,
        execution_context: dict | None = None,
        support_result: dict | None = None,
        count_action_attempt: bool = False,
    ) -> None:
        if count_action_attempt:
            self.failure_session.observe_action_attempt()
        self.failure_session.record_failure({
            "failure_kind": "executor_error",
            "failed_module": failed_module,
            "error": error,
            "world_route": failed_module,
            "action_proposal": action_proposal,
            "world_feedback": world_feedback,
            "world_judgments": {
                "node_support": dict(support_result or {}),
                "interaction_focus": {},
                "state_transition": {},
                "transition_check": {},
                "world_action_event": {},
                "warnings": [error],
            },
            "contextual_world_trace": [],
            "graph_transition_report": {},
            "actor_state": self._actor_state_for_failure(actor),
            "execution_context": dict(execution_context or {}),
        })

    def sync_actor_to_graph(self, actor: Any, engine: PhysicsEngine) -> None:
        if self.graph is None:
            return
        area = engine.get_area_for_point(*actor.center)
        current_area_id = area.node_id if area is not None else getattr(actor, "current_area_id", "") or ""
        updates = {
            "center": list(actor.center),
            "size": list(getattr(actor, "size", []) or []),
            "facing": getattr(actor, "facing", 0.0),
            "current_area_id": current_area_id,
            "posture": getattr(actor, "posture", "standing"),
            "interaction_elements": list(getattr(actor, "interaction_elements", []) or []),
            "interaction_method": getattr(actor, "interaction_method", "") or "",
            "gaze_target": getattr(actor, "gaze_target", "") or "",
            "worn_items": list(getattr(actor, "worn_items", []) or []),
            "body_surface": getattr(actor, "body_surface", "dry_clean") or "dry_clean",
            "text_to_motion_description": getattr(actor, "text_to_motion_description", "") or "",
        }
        self.graph.update_node_state(self.agent_node_id, updates)
        self.graph.remove_edges(
            from_node_id=self.agent_node_id,
            relation="located_in",
            edge_kind="fact",
        )
        if current_area_id:
            self.graph.add_fact_edge(
                from_node_id=self.agent_node_id,
                relation="located_in",
                to_node_id=current_area_id,
            )
        actor.current_area_id = current_area_id

    def sync_actor_from_graph(self, actor: Any) -> None:
        if self.graph is None or self.agent_node_id not in self.graph.nodes:
            return
        state = self.graph.nodes[self.agent_node_id].state
        for key in [
            "posture",
            "interaction_method",
            "gaze_target",
            "body_surface",
            "text_to_motion_description",
        ]:
            if key in state:
                setattr(actor, key, state[key])
        if "interaction_elements" in state:
            actor.interaction_elements = list(state.get("interaction_elements") or [])
        if "worn_items" in state:
            actor.worn_items = list(state.get("worn_items") or [])

    def _unsupported_result(
        self,
        *,
        actor: Any,
        action_proposal: str,
        support_result: dict,
        execution_context: dict | None = None,
    ) -> ContextualWorldActionPipeResult:
        fallback = support_result.get("fallback_result", {}) if isinstance(support_result, dict) else {}
        message = (
            str(fallback.get("actual_event", "") or "").strip()
            or str(support_result.get("support_reason", "") or "").strip()
            or f"{actor.name}这一步缺少必要环境支持，动作没有落实。"
        )
        report = WorldGraphTransitionReport(
            world_action_event={
                "accepted": False,
                "actual_event": message,
                "event_effects": [],
                "estimated_duration": "10s",
                "rejection": {
                    "reason": str(fallback.get("reason", "") or support_result.get("support_reason", "") or ""),
                    "state_statement": message,
                },
            }
        )
        feedback = EnvironmentFeedback(
            route="unsupported",
            perception_summary=message,
            graph_transition_report=report.to_dict(),
            world_state_diff={},
        )
        return ContextualWorldActionPipeResult(
            route="unsupported",
            resolved_action=message,
            feedback=feedback,
            transition_report=report,
            support_result=support_result,
            world_action_event=report.world_action_event,
            estimated_duration="10s",
            execution_context=dict(execution_context or {}),
        )

    def _error_result(
        self,
        *,
        actor: Any,
        route: str,
        action_proposal: str,
        error: Exception,
        support_result: dict | None = None,
        diagnostics: dict | None = None,
        execution_context: dict | None = None,
    ) -> ContextualWorldActionPipeResult:
        diagnostics = dict(diagnostics or {})
        message = f"{actor.name}这一步没有得到稳定的环境反馈，动作暂时没有落实。"
        report = WorldGraphTransitionReport(
            warnings=[f"{route}: {error}"],
            world_action_event={
                "accepted": False,
                "actual_event": message,
                "event_effects": [],
                "estimated_duration": "10s",
                "rejection": {
                    "reason": str(error),
                    "state_statement": message,
                },
            },
        )
        feedback = EnvironmentFeedback(
            route=route,
            perception_summary=message,
            graph_transition_report=report.to_dict(),
            world_state_diff={},
        )
        result = ContextualWorldActionPipeResult(
            route=route,
            resolved_action=message,
            failed_module=str(diagnostics.get("failed_module") or route),
            error=str(diagnostics.get("error") or error),
            feedback=feedback,
            transition_report=report,
            support_result=dict(diagnostics.get("support_result") or support_result or {}),
            focus_result=dict(diagnostics.get("focus_result") or {}),
            transition_result=dict(diagnostics.get("world_state_transition") or {}),
            transition_check=dict(diagnostics.get("transition_check") or {}),
            world_action_event=report.world_action_event,
            estimated_duration="10s",
            trace=list(diagnostics.get("contextual_world_trace") or []),
            execution_context=dict(execution_context or {}),
        )
        self.failure_session.record_failure({
            "failure_kind": "pipeline_error",
            "failed_module": result.failed_module,
            "error": result.error,
            "world_route": route,
            "action_proposal": action_proposal,
            "world_feedback": message,
            "world_judgments": {
                "node_support": result.support_result,
                "interaction_focus": result.focus_result,
                "state_transition": result.transition_result,
                "transition_check": result.transition_check,
                "world_action_event": result.world_action_event,
                "warnings": list(report.warnings or []),
            },
            "contextual_world_trace": result.trace,
            "graph_transition_report": report.to_dict(),
            "actor_state": self._actor_state_for_failure(actor),
            "execution_context": result.execution_context,
        })
        return result

    def _actor_state_for_failure(self, actor: Any) -> dict:
        return {
            "node_id": str(getattr(actor, "node_id", "") or ""),
            "name": str(getattr(actor, "name", "") or ""),
            "position": list(getattr(actor, "center", ()) or ()),
            "current_area_id": str(getattr(actor, "current_area_id", "") or ""),
            "posture": str(getattr(actor, "posture", "") or ""),
            "interaction_elements": list(getattr(actor, "interaction_elements", []) or []),
            "interaction_method": str(getattr(actor, "interaction_method", "") or ""),
            "gaze_target": str(getattr(actor, "gaze_target", "") or ""),
            "worn_items": list(getattr(actor, "worn_items", []) or []),
            "body_surface": str(getattr(actor, "body_surface", "") or ""),
        }

    def _sync_updated_permanent_elements_to_scene(
        self,
        report: WorldGraphTransitionReport,
    ) -> None:
        if self.graph is None:
            return
        for item in report.updated_element_nodes:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id", "") or "").strip()
            node = self.graph.nodes.get(node_id)
            if node is None or node.node_type != "permanent_element":
                continue
            element = self.home.find_element(node_id)
            if element is None:
                continue
            state = node.state
            if state.get("physical_status"):
                element.set_physical_status(str(state["physical_status"]))
            if state.get("evolution_status"):
                element.set_evolution_status(str(state["evolution_status"]))
            if state.get("interaction_status"):
                element.set_interaction_status(str(state["interaction_status"]))
            details = state.get("state_details")
            if isinstance(details, dict):
                element.update_state_details({
                    str(key): str(value)
                    for key, value in details.items()
                    if str(key).strip() and str(value).strip()
                })

    def _temporary_changes_from_report(self, report: WorldGraphTransitionReport) -> list[dict]:
        changes = []
        for node in report.created_temporary_nodes:
            changes.append({
                "change_type": "created",
                "temporary_element_id": node.get("node_id", ""),
                "name": node.get("name", ""),
                "new_status": (node.get("state", {}) or {}).get("status", ""),
            })
        for item in report.updated_temporary_nodes:
            old_state = item.get("old_state", {}) or {}
            new_state = item.get("new_state", {}) or {}
            node_id = item.get("node_id", "")
            name = self.graph.nodes[node_id].name if self.graph is not None and node_id in self.graph.nodes else node_id
            changes.append({
                "change_type": "updated",
                "temporary_element_id": node_id,
                "name": name,
                "old_status": old_state.get("status", ""),
                "new_status": new_state.get("status", ""),
            })
        return changes

    def _agent_state_patch_from_transition(self, transition_result: dict) -> dict:
        delta = transition_result.get("next_state_delta", {}) if isinstance(transition_result, dict) else {}
        patch = {}
        for item in delta.get("node_updates", []) or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("node_id", "") or "") != self.agent_node_id:
                continue
            state_patch = item.get("state_patch", {})
            if isinstance(state_patch, dict):
                patch.update(state_patch)
        return patch

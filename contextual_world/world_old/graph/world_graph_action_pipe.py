from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from core.feedback_types import EnvironmentFeedback
from contextual_world.world_old.agent_interaction import judge_agent_environment_element_support
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from contextual_world.structure.scene_schema import Home

from .agent_body_state_update import judge_agent_body_state_update
from .world_action_event import generate_world_action_event
from .world_element_state_update import judge_world_element_state_update
from .world_feedback_projection import build_feedback_current_state, generate_world_feedback
from .world_graph import WorldGraph
from .world_state_diff import build_agent_centered_world_state_diff
from .world_state_transition_check import check_world_state_transition
from .world_graph_transition import WorldGraphTransitionApplier, WorldGraphTransitionReport
from .world_relation_update import judge_world_relation_update


@dataclass
class WorldGraphActionPipeResult:
    route: str = ""
    resolved_action: str = ""
    feedback: EnvironmentFeedback | None = None
    transition_report: WorldGraphTransitionReport = field(default_factory=WorldGraphTransitionReport)
    agent_state_patch: dict = field(default_factory=dict)
    temporary_element_changes: list[dict] = field(default_factory=list)
    support_result: dict = field(default_factory=dict)
    relation_update_result: dict = field(default_factory=dict)
    element_state_update_result: dict = field(default_factory=dict)
    world_action_event: dict = field(default_factory=dict)
    estimated_duration: str = ""

    @property
    def feedback_text(self) -> str:
        if self.feedback is not None:
            return self.feedback.adapter_text()
        return self.resolved_action


@dataclass
class WorldGraphActionPipe:
    home: Home
    provider_name: str = "ollama"
    model: str | None = None
    agent_node_id: str = "agent_01"
    use_llm_summary: bool = True
    graph: WorldGraph | None = None

    def ensure_graph(self, actor: Any, engine: PhysicsEngine) -> WorldGraph:
        if self.graph is None:
            self.graph = WorldGraph.from_scene(self.home, actor=actor)
        self.sync_actor_to_graph(actor, engine)
        return self.graph

    def reset(self) -> None:
        self.graph = None

    def process(
        self,
        *,
        actor: Any,
        engine: PhysicsEngine,
        action_proposal: str,
    ) -> WorldGraphActionPipeResult:
        try:
            support_dict = self.plan_support(
                actor=actor,
                engine=engine,
                action_proposal=action_proposal,
            )
        except Exception as error:
            return self._error_result(
                actor=actor,
                route="element_support_error",
                action_proposal=action_proposal,
                error=error,
            )
        return self.execute_with_support(
            actor=actor,
            engine=engine,
            action_proposal=action_proposal,
            support_result=support_dict,
        )

    def plan_support(
        self,
        *,
        actor: Any,
        engine: PhysicsEngine,
        action_proposal: str,
    ) -> dict:
        graph = self.ensure_graph(actor, engine)
        support = judge_agent_environment_element_support(
            agent_name=actor.name,
            action_proposal=action_proposal,
            agent_state_for_support=self._agent_state_for_support(graph),
            current_agent_facts=self._current_agent_fact_edges(graph),
            permanent_elements=self._permanent_elements_for_support(graph),
            temporary_elements=self._temporary_elements_for_support(graph),
            provider_name=self.provider_name,
            model=self.model,
            print_output=False,
        )
        support_dict = support.__dict__.copy()
        support_dict.pop("raw_response", None)
        return support_dict

    def execute_with_support(
        self,
        *,
        actor: Any,
        engine: PhysicsEngine,
        action_proposal: str,
        support_result: dict,
    ) -> WorldGraphActionPipeResult:
        graph = self.ensure_graph(actor, engine)
        support_dict = dict(support_result or {})
        route = str(support_dict.get("route", "") or "")

        if not support_dict:
            return self._error_result(
                actor=actor,
                route="element_support_error",
                action_proposal=action_proposal,
                error=RuntimeError("Missing planned support result."),
            )

        if route == "reject":
            feedback = EnvironmentFeedback(
                route="reject",
                perception_summary=f"{actor.name}这一步没有真正完成：{support_dict.get('reason', '')}",
                graph_transition_report={},
            )
            return WorldGraphActionPipeResult(
                route="reject",
                resolved_action=feedback.perception_summary,
                feedback=feedback,
                support_result=support_dict,
            )

        applier = WorldGraphTransitionApplier(graph, agent_node_id=self.agent_node_id)
        if route == "agent_body_action":
            try:
                return self._process_agent_body_action(
                    actor=actor,
                    engine=engine,
                    graph=graph,
                    applier=applier,
                    action_proposal=action_proposal,
                    support_dict=support_dict,
                )
            except Exception as error:
                return self._error_result(
                    actor=actor,
                    route="agent_body_update_error",
                    action_proposal=action_proposal,
                    error=error,
                    support_result=support_dict,
                )

        try:
            return self._process_element_interaction(
                actor=actor,
                engine=engine,
                graph=graph,
                applier=applier,
                action_proposal=action_proposal,
                support_dict=support_dict,
            )
        except Exception as error:
            return self._error_result(
                actor=actor,
                route="element_interaction_error",
                action_proposal=action_proposal,
                error=error,
                support_result=support_dict,
            )

    def _error_result(
        self,
        *,
        actor: Any,
        route: str,
        action_proposal: str,
        error: Exception,
        support_result: dict | None = None,
    ) -> WorldGraphActionPipeResult:
        message = f"{actor.name}这一步没有得到稳定的环境反馈，动作暂时没有落实。"
        report = WorldGraphTransitionReport(warnings=[f"{route}: {error}"])
        feedback = EnvironmentFeedback(
            route=route,
            perception_summary=message,
            graph_transition_report=report.to_dict(),
        )
        return WorldGraphActionPipeResult(
            route=route,
            resolved_action=message,
            feedback=feedback,
            transition_report=report,
            support_result=dict(support_result or {}),
        )

    def _event_rejected_result(
        self,
        *,
        actor: Any,
        action_proposal: str,
        world_event: dict,
        support_result: dict,
    ) -> WorldGraphActionPipeResult:
        reason = str(world_event.get("reason", "") or "").strip()
        actual_event = str(world_event.get("actual_event", "") or "").strip()
        effects = [
            str(item).strip()
            for item in (world_event.get("event_effects", []) or [])
            if str(item).strip()
        ]
        message = actual_event or "。".join(effects[:2]) or reason or f"{actor.name}这一步的状态没有明显改变。"
        feedback = EnvironmentFeedback(
            route="reject",
            perception_summary=message,
            graph_transition_report={
                "world_action_event": dict(world_event or {}),
            },
            world_state_diff={},
        )
        return WorldGraphActionPipeResult(
            route="reject",
            resolved_action=message,
            feedback=feedback,
            support_result=dict(support_result or {}),
            world_action_event=dict(world_event or {}),
            estimated_duration=str(world_event.get("estimated_duration", "") or ""),
        )

    def _process_agent_body_action(
        self,
        *,
        actor: Any,
        engine: PhysicsEngine,
        graph: WorldGraph,
        applier: WorldGraphTransitionApplier,
        action_proposal: str,
        support_dict: dict,
    ) -> WorldGraphActionPipeResult:
        relevant_nodes = self._select_relevant_nodes(graph, support_dict)
        current_fact_edges = [
            edge.to_dict()
            for edge in graph.edges.values()
            if edge.edge_kind == "fact"
        ]
        element_recent_history = self._element_recent_history_for_nodes(graph, relevant_nodes)
        world_event = generate_world_action_event(
            agent_name=actor.name,
            action_proposal=action_proposal,
            route=str(support_dict.get("route", "") or "agent_body_action"),
            element_support_result=support_dict,
            agent_node=graph.nodes[self.agent_node_id].to_dict(),
            relevant_nodes=relevant_nodes,
            current_fact_edges=current_fact_edges,
            element_recent_history=element_recent_history,
            provider_name=self.provider_name,
            model=self.model,
            print_output=False,
        )
        if not world_event.accepted:
            return self._event_rejected_result(
                actor=actor,
                action_proposal=action_proposal,
                world_event=world_event.to_dict(),
                support_result=support_dict,
            )
        body_update = judge_agent_body_state_update(
            agent_name=actor.name,
            action_proposal=action_proposal,
            world_action_event=world_event.to_dict(),
            agent_node=graph.nodes[self.agent_node_id].to_dict(),
            current_agent_fact_edges=self._current_agent_fact_edges(graph),
            context_facts=support_dict.get("context_facts", []) or [],
            provider_name=self.provider_name,
            model=self.model,
            print_output=False,
        )
        relation_dict = body_update.to_relation_update_dict()
        before_graph = copy.deepcopy(graph)
        planned_transition = {
            "world_action_event": world_event.to_dict(),
            "relation_update_result": relation_dict,
            "element_state_update_result": {"element_state_updates": []},
        }
        transition_check = check_world_state_transition(
            previous_graph=before_graph,
            action_proposal=action_proposal,
            support_result=support_dict,
            planned_transition=planned_transition,
            recent_transition_history=[],
        )
        apply_report = applier.apply_relation_update(
            relation_update_result=relation_dict,
        )
        apply_report.world_action_event = world_event.to_dict()
        apply_report.transition_check = transition_check.to_dict()
        self.sync_actor_from_graph(actor)
        self.sync_actor_to_graph(actor, engine)
        feedback = self._build_feedback(
            actor=actor,
            graph=graph,
            action_proposal=action_proposal,
            route=str(support_dict.get("route", "") or "agent_body_action"),
            transition_report=apply_report,
            agent_state_patch=body_update.agent_state_patch,
            before_graph=before_graph,
        )
        return WorldGraphActionPipeResult(
            route="agent_body_action",
            resolved_action=world_event.actual_event or body_update.feedback_hint or feedback.adapter_text(),
            feedback=feedback,
            transition_report=apply_report,
            agent_state_patch=body_update.agent_state_patch,
            temporary_element_changes=self._temporary_changes_from_report(apply_report),
            support_result=support_dict,
            relation_update_result=relation_dict,
            element_state_update_result={"element_state_updates": []},
            world_action_event=world_event.to_dict(),
            estimated_duration=world_event.estimated_duration,
        )

    def _process_element_interaction(
        self,
        *,
        actor: Any,
        engine: PhysicsEngine,
        graph: WorldGraph,
        applier: WorldGraphTransitionApplier,
        action_proposal: str,
        support_dict: dict,
    ) -> WorldGraphActionPipeResult:
        before_graph = copy.deepcopy(graph)
        materialize_report = WorldGraphTransitionReport()
        materialized_support = applier.materialize_element_support(
            action_proposal=action_proposal,
            element_support_result=support_dict,
            report=materialize_report,
        )
        relevant_nodes = self._select_relevant_nodes(graph, materialized_support)
        current_fact_edges = [
            edge.to_dict()
            for edge in graph.edges.values()
            if edge.edge_kind == "fact"
        ]
        element_recent_history = self._element_recent_history_for_nodes(graph, relevant_nodes)
        world_event = generate_world_action_event(
            agent_name=actor.name,
            action_proposal=action_proposal,
            route=str(materialized_support.get("route", "") or "element_interaction"),
            element_support_result=materialized_support,
            agent_node=graph.nodes[self.agent_node_id].to_dict(),
            relevant_nodes=relevant_nodes,
            current_fact_edges=current_fact_edges,
            element_recent_history=element_recent_history,
            provider_name=self.provider_name,
            model=self.model,
            print_output=False,
        )
        if not world_event.accepted:
            return self._event_rejected_result(
                actor=actor,
                action_proposal=action_proposal,
                world_event=world_event.to_dict(),
                support_result=materialized_support,
            )
        relation_update = judge_world_relation_update(
            agent_name=actor.name,
            action_proposal=action_proposal,
            world_action_event=world_event.to_dict(),
            element_support_result=materialized_support,
            agent_node=graph.nodes[self.agent_node_id].to_dict(),
            relevant_nodes=relevant_nodes,
            current_fact_edges=current_fact_edges,
            provider_name=self.provider_name,
            model=self.model,
            print_output=False,
        )
        relation_dict = relation_update.__dict__.copy()
        relation_dict.pop("raw_response", None)
        element_state_update = judge_world_element_state_update(
            agent_name=actor.name,
            action_proposal=action_proposal,
            world_action_event=world_event.to_dict(),
            element_support_result=materialized_support,
            relation_update_result=relation_dict,
            relevant_nodes=relevant_nodes,
            element_recent_history=element_recent_history,
            provider_name=self.provider_name,
            model=self.model,
            print_output=False,
        )
        element_state_dict = element_state_update.__dict__.copy()
        element_state_dict.pop("raw_response", None)
        planned_transition = {
            "world_action_event": world_event.to_dict(),
            "relation_update_result": relation_dict,
            "element_state_update_result": element_state_dict,
        }
        transition_check = check_world_state_transition(
            previous_graph=before_graph,
            action_proposal=action_proposal,
            support_result=materialized_support,
            planned_transition=planned_transition,
            recent_transition_history=[],
        )
        apply_report = applier.apply_relation_update(
            relation_update_result=relation_dict,
        )
        element_state_report = applier.apply_element_state_update(
            element_state_update_result=element_state_dict,
            action_proposal=action_proposal,
            world_action_event=world_event.to_dict(),
        )
        transition_report = self._merge_reports(
            self._merge_reports(materialize_report, apply_report),
            element_state_report,
        )
        transition_report.world_action_event = world_event.to_dict()
        transition_report.transition_check = transition_check.to_dict()
        self._sync_updated_permanent_elements_to_scene(transition_report)
        self.sync_actor_from_graph(actor)
        self.sync_actor_to_graph(actor, engine)
        feedback = self._build_feedback(
            actor=actor,
            graph=graph,
            action_proposal=action_proposal,
            route=str(materialized_support.get("route", "") or "element_interaction"),
            transition_report=transition_report,
            agent_state_patch=relation_dict.get("agent_state_patch", {}) or {},
            before_graph=before_graph,
        )
        return WorldGraphActionPipeResult(
            route="element_interaction",
            resolved_action=world_event.actual_event or relation_update.feedback_narration or feedback.adapter_text(),
            feedback=feedback,
            transition_report=transition_report,
            agent_state_patch=relation_dict.get("agent_state_patch", {}) or {},
            temporary_element_changes=self._temporary_changes_from_report(transition_report),
            support_result=materialized_support,
            relation_update_result=relation_dict,
            element_state_update_result=element_state_dict,
            world_action_event=world_event.to_dict(),
            estimated_duration=world_event.estimated_duration,
        )

    def _build_feedback(
        self,
        *,
        actor: Any,
        graph: WorldGraph,
        action_proposal: str,
        route: str,
        transition_report: WorldGraphTransitionReport,
        agent_state_patch: dict,
        before_graph: WorldGraph | None = None,
    ) -> EnvironmentFeedback:
        report_dict = transition_report.to_dict()
        feedback_state = build_feedback_current_state(
            graph,
            agent_node_id=self.agent_node_id,
            transition_report=report_dict,
        )
        world_state_diff = {}
        if before_graph is not None:
            world_state_diff = build_agent_centered_world_state_diff(
                before_graph=before_graph,
                after_graph=graph,
                transition_report=report_dict,
                agent_node_id=self.agent_node_id,
            ).to_dict()
        # TODO: after action-driven graph transition is stable, run an optional
        # graph-based commonsense evolution tick here or at the engine boundary.
        # It should consume the updated WorldGraph and emit another transition
        # report instead of mutating scene elements directly.
        return generate_world_feedback(
            agent_name=actor.name,
            action_proposal=action_proposal,
            current_state=feedback_state,
            route=route,
            agent_state_patch=agent_state_patch,
            graph_transition_report=report_dict,
            world_state_diff=world_state_diff,
            provider_name=self.provider_name,
            model=self.model,
            print_output=False,
            use_llm_summary=self.use_llm_summary,
        )

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

    def _agent_state_for_support(self, graph: WorldGraph) -> dict:
        agent_node = graph.nodes.get(self.agent_node_id)
        state = dict(agent_node.state) if agent_node is not None else {}
        held = []
        for edge in graph.edges_for_node(self.agent_node_id, edge_kind="fact"):
            if edge.from_node_id != self.agent_node_id or edge.relation != "holding":
                continue
            node = graph.nodes.get(edge.to_node_id)
            if node is not None and node.node_type == "temporary_element":
                held.append(node.to_dict())
        state["held_temporary_elements"] = held
        return state

    def _current_agent_fact_edges(self, graph: WorldGraph) -> list[dict]:
        return [
            edge.to_dict()
            for edge in graph.edges_for_node(self.agent_node_id, edge_kind="fact")
        ]

    def _permanent_elements_for_support(self, graph: WorldGraph) -> list[dict]:
        elements = []
        for node in graph.nodes.values():
            if node.node_type != "permanent_element":
                continue
            elements.append({
                "element_id": node.node_id,
                "element_name": node.name,
                "physical_status": node.state.get("physical_status", ""),
                "evolution_status": node.state.get("evolution_status", ""),
                "interaction_status": node.state.get("interaction_status", ""),
                "state_details": dict(node.state.get("state_details", {}) or {}),
            })
        return elements

    def _temporary_elements_for_support(self, graph: WorldGraph) -> list[dict]:
        elements = []
        for node in graph.nodes.values():
            if node.node_type != "temporary_element":
                continue
            if node.state.get("visible") is False:
                continue
            elements.append({
                "temporary_element_id": node.node_id,
                "name": node.name,
                **dict(node.state),
            })
        return elements

    def _select_relevant_nodes(self, graph: WorldGraph, support_dict: dict) -> list[dict]:
        ids = {self.agent_node_id}
        for item in support_dict.get("permanent_targets", []) or []:
            if isinstance(item, dict) and item.get("element_id"):
                ids.add(str(item["element_id"]))
        for item in support_dict.get("temporary_targets", []) or []:
            if isinstance(item, dict) and item.get("temporary_element_id"):
                ids.add(str(item["temporary_element_id"]))
        for item in support_dict.get("temporary_element_creations", []) or []:
            if not isinstance(item, dict):
                continue
            for key in ("temporary_element_id", "anchor_element_id"):
                if item.get(key):
                    ids.add(str(item[key]))
        for item in support_dict.get("temporary_element_updates", []) or []:
            if not isinstance(item, dict):
                continue
            for key in ("temporary_element_id", "anchor_element_id"):
                if item.get(key):
                    ids.add(str(item[key]))
        for edge in graph.edges_for_node(self.agent_node_id, edge_kind="fact"):
            ids.add(edge.from_node_id)
            ids.add(edge.to_node_id)
        return [
            graph.nodes[node_id].to_dict()
            for node_id in sorted(ids)
            if node_id in graph.nodes
        ]

    def _element_recent_history_for_nodes(
        self,
        graph: WorldGraph,
        relevant_nodes: list[dict],
    ) -> dict:
        history: dict[str, list[dict]] = {}
        for item in relevant_nodes:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id", "") or "").strip()
            node = graph.nodes.get(node_id)
            if node is None or node.node_type not in {"permanent_element", "temporary_element"}:
                continue
            recent = graph.recent_node_history(node_id, limit=6)
            if recent:
                history[node_id] = recent
        return history

    def _merge_reports(
        self,
        first: WorldGraphTransitionReport,
        second: WorldGraphTransitionReport,
    ) -> WorldGraphTransitionReport:
        return WorldGraphTransitionReport(
            created_temporary_nodes=[
                *first.created_temporary_nodes,
                *second.created_temporary_nodes,
            ],
            updated_temporary_nodes=[
                *first.updated_temporary_nodes,
                *second.updated_temporary_nodes,
            ],
            removed_fact_edges=[
                *first.removed_fact_edges,
                *second.removed_fact_edges,
            ],
            added_fact_edges=[
                *first.added_fact_edges,
                *second.added_fact_edges,
            ],
            updated_agent_node=second.updated_agent_node or first.updated_agent_node,
            updated_element_nodes=[
                *first.updated_element_nodes,
                *second.updated_element_nodes,
            ],
            world_action_event=second.world_action_event or first.world_action_event,
            transition_check=second.transition_check or first.transition_check,
            warnings=[*first.warnings, *second.warnings],
        )

    def _sync_updated_permanent_elements_to_scene(
        self,
        report: WorldGraphTransitionReport,
    ) -> None:
        for item in report.updated_element_nodes:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id", "") or "").strip()
            node = self.graph.nodes.get(node_id) if self.graph is not None else None
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

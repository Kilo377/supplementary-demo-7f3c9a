from __future__ import annotations

from dataclasses import dataclass, field

from contextual_world.world_old.temporary_elements import TemporaryElement

from .world_graph import GraphEdge, GraphNode, WorldGraph


ENDED_TEMPORARY_STATUSES = {"consumed", "disposed", "discarded"}


@dataclass
class WorldGraphTransitionReport:
    created_temporary_nodes: list[dict] = field(default_factory=list)
    updated_temporary_nodes: list[dict] = field(default_factory=list)
    removed_fact_edges: list[dict] = field(default_factory=list)
    added_fact_edges: list[dict] = field(default_factory=list)
    updated_agent_node: dict | None = None
    updated_element_nodes: list[dict] = field(default_factory=list)
    world_action_event: dict = field(default_factory=dict)
    transition_check: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "created_temporary_nodes": self.created_temporary_nodes,
            "updated_temporary_nodes": self.updated_temporary_nodes,
            "removed_fact_edges": self.removed_fact_edges,
            "added_fact_edges": self.added_fact_edges,
            "updated_agent_node": self.updated_agent_node,
            "updated_element_nodes": self.updated_element_nodes,
            "world_action_event": dict(self.world_action_event),
            "transition_check": dict(self.transition_check),
            "warnings": list(self.warnings),
        }


class WorldGraphTransitionApplier:
    def __init__(self, graph: WorldGraph, *, agent_node_id: str = "agent_01") -> None:
        self.graph = graph
        self.agent_node_id = agent_node_id

    def apply(
        self,
        *,
        action_proposal: str,
        element_support_result: dict,
        relation_update_result: dict,
    ) -> WorldGraphTransitionReport:
        report = WorldGraphTransitionReport()
        materialized_support = self.materialize_element_support(
            action_proposal=action_proposal,
            element_support_result=element_support_result,
            report=report,
        )
        self.apply_relation_update(
            relation_update_result=relation_update_result,
            report=report,
        )
        element_support_result.clear()
        element_support_result.update(materialized_support)
        return report

    def materialize_element_support(
        self,
        *,
        action_proposal: str,
        element_support_result: dict,
        report: WorldGraphTransitionReport | None = None,
    ) -> dict:
        result = _deep_copy_dict(element_support_result)
        report = report or WorldGraphTransitionReport()
        self._materialize_temporary_creations(
            action_proposal=action_proposal,
            element_support_result=result,
            report=report,
        )
        self._apply_temporary_updates(
            element_support_result=result,
            report=report,
        )
        return result

    def apply_relation_update(
        self,
        *,
        relation_update_result: dict,
        report: WorldGraphTransitionReport | None = None,
    ) -> WorldGraphTransitionReport:
        report = report or WorldGraphTransitionReport()
        self._remove_fact_edges(
            relation_update_result.get("fact_edges_to_remove", []) or [],
            report=report,
        )
        self._add_fact_edges(
            relation_update_result.get("fact_edges_to_add", []) or [],
            report=report,
        )
        self._apply_agent_state_patch(
            relation_update_result.get("agent_state_patch", {}) or {},
            report=report,
        )
        return report

    def apply_element_state_update(
        self,
        *,
        element_state_update_result: dict,
        action_proposal: str = "",
        world_action_event: dict | None = None,
        report: WorldGraphTransitionReport | None = None,
    ) -> WorldGraphTransitionReport:
        report = report or WorldGraphTransitionReport()
        self._apply_element_state_updates(
            element_state_update_result.get("element_state_updates", []) or [],
            action_proposal=action_proposal,
            world_action_event=world_action_event or {},
            report=report,
        )
        return report

    def _materialize_temporary_creations(
        self,
        *,
        action_proposal: str,
        element_support_result: dict,
        report: WorldGraphTransitionReport,
    ) -> None:
        creations = element_support_result.get("temporary_element_creations", []) or []
        for creation in creations:
            if not isinstance(creation, dict):
                continue
            temp_id = str(creation.get("temporary_element_id", "") or "").strip()
            if not temp_id:
                temp_id = self._next_temporary_id()
                creation["temporary_element_id"] = temp_id
            if temp_id in self.graph.nodes:
                report.warnings.append(f"temporary node already exists: {temp_id}")
                continue
            name = str(creation.get("name", "") or temp_id).strip()
            anchor_element_id = str(creation.get("anchor_element_id", "") or "").strip()
            anchor_area_id = str(creation.get("anchor_area_id", "") or "").strip()
            if not anchor_area_id and anchor_element_id in self.graph.nodes:
                anchor_area_id = str(
                    self.graph.nodes[anchor_element_id].state.get("area_id", "") or ""
                )
            status = str(
                creation.get("initial_status", "")
                or creation.get("status", "")
                or "placed"
            ).strip()
            temporary = TemporaryElement(
                temporary_element_id=temp_id,
                name=name,
                anchor_element_id=anchor_element_id,
                anchor_area_id=anchor_area_id,
                lifecycle=str(creation.get("lifecycle", "") or "game").strip(),
                status=status,
                created_by_action=action_proposal,
            )
            self.graph.add_temporary_element(
                temporary,
                holder_actor_id="",
            )
            node = self.graph.nodes[temp_id]
            node.state["visible"] = status not in ENDED_TEMPORARY_STATUSES
            report.created_temporary_nodes.append(node.to_dict())

    def _apply_temporary_updates(
        self,
        *,
        element_support_result: dict,
        report: WorldGraphTransitionReport,
    ) -> None:
        updates = element_support_result.get("temporary_element_updates", []) or []
        for update in updates:
            if not isinstance(update, dict):
                continue
            temp_id = str(update.get("temporary_element_id", "") or "").strip()
            if not temp_id or temp_id not in self.graph.nodes:
                report.warnings.append(f"missing temporary node for update: {temp_id}")
                continue
            node = self.graph.nodes[temp_id]
            if node.node_type != "temporary_element":
                report.warnings.append(f"node is not temporary_element: {temp_id}")
                continue
            old_state = dict(node.state)
            new_status = str(
                update.get("new_status", "")
                or update.get("status", "")
                or ""
            ).strip()
            new_anchor_element_id = str(update.get("anchor_element_id", "") or "").strip()
            new_anchor_area_id = str(update.get("anchor_area_id", "") or "").strip()
            if new_status:
                node.state["status"] = new_status
            if new_anchor_element_id:
                node.state["anchor_element_id"] = new_anchor_element_id
                self.graph.remove_edges(
                    from_node_id=temp_id,
                    relation="anchored_to",
                    edge_kind="affiliation",
                )
                self.graph.add_edge(
                    from_node_id=temp_id,
                    relation="anchored_to",
                    to_node_id=new_anchor_element_id,
                    edge_kind="affiliation",
                )
            if new_anchor_area_id:
                node.state["anchor_area_id"] = new_anchor_area_id
                self.graph.remove_edges(
                    from_node_id=temp_id,
                    relation="located_in",
                    edge_kind="affiliation",
                )
                self.graph.add_edge(
                    from_node_id=temp_id,
                    relation="located_in",
                    to_node_id=new_anchor_area_id,
                    edge_kind="affiliation",
                )
            if node.state.get("status") in ENDED_TEMPORARY_STATUSES:
                node.state["visible"] = False
                removed = self.graph.remove_edges(to_node_id=temp_id, edge_kind="fact")
                removed += self.graph.remove_edges(from_node_id=temp_id, edge_kind="fact")
                report.removed_fact_edges.extend(edge.to_dict() for edge in removed)
            else:
                node.state.setdefault("visible", True)
            if old_state != node.state:
                report.updated_temporary_nodes.append(
                    {
                        "node_id": temp_id,
                        "old_state": old_state,
                        "new_state": dict(node.state),
                    }
                )

    def _remove_fact_edges(
        self,
        edges: list[dict],
        *,
        report: WorldGraphTransitionReport,
    ) -> None:
        for item in edges:
            if not isinstance(item, dict):
                continue
            subject_id = str(item.get("subject_id", "") or "").strip()
            relation = str(item.get("relation", "") or "").strip()
            object_id = str(item.get("object_id", "") or "").strip()
            removed = self.graph.remove_edges(
                from_node_id=subject_id,
                relation=relation,
                to_node_id=object_id,
                edge_kind="fact",
            )
            if not removed:
                if self._is_ended_temporary_node(subject_id) or self._is_ended_temporary_node(object_id):
                    continue
                report.warnings.append(
                    f"fact edge not found for removal: {subject_id} --{relation}--> {object_id}"
                )
            report.removed_fact_edges.extend(edge.to_dict() for edge in removed)

    def _is_ended_temporary_node(self, node_id: str) -> bool:
        node = self.graph.nodes.get(node_id)
        if node is None or node.node_type != "temporary_element":
            return False
        return node.state.get("status") in ENDED_TEMPORARY_STATUSES

    def _add_fact_edges(
        self,
        edges: list[dict],
        *,
        report: WorldGraphTransitionReport,
    ) -> None:
        for item in edges:
            if not isinstance(item, dict):
                continue
            subject_id = str(item.get("subject_id", "") or "").strip()
            relation = str(item.get("relation", "") or "").strip()
            object_id = str(item.get("object_id", "") or "").strip()
            if not subject_id or not relation or not object_id:
                report.warnings.append(f"incomplete fact edge add: {item}")
                continue
            if subject_id not in self.graph.nodes or object_id not in self.graph.nodes:
                report.warnings.append(
                    f"unknown node in fact edge add: {subject_id} --{relation}--> {object_id}"
                )
                continue
            edge_id = self.graph._stable_edge_id(
                from_node_id=subject_id,
                relation=relation,
                to_node_id=object_id,
                edge_kind="fact",
            )
            if edge_id in self.graph.edges:
                continue
            edge = self.graph.add_fact_edge(
                from_node_id=subject_id,
                relation=relation,
                to_node_id=object_id,
                state={"reason": str(item.get("reason", "") or "")},
            )
            report.added_fact_edges.append(edge.to_dict())

    def _apply_agent_state_patch(
        self,
        patch: dict,
        *,
        report: WorldGraphTransitionReport,
    ) -> None:
        if not isinstance(patch, dict) or not patch:
            return
        allowed = {
            "posture",
            "interaction_elements",
            "interaction_method",
            "gaze_target",
            "worn_items",
            "body_surface",
            "text_to_motion_description",
        }
        updates = {key: value for key, value in patch.items() if key in allowed}
        if not updates:
            return
        node = self.graph.update_node_state(self.agent_node_id, updates)
        if node is None:
            report.warnings.append(f"missing agent node: {self.agent_node_id}")
            return
        report.updated_agent_node = node.to_dict()

    def _apply_element_state_updates(
        self,
        updates: list[dict],
        *,
        action_proposal: str = "",
        world_action_event: dict | None = None,
        report: WorldGraphTransitionReport,
    ) -> None:
        allowed = {
            "physical_status",
            "evolution_status",
            "interaction_status",
        }
        for item in updates:
            if not isinstance(item, dict):
                continue
            element_id = str(item.get("element_id", "") or "").strip()
            if not element_id:
                report.warnings.append(f"missing element_id in element state update: {item}")
                continue
            node = self.graph.nodes.get(element_id)
            if node is None:
                report.warnings.append(f"missing element node for state update: {element_id}")
                continue
            if node.node_type not in {"permanent_element", "temporary_element"}:
                report.warnings.append(f"node is not an element for state update: {element_id}")
                continue

            old_state = _deep_copy_dict(node.state)
            state_updates = {}
            for key in allowed:
                value = item.get(key)
                if value is None:
                    continue
                cleaned = str(value).strip()
                if cleaned:
                    state_updates[key] = cleaned

            details = item.get("state_details")
            if isinstance(details, dict):
                current_details = dict(node.state.get("state_details", {}) or {})
                for key, value in details.items():
                    cleaned_key = str(key).strip()
                    cleaned_value = str(value).strip()
                    if cleaned_key and cleaned_value:
                        current_details[cleaned_key] = cleaned_value
                if current_details:
                    state_updates["state_details"] = current_details

            if not state_updates:
                continue
            self.graph.update_node_state(element_id, state_updates)
            if old_state == node.state:
                continue
            self.graph.append_node_history(
                element_id,
                {
                    "action_proposal": action_proposal,
                    "actual_event": str((world_action_event or {}).get("actual_event", "") or ""),
                    "old_state": old_state,
                    "new_state": _deep_copy_dict(node.state),
                },
            )
            report.updated_element_nodes.append(
                {
                    "node_id": element_id,
                    "old_state": old_state,
                    "new_state": _deep_copy_dict(node.state),
                }
            )

    def _next_temporary_id(self) -> str:
        existing_indexes = []
        for node_id in self.graph.nodes:
            if not node_id.startswith("temporary_generated_"):
                continue
            suffix = node_id.rsplit("_", 1)[-1]
            if suffix.isdigit():
                existing_indexes.append(int(suffix))
        next_index = max(existing_indexes or [0]) + 1
        while True:
            candidate = f"temporary_generated_{next_index:03d}"
            if candidate not in self.graph.nodes:
                return candidate
            next_index += 1


def _deep_copy_dict(value: dict) -> dict:
    result: dict = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[key] = _deep_copy_dict(item)
        elif isinstance(item, list):
            result[key] = [_deep_copy_dict(child) if isinstance(child, dict) else child for child in item]
        else:
            result[key] = item
    return result

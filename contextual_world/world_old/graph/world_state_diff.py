from __future__ import annotations

from dataclasses import dataclass, field

from .world_graph import WorldGraph


AGENT_STATE_FIELDS = [
    "posture",
    "interaction_elements",
    "interaction_method",
    "gaze_target",
    "worn_items",
    "body_surface",
    "text_to_motion_description",
]

ENDED_TEMPORARY_STATUSES = {"consumed", "disposed", "discarded"}


@dataclass
class StateFieldChange:
    field: str
    before: object
    after: object

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "before": self.before,
            "after": self.after,
        }


@dataclass
class RelationChange:
    change_type: str
    relation: str
    other_node_id: str
    other_name: str

    def to_dict(self) -> dict:
        return {
            "change_type": self.change_type,
            "relation": self.relation,
            "other_name": self.other_name,
        }


@dataclass
class ElementSnapshot:
    node_id: str
    name: str
    node_type: str
    state: dict = field(default_factory=dict)
    relation_to_self: str = ""

    def to_dict(self) -> dict:
        result = {"name": self.name}
        if self.relation_to_self:
            result["relation_to_self"] = self.relation_to_self
        status = self.state.get("status") or self.state.get("interaction_status")
        if status:
            result["status"] = status
        lifecycle = self.state.get("lifecycle")
        if lifecycle:
            result["lifecycle"] = lifecycle
        details = _agent_visible_state_details(self.state)
        if details:
            result["details"] = details
        return result


@dataclass
class ElementChange:
    node_id: str
    name: str
    node_type: str
    state_changes: list[StateFieldChange] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state_changes": [change.to_dict() for change in self.state_changes],
        }


@dataclass
class AgentCenteredWorldStateDiff:
    self_state_before: dict = field(default_factory=dict)
    self_state_after: dict = field(default_factory=dict)
    self_state_changes: list[StateFieldChange] = field(default_factory=list)
    self_relation_changes: list[RelationChange] = field(default_factory=list)
    acquired_or_noticed_elements: list[ElementSnapshot] = field(default_factory=list)
    consumed_or_disposed_elements: list[ElementSnapshot] = field(default_factory=list)
    affected_external_elements: list[ElementChange] = field(default_factory=list)
    current_relevant_elements: list[ElementSnapshot] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "self_state_before": dict(self.self_state_before),
            "self_state_after": dict(self.self_state_after),
            "self_state_changes": [change.to_dict() for change in self.self_state_changes],
            "self_relation_changes": [change.to_dict() for change in self.self_relation_changes],
            "acquired_or_noticed_elements": [
                element.to_dict() for element in self.acquired_or_noticed_elements
            ],
            "consumed_or_disposed_elements": [
                element.to_dict() for element in self.consumed_or_disposed_elements
            ],
            "affected_external_elements": [
                element.to_dict() for element in self.affected_external_elements
            ],
            "current_relevant_elements": [
                element.to_dict() for element in self.current_relevant_elements
            ],
            "warnings": list(self.warnings),
        }


def build_agent_centered_world_state_diff(
    *,
    before_graph: WorldGraph,
    after_graph: WorldGraph,
    transition_report: dict,
    agent_node_id: str = "agent_01",
) -> AgentCenteredWorldStateDiff:
    before_self = _agent_state(before_graph, agent_node_id)
    after_self = _agent_state(after_graph, agent_node_id)
    self_relation_changes = _self_relation_changes(
        before_graph=before_graph,
        after_graph=after_graph,
        transition_report=transition_report,
        agent_node_id=agent_node_id,
    )
    return AgentCenteredWorldStateDiff(
        self_state_before=before_self,
        self_state_after=after_self,
        self_state_changes=_state_changes(before_self, after_self),
        self_relation_changes=self_relation_changes,
        acquired_or_noticed_elements=_acquired_or_noticed_elements(
            after_graph=after_graph,
            relation_changes=self_relation_changes,
            transition_report=transition_report,
        ),
        consumed_or_disposed_elements=_consumed_or_disposed_elements(
            before_graph=before_graph,
            after_graph=after_graph,
            transition_report=transition_report,
        ),
        affected_external_elements=_affected_external_elements(
            before_graph=before_graph,
            after_graph=after_graph,
            transition_report=transition_report,
            agent_node_id=agent_node_id,
        ),
        current_relevant_elements=_current_relevant_elements(
            after_graph=after_graph,
            transition_report=transition_report,
            agent_node_id=agent_node_id,
        ),
        warnings=list(transition_report.get("warnings", []) or []),
    )


def _agent_state(graph: WorldGraph, agent_node_id: str) -> dict:
    node = graph.nodes.get(agent_node_id)
    state = node.state if node is not None else {}
    return {
        field: _agent_visible_agent_state_value(field, state.get(field), graph)
        for field in AGENT_STATE_FIELDS
    }


def _state_changes(before: dict, after: dict) -> list[StateFieldChange]:
    changes = []
    for field in AGENT_STATE_FIELDS:
        before_value = before.get(field)
        after_value = after.get(field)
        if before_value != after_value:
            changes.append(StateFieldChange(field=field, before=before_value, after=after_value))
    return changes


def _self_relation_changes(
    *,
    before_graph: WorldGraph,
    after_graph: WorldGraph,
    transition_report: dict,
    agent_node_id: str,
) -> list[RelationChange]:
    changes = []
    seen = set()
    for edge in transition_report.get("added_fact_edges", []) or []:
        relation_change = _relation_change_from_edge(
            edge,
            graph=after_graph,
            agent_node_id=agent_node_id,
            change_type="added",
        )
        if relation_change is None:
            continue
        key = ("added", relation_change.relation, relation_change.other_node_id)
        if key not in seen:
            seen.add(key)
            changes.append(relation_change)
    for edge in transition_report.get("removed_fact_edges", []) or []:
        relation_change = _relation_change_from_edge(
            edge,
            graph=before_graph,
            agent_node_id=agent_node_id,
            change_type="removed",
        )
        if relation_change is None:
            continue
        key = ("removed", relation_change.relation, relation_change.other_node_id)
        if key not in seen:
            seen.add(key)
            changes.append(relation_change)
    return changes


def _relation_change_from_edge(
    edge: dict,
    *,
    graph: WorldGraph,
    agent_node_id: str,
    change_type: str,
) -> RelationChange | None:
    from_id = str(edge.get("from_node_id", "") or edge.get("subject_id", "") or "")
    to_id = str(edge.get("to_node_id", "") or edge.get("object_id", "") or "")
    relation = str(edge.get("relation", "") or "")
    if from_id == agent_node_id:
        other_id = to_id
    elif to_id == agent_node_id:
        other_id = from_id
    else:
        return None
    other = graph.nodes.get(other_id)
    return RelationChange(
        change_type=change_type,
        relation=relation,
        other_node_id=other_id,
        other_name=other.name if other is not None else other_id,
    )


def _acquired_or_noticed_elements(
    *,
    after_graph: WorldGraph,
    relation_changes: list[RelationChange],
    transition_report: dict,
) -> list[ElementSnapshot]:
    relation_priority = {"holding", "looking_at"}
    node_relation: dict[str, str] = {}
    for change in relation_changes:
        if change.change_type != "added":
            continue
        if change.relation not in relation_priority:
            continue
        node_relation.setdefault(change.other_node_id, change.relation)

    created_ids = {
        str(node.get("node_id", "") or "")
        for node in transition_report.get("created_temporary_nodes", []) or []
        if isinstance(node, dict)
    }
    result = []
    seen = set()
    for node_id in [*node_relation.keys(), *created_ids]:
        if not node_id or node_id in seen:
            continue
        node = after_graph.nodes.get(node_id)
        if node is None or node.node_type not in {"temporary_element", "permanent_element"}:
            continue
        if node.node_type == "temporary_element" and node.state.get("visible") is False:
            continue
        seen.add(node_id)
        result.append(_element_snapshot(node, relation_to_self=node_relation.get(node_id, "")))
    return result


def _consumed_or_disposed_elements(
    *,
    before_graph: WorldGraph,
    after_graph: WorldGraph,
    transition_report: dict,
) -> list[ElementSnapshot]:
    result = []
    seen = set()
    for item in transition_report.get("updated_temporary_nodes", []) or []:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id", "") or "")
        new_state = item.get("new_state", {}) or {}
        if new_state.get("status") not in ENDED_TEMPORARY_STATUSES and new_state.get("visible") is not False:
            continue
        node = after_graph.nodes.get(node_id) or before_graph.nodes.get(node_id)
        if node is None or node_id in seen:
            continue
        seen.add(node_id)
        result.append(_element_snapshot(node))
    return result


def _affected_external_elements(
    *,
    before_graph: WorldGraph,
    after_graph: WorldGraph,
    transition_report: dict,
    agent_node_id: str,
) -> list[ElementChange]:
    affected_ids = set()
    created_temporary_ids = {
        str(item.get("node_id", "") or "")
        for item in transition_report.get("created_temporary_nodes", []) or []
        if isinstance(item, dict)
    }
    ended_temporary_ids = set()
    for item in transition_report.get("updated_temporary_nodes", []) or []:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id", "") or "")
        new_state = item.get("new_state", {}) or {}
        if new_state.get("status") in ENDED_TEMPORARY_STATUSES or new_state.get("visible") is False:
            ended_temporary_ids.add(node_id)

    for key in ["created_temporary_nodes", "updated_temporary_nodes", "updated_element_nodes"]:
        for item in transition_report.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            state = item.get("state") or item.get("new_state") or {}
            for ref_key in ["anchor_element_id", "anchor_area_id"]:
                ref_id = str(state.get(ref_key, "") or "")
                if ref_id:
                    affected_ids.add(ref_id)
    for key in ["added_fact_edges", "removed_fact_edges"]:
        for edge in transition_report.get(key, []) or []:
            if not isinstance(edge, dict):
                continue
            for node_key in ["from_node_id", "to_node_id", "subject_id", "object_id"]:
                node_id = str(edge.get(node_key, "") or "")
                if node_id and node_id != agent_node_id:
                    affected_ids.add(node_id)
    for node_id, after_node in after_graph.nodes.items():
        if node_id == agent_node_id:
            continue
        before_node = before_graph.nodes.get(node_id)
        if before_node is None:
            affected_ids.add(node_id)
            continue
        if before_node.state != after_node.state:
            affected_ids.add(node_id)

    result = []
    for node_id in sorted(affected_ids):
        if node_id in created_temporary_ids or node_id in ended_temporary_ids:
            continue
        after_node = after_graph.nodes.get(node_id)
        before_node = before_graph.nodes.get(node_id)
        node = after_node or before_node
        if node is None or node.node_type not in {"permanent_element", "temporary_element"}:
            continue
        state_changes = _dict_state_changes(
            before_node.state if before_node is not None else {},
            after_node.state if after_node is not None else {},
        )
        if not state_changes and node.node_type == "temporary_element":
            continue
        if not state_changes:
            continue
        result.append(
            ElementChange(
                node_id=node.node_id,
                name=node.name,
                node_type=node.node_type,
                state_changes=state_changes,
            )
        )
    return result


def _current_relevant_elements(
    *,
    after_graph: WorldGraph,
    transition_report: dict,
    agent_node_id: str,
) -> list[ElementSnapshot]:
    relevant_ids = set()
    for edge in after_graph.edges_for_node(agent_node_id, edge_kind="fact"):
        if edge.from_node_id == agent_node_id:
            relevant_ids.add(edge.to_node_id)
        elif edge.to_node_id == agent_node_id:
            relevant_ids.add(edge.from_node_id)
    for key in ["created_temporary_nodes", "updated_temporary_nodes", "updated_element_nodes"]:
        for item in transition_report.get(key, []) or []:
            if isinstance(item, dict) and item.get("node_id"):
                relevant_ids.add(str(item["node_id"]))
    result = []
    for node_id in sorted(relevant_ids):
        node = after_graph.nodes.get(node_id)
        if node is None or node.node_type not in {"temporary_element", "permanent_element"}:
            continue
        if node.node_type == "temporary_element" and node.state.get("visible") is False:
            continue
        result.append(_element_snapshot(node, relation_to_self=_relation_to_agent(after_graph, agent_node_id, node_id)))
    return result


def _relation_to_agent(graph: WorldGraph, agent_node_id: str, node_id: str) -> str:
    for edge in graph.edges_for_node(agent_node_id, edge_kind="fact"):
        if edge.from_node_id == agent_node_id and edge.to_node_id == node_id:
            return edge.relation
        if edge.to_node_id == agent_node_id and edge.from_node_id == node_id:
            return edge.relation
    return ""


def _element_snapshot(node, *, relation_to_self: str = "") -> ElementSnapshot:
    return ElementSnapshot(
        node_id=node.node_id,
        name=node.name,
        node_type=node.node_type,
        state=dict(node.state),
        relation_to_self=relation_to_self,
    )


def _dict_state_changes(before: dict, after: dict) -> list[StateFieldChange]:
    changes = []
    skipped_fields = {
        "area_id",
        "anchor_area_id",
        "anchor_element_id",
        "center",
        "size",
        "movable",
        "blocks_movement",
        "created_by_action",
        "visible",
    }
    for key in sorted(set(before) | set(after)):
        if key in skipped_fields:
            continue
        if key == "state_details":
            before_details = before.get(key) if isinstance(before.get(key), dict) else {}
            after_details = after.get(key) if isinstance(after.get(key), dict) else {}
            for detail_key in sorted(set(before_details) | set(after_details)):
                before_value = _copy_value(before_details.get(detail_key))
                after_value = _copy_value(after_details.get(detail_key))
                if before_value != after_value:
                    changes.append(
                        StateFieldChange(
                            field=f"details.{detail_key}",
                            before=before_value,
                            after=after_value,
                        )
                    )
            continue
        before_value = _copy_value(before.get(key))
        after_value = _copy_value(after.get(key))
        if before_value != after_value:
            changes.append(StateFieldChange(field=key, before=before_value, after=after_value))
    return changes


def _agent_visible_agent_state_value(field: str, value, graph: WorldGraph):
    if field == "interaction_elements":
        return _agent_visible_interaction_elements(value, graph)
    if field == "gaze_target":
        if not value:
            return ""
        node = graph.nodes.get(str(value))
        return node.name if node is not None else value
    return _copy_value(value)


def _agent_visible_interaction_elements(value, graph: WorldGraph) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, dict):
            name = str(item.get("name", "") or "")
            node_id = str(item.get("id", "") or item.get("node_id", "") or "")
            if not name and node_id in graph.nodes:
                name = graph.nodes[node_id].name
            if name:
                result.append(name)
            continue
        node_id = str(item or "")
        if node_id in graph.nodes:
            result.append(graph.nodes[node_id].name)
        elif node_id:
            result.append(node_id)
    return result


def _agent_visible_state_details(state: dict) -> dict:
    details = state.get("state_details")
    if not isinstance(details, dict):
        return {}
    return {
        key: value
        for key, value in details.items()
        if value not in {None, "", "unset"}
    }


def _copy_value(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value

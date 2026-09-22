from __future__ import annotations

from contextual_world.world_old.graph.world_graph import WorldGraph
from contextual_world.world_old.graph.world_graph_transition import ENDED_TEMPORARY_STATUSES, WorldGraphTransitionReport
from contextual_world.world_old.temporary_elements import TemporaryElement


def materialize_node_support(
    *,
    graph: WorldGraph,
    action_proposal: str,
    support_result: dict,
    report: WorldGraphTransitionReport | None = None,
) -> WorldGraphTransitionReport:
    report = report or WorldGraphTransitionReport()
    for creation in support_result.get("temporary_node_creations", []) or []:
        if not isinstance(creation, dict):
            continue
        temp_id = str(creation.get("temporary_element_id", "") or "").strip()
        if not temp_id:
            temp_id = _next_temporary_id(graph)
            creation["temporary_element_id"] = temp_id
        if temp_id in graph.nodes:
            report.warnings.append(f"temporary node already exists: {temp_id}")
            continue
        anchor_element_id = str(creation.get("anchor_element_id", "") or "").strip()
        anchor_area_id = str(creation.get("anchor_area_id", "") or "").strip()
        if not anchor_area_id and anchor_element_id in graph.nodes:
            anchor_area_id = str(graph.nodes[anchor_element_id].state.get("area_id", "") or "")
        status = _initial_temporary_status(creation.get("initial_status"))
        temporary = TemporaryElement(
            temporary_element_id=temp_id,
            name=str(creation.get("name", "") or temp_id),
            anchor_element_id=anchor_element_id,
            anchor_area_id=anchor_area_id,
            lifecycle=str(creation.get("lifecycle", "") or "game"),
            status=status,
            created_by_action=action_proposal,
        )
        graph.add_temporary_element(temporary)
        node = graph.nodes[temp_id]
        node.state["visible"] = status not in ENDED_TEMPORARY_STATUSES
        report.created_temporary_nodes.append(node.to_dict())
    return report


def apply_state_transition_delta(
    *,
    graph: WorldGraph,
    transition_result: dict,
    report: WorldGraphTransitionReport | None = None,
) -> WorldGraphTransitionReport:
    report = report or WorldGraphTransitionReport()
    delta = transition_result.get("next_state_delta", {}) or {}
    for item in delta.get("fact_edges_to_remove", []) or []:
        subject_id = str(item.get("subject_id", "") or "").strip()
        relation = str(item.get("relation", "") or "").strip()
        object_id = str(item.get("object_id", "") or "").strip()
        removed = graph.remove_edges(
            from_node_id=subject_id,
            relation=relation,
            to_node_id=object_id,
            edge_kind="fact",
        )
        if not removed:
            report.warnings.append(f"fact edge not found for removal: {subject_id} --{relation}--> {object_id}")
        report.removed_fact_edges.extend(edge.to_dict() for edge in removed)
    for item in delta.get("fact_edges_to_add", []) or []:
        subject_id = str(item.get("subject_id", "") or "").strip()
        relation = str(item.get("relation", "") or "").strip()
        object_id = str(item.get("object_id", "") or "").strip()
        if not subject_id or not relation or not object_id:
            report.warnings.append(f"incomplete fact edge add: {item}")
            continue
        if subject_id not in graph.nodes or object_id not in graph.nodes:
            report.warnings.append(f"unknown node in fact edge add: {subject_id} --{relation}--> {object_id}")
            continue
        edge_id = graph._stable_edge_id(
            from_node_id=subject_id,
            relation=relation,
            to_node_id=object_id,
            edge_kind="fact",
        )
        if edge_id in graph.edges:
            continue
        edge = graph.add_fact_edge(
            from_node_id=subject_id,
            relation=relation,
            to_node_id=object_id,
            state={"reason": str(item.get("reason", "") or "")},
        )
        report.added_fact_edges.append(edge.to_dict())
    for item in delta.get("node_updates", []) or []:
        _apply_node_update(graph=graph, item=item, report=report)
    report.world_action_event = {
        "accepted": bool(transition_result.get("accepted", False)),
        **dict(transition_result.get("execution_result", {}) or {}),
    }
    return report


def _apply_node_update(*, graph: WorldGraph, item: dict, report: WorldGraphTransitionReport) -> None:
    node_id = str(item.get("node_id", "") or "").strip()
    patch = item.get("state_patch", {}) if isinstance(item, dict) else {}
    if node_id not in graph.nodes or not isinstance(patch, dict):
        report.warnings.append(f"invalid node update: {item}")
        return
    node = graph.nodes[node_id]
    old_state = _deep_copy(node.state)
    merged = _merge_state_patch(node.state, patch)
    node.state.clear()
    node.state.update(merged)
    if old_state == node.state:
        return
    if node.node_type == "human_agent":
        report.updated_agent_node = node.to_dict()
    elif node.node_type == "temporary_element":
        if node.state.get("status") in ENDED_TEMPORARY_STATUSES:
            node.state["visible"] = False
            removed = graph.remove_edges(to_node_id=node_id, edge_kind="fact")
            removed += graph.remove_edges(from_node_id=node_id, edge_kind="fact")
            report.removed_fact_edges.extend(edge.to_dict() for edge in removed)
        report.updated_temporary_nodes.append({
            "node_id": node_id,
            "old_state": old_state,
            "new_state": _deep_copy(node.state),
        })
    else:
        graph.append_node_history(
            node_id,
            {
                "old_state": old_state,
                "new_state": _deep_copy(node.state),
            },
        )
        report.updated_element_nodes.append({
            "node_id": node_id,
            "old_state": old_state,
            "new_state": _deep_copy(node.state),
        })


def _merge_state_patch(current: dict, patch: dict) -> dict:
    result = _deep_copy(current)
    for key, value in patch.items():
        if key == "state_details" and isinstance(value, dict):
            details = dict(result.get("state_details", {}) or {})
            details.update(value)
            result["state_details"] = details
        else:
            result[key] = value
    return result


def _next_temporary_id(graph: WorldGraph) -> str:
    existing = []
    for node_id in graph.nodes:
        if node_id.startswith("temporary_generated_"):
            suffix = node_id.rsplit("_", 1)[-1]
            if suffix.isdigit():
                existing.append(int(suffix))
    index = max(existing or [0]) + 1
    while True:
        candidate = f"temporary_generated_{index:03d}"
        if candidate not in graph.nodes:
            return candidate
        index += 1


def _initial_temporary_status(value) -> str:
    status = str(value or "available").strip()
    if status in ENDED_TEMPORARY_STATUSES or status in {"held", "in_use", "used"}:
        return "available"
    return status or "available"


def _deep_copy(value):
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value

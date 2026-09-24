from __future__ import annotations

from contextual_world.world_old.graph.world_graph import WorldGraph

from contextual_world.modules.interaction_focus.types import WorldInteractionFocusResult

from .types import WorldStateTransitionVariables


def build_world_state_transition_variables(
    *,
    graph: WorldGraph,
    agent_node_id: str,
    action_proposal: str,
    focus_result: WorldInteractionFocusResult,
    support_result: dict | None = None,
    history_limit: int = 5,
) -> WorldStateTransitionVariables:
    focused_ids = _normalize_focused_ids(graph=graph, agent_node_id=agent_node_id, focused_ids=focus_result.focused_node_ids)
    focused_nodes = [graph.nodes[node_id].to_dict() for node_id in focused_ids if node_id in graph.nodes]
    focused_id_set = set(focused_ids)
    focused_fact_edges = [
        edge.to_dict()
        for edge in graph.edges.values()
        if edge.edge_kind == "fact"
        and edge.from_node_id in focused_id_set
        and edge.to_node_id in focused_id_set
    ]
    node_reference = {
        node["node_id"]: {
            "name": node.get("name", ""),
            "node_type": node.get("node_type", ""),
        }
        for node in focused_nodes
        if node.get("node_id")
    }
    return WorldStateTransitionVariables(
        agent_name=graph.nodes[agent_node_id].name if agent_node_id in graph.nodes else agent_node_id,
        action_proposal=action_proposal,
        subgraph_state_text=_build_subgraph_state_text(
            focused_nodes=focused_nodes,
            focused_fact_edges=focused_fact_edges,
            interaction_frame=focus_result.interaction_frame,
            node_reference=node_reference,
        ),
        recent_state_history_text=_build_recent_state_history_text(
            graph=graph,
            focused_ids=focused_ids,
            limit=history_limit,
        ),
        transition_constraints_text=_build_transition_constraints_text(
            focused_ids=focused_ids,
            focused_fact_edges=focused_fact_edges,
        ),
        support_context_facts=[
            str(item).strip()
            for item in ((support_result or {}).get("context_facts", []) or [])
            if str(item).strip()
        ],
        node_reference=node_reference,
        focused_nodes=focused_nodes,
        focused_fact_edges=focused_fact_edges,
        interaction_frame=list(focus_result.interaction_frame),
    )


def _normalize_focused_ids(*, graph: WorldGraph, agent_node_id: str, focused_ids: list[str]) -> list[str]:
    result = []
    for node_id in [agent_node_id, *focused_ids]:
        if node_id in graph.nodes and node_id not in result:
            result.append(node_id)
    return result


def _build_subgraph_state_text(
    *,
    focused_nodes: list[dict],
    focused_fact_edges: list[dict],
    interaction_frame: list[dict],
    node_reference: dict[str, dict],
) -> str:
    lines = ["Current subgraph state:"]
    lines.append("Nodes:")
    for node in focused_nodes:
        node_id = node.get("node_id", "")
        name = node.get("name", "")
        node_type = node.get("node_type", "")
        state = node.get("state", {}) or {}
        lines.append(f"- {node_id} / {name}: {node_type}，state={state}")
    if focused_fact_edges:
        lines.append("Current factual relationships:")
        for edge in focused_fact_edges:
            from_name = _name(edge.get("from_node_id", ""), node_reference)
            to_name = _name(edge.get("to_node_id", ""), node_reference)
            lines.append(f"- {from_name} --{edge.get('relation', '')}--> {to_name}")
    if interaction_frame:
        lines.append("Action focus hints:")
        for frame in interaction_frame:
            subject = _name(frame.get("subject_id", ""), node_reference)
            obj = _name(frame.get("object_id", ""), node_reference)
            lines.append(f"- {subject} --{frame.get('relation_hint', '')}--> {obj}")
    return "\n".join(lines)


def _build_recent_state_history_text(*, graph: WorldGraph, focused_ids: list[str], limit: int) -> str:
    lines = []
    for node_id in focused_ids:
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        history = graph.recent_node_history(node_id, limit=limit)
        if not history:
            continue
        lines.append(f"{node.name} / {node_id}:")
        for item in history:
            action = str(item.get("action_proposal", "") or "")
            event = str(item.get("actual_event", "") or "")
            new_state = item.get("new_state", {})
            lines.append(f"- action={action}; event={event}; new_state={new_state}")
    return "\n".join(lines) if lines else "No local state history has been recorded for the relevant nodes recently."


def _build_transition_constraints_text(*, focused_ids: list[str], focused_fact_edges: list[dict]) -> str:
    lines = [
        f"Only these nodes can be updated: {', '.join(focused_ids)}.",
        "Cannot create new nodes.",
        "The subject_id and object_id in fact_edges_to_add must come from the nodes listed above.",
    ]
    if focused_fact_edges:
        lines.append("fact_edges_to_remove can only remove the following existing factual relationships:")
        for edge in focused_fact_edges:
            lines.append(
                f"- {edge.get('from_node_id', '')} --{edge.get('relation', '')}--> {edge.get('to_node_id', '')}"
            )
    else:
        lines.append("There are no removable existing fact edges in the current focused subgraph.")
    return "\n".join(lines)


def _name(node_id: str, node_reference: dict[str, dict]) -> str:
    item = node_reference.get(node_id, {})
    name = item.get("name", "")
    if name:
        return f"{name}({node_id})"
    return node_id

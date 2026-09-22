from __future__ import annotations

from contextual_world.world_old.graph.world_graph import WorldGraph

from .types import WorldInteractionFocusVariables


def build_world_interaction_focus_variables(
    *,
    graph: WorldGraph,
    agent_node_id: str,
    action_proposal: str,
    support_result: dict,
) -> WorldInteractionFocusVariables:
    agent_node = graph.nodes.get(agent_node_id)
    if agent_node is None:
        raise ValueError(f"Missing human agent node: {agent_node_id}")
    nodes = [
        _node_directory(node.to_dict())
        for node in graph.nodes.values()
        if node.node_type in {"human_agent", "permanent_element", "temporary_element", "area"}
        and not (node.node_type == "temporary_element" and node.state.get("visible") is False)
    ]
    fact_edges = [
        _fact_reference(edge.to_dict())
        for edge in graph.edges.values()
        if edge.edge_kind == "fact"
    ]
    return WorldInteractionFocusVariables(
        agent_name=agent_node.name,
        action_proposal=action_proposal,
        support_result=dict(support_result or {}),
        graph_state_text=_build_graph_state_text(nodes=nodes, fact_edges=fact_edges, agent_node_id=agent_node_id),
        node_reference={
            node["node_id"]: {
                "name": node.get("name", ""),
                "node_type": node.get("node_type", ""),
                "area_id": node.get("area_id", ""),
            }
            for node in nodes
            if node.get("node_id")
        },
        nodes=nodes,
        fact_edges=fact_edges,
    )


def _build_graph_state_text(*, nodes: list[dict], fact_edges: list[dict], agent_node_id: str) -> str:
    names = {node.get("node_id"): node.get("name", "") for node in nodes}
    lines = ["当前可选节点："]
    for node in nodes:
        node_id = node.get("node_id", "")
        node_type = node.get("node_type", "")
        name = node.get("name", "")
        area_id = node.get("area_id", "")
        area_text = f"，area={area_id}" if area_id else ""
        marker = "human_agent" if node_id == agent_node_id else node_type
        lines.append(f"- {node_id} / {name}: {marker}{area_text}。")
    if fact_edges:
        lines.append("当前事实关系：")
        for edge in fact_edges:
            from_name = names.get(edge.get("from_node_id"), edge.get("from_node_id", ""))
            to_name = names.get(edge.get("to_node_id"), edge.get("to_node_id", ""))
            lines.append(f"- {from_name} --{edge.get('relation', '')}--> {to_name}")
    return "\n".join(lines)


def _node_directory(node: dict) -> dict:
    state = node.get("state", {}) or {}
    node_type = node.get("node_type", "")
    if node_type == "human_agent":
        area_id = state.get("current_area_id", "")
    elif node_type == "permanent_element":
        area_id = state.get("area_id", "")
    elif node_type == "temporary_element":
        area_id = state.get("anchor_area_id", "")
    else:
        area_id = ""
    return {
        "node_id": node.get("node_id", ""),
        "node_type": node_type,
        "name": node.get("name", ""),
        "area_id": area_id,
    }


def _fact_reference(edge: dict) -> dict:
    return {
        "from_node_id": edge.get("from_node_id", ""),
        "relation": edge.get("relation", ""),
        "to_node_id": edge.get("to_node_id", ""),
    }

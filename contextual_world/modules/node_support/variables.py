from __future__ import annotations

from contextual_world.world_old.graph.world_graph import WorldGraph

from .types import WorldNodeSupportVariables


def build_world_node_support_variables(
    *,
    graph: WorldGraph,
    agent_node_id: str,
    action_proposal: str,
) -> WorldNodeSupportVariables:
    agent_node = graph.nodes.get(agent_node_id)
    if agent_node is None:
        raise ValueError(f"Missing human agent node: {agent_node_id}")
    return WorldNodeSupportVariables(
        agent_name=agent_node.name,
        action_proposal=action_proposal,
        agent_node=_node_reference(agent_node.to_dict()),
        agent_related_facts=[
            _fact_reference(edge.to_dict())
            for edge in graph.edges_for_node(agent_node_id, edge_kind="fact")
        ],
        permanent_nodes=[
            _node_reference(node.to_dict())
            for node in graph.nodes.values()
            if node.node_type == "permanent_element"
        ],
        temporary_nodes=[
            _node_reference(node.to_dict())
            for node in graph.nodes.values()
            if node.node_type == "temporary_element"
            and node.state.get("visible") is not False
        ],
    )


def _node_reference(node: dict) -> dict:
    state = node.get("state", {}) or {}
    reference = {
        "node_id": node.get("node_id", ""),
        "node_type": node.get("node_type", ""),
        "name": node.get("name", ""),
    }
    semantic_type = str(state.get("semantic_type", "") or "").strip()
    if semantic_type:
        reference["semantic_type"] = semantic_type
    components = sorted({str(value).strip() for value in state.get("components", []) or [] if str(value).strip()})
    if components:
        reference["components"] = components
    return reference


def _fact_reference(edge: dict) -> dict:
    return {
        "from_node_id": edge.get("from_node_id", ""),
        "relation": edge.get("relation", ""),
        "to_node_id": edge.get("to_node_id", ""),
    }

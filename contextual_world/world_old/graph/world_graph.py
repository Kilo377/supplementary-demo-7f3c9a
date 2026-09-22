from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contextual_world.structure.scene_schema import Area, Element, Home
from contextual_world.world_old.temporary_elements import TemporaryElement


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    name: str
    state: dict = field(default_factory=dict)
    recent_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "state": dict(self.state),
        }


@dataclass
class GraphEdge:
    edge_id: str
    from_node_id: str
    relation: str
    to_node_id: str
    edge_kind: str = "fact"
    state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "from_node_id": self.from_node_id,
            "relation": self.relation,
            "to_node_id": self.to_node_id,
            "edge_kind": self.edge_kind,
            "state": dict(self.state),
        }


@dataclass
class WorldGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[str, GraphEdge] = field(default_factory=dict)
    next_edge_index: int = 1

    @classmethod
    def from_scene(
        cls,
        home: Home,
        *,
        actor: Any = None,
        temporary_elements: list[TemporaryElement] | None = None,
    ) -> "WorldGraph":
        graph = cls()
        graph.add_scene(home)
        if actor is not None:
            graph.add_actor(actor, home=home)
            holder_actor_id = getattr(actor, "node_id", "") or getattr(actor, "actor_id", "")
        else:
            holder_actor_id = ""
        for temporary_element in temporary_elements or []:
            graph.add_temporary_element(
                temporary_element,
                holder_actor_id=holder_actor_id,
            )
        return graph

    def add_scene(self, home: Home) -> None:
        self.add_node(
            node_id=home.node_id,
            node_type="scene",
            name=home.name,
            state={
                "default_agent_start": list(home.default_agent_start or []),
            },
        )
        for area in home.areas:
            self.add_area(area, scene_id=home.node_id)

    def add_area(self, area: Area, *, scene_id: str = "") -> None:
        self.add_node(
            node_id=area.node_id,
            node_type="area",
            name=area.name,
            state={
                "bounds": list(area.bounds),
            },
        )
        if scene_id:
            self.add_edge(
                from_node_id=scene_id,
                relation="contains",
                to_node_id=area.node_id,
                edge_kind="structural",
            )
        for element in area.elements:
            self.add_permanent_element(element, area=area)

    def add_permanent_element(self, element: Element, *, area: Area) -> None:
        self.add_node(
            node_id=element.node_id,
            node_type="permanent_element",
            name=element.name,
            state={
                "area_id": area.node_id,
                "semantic_type": element.semantic_type,
                "components": list(element.components),
                "center": list(element.center),
                "size": list(element.size),
                "movable": element.movable,
                "blocks_movement": element.blocks_movement,
                "physical_status": element.physical_status,
                "evolution_status": element.evolution_status,
                "interaction_status": element.interaction_status,
                "state_details": dict(element.state_details),
            },
        )
        self.add_edge(
            from_node_id=area.node_id,
            relation="contains",
            to_node_id=element.node_id,
            edge_kind="structural",
        )

    def add_actor(self, actor: Any, *, home: Home | None = None) -> None:
        self.add_human_agent(actor, home=home)

    def add_human_agent(self, actor: Any, *, home: Home | None = None) -> None:
        actor_id = getattr(actor, "node_id", "") or getattr(actor, "actor_id", "") or "agent_01"
        center = tuple(getattr(actor, "center", ()))
        current_area_id = getattr(actor, "current_area_id", None)
        if not current_area_id and home is not None and len(center) == 2:
            current_area_id = self._area_id_for_point(home, center)
        self.add_node(
            node_id=actor_id,
            node_type="human_agent",
            name=getattr(actor, "name", actor_id),
            state={
                "center": list(center),
                "size": list(getattr(actor, "size", [])),
                "facing": getattr(actor, "facing", 0.0),
                "current_area_id": current_area_id or "",
                "posture": getattr(actor, "posture", "standing"),
                "interaction_elements": list(getattr(actor, "interaction_elements", []) or []),
                "interaction_method": getattr(actor, "interaction_method", "") or "",
                "gaze_target": getattr(actor, "gaze_target", "") or "",
                "worn_items": list(getattr(actor, "worn_items", []) or []),
                "body_surface": getattr(actor, "body_surface", "dry_clean") or "dry_clean",
                "text_to_motion_description": getattr(actor, "text_to_motion_description", "") or "",
            },
        )
        if current_area_id:
            self.add_edge(
                from_node_id=actor_id,
                relation="located_in",
                to_node_id=current_area_id,
                edge_kind="fact",
            )

    def add_temporary_element(
        self,
        temporary_element: TemporaryElement,
        *,
        holder_actor_id: str = "",
    ) -> None:
        node_id = temporary_element.temporary_element_id
        self.add_node(
            node_id=node_id,
            node_type="temporary_element",
            name=temporary_element.name,
            state={
                "anchor_element_id": temporary_element.anchor_element_id,
                "anchor_area_id": temporary_element.anchor_area_id,
                "lifecycle": temporary_element.lifecycle,
                "status": temporary_element.status,
                "created_by_action": temporary_element.created_by_action,
            },
        )
        if temporary_element.anchor_element_id:
            self.add_edge(
                from_node_id=node_id,
                relation="anchored_to",
                to_node_id=temporary_element.anchor_element_id,
                edge_kind="affiliation",
            )
        if temporary_element.anchor_area_id:
            self.add_edge(
                from_node_id=node_id,
                relation="located_in",
                to_node_id=temporary_element.anchor_area_id,
                edge_kind="affiliation",
            )
        if holder_actor_id and temporary_element.status == "held":
            self.add_edge(
                from_node_id=holder_actor_id,
                relation="holding",
                to_node_id=node_id,
                edge_kind="fact",
            )

    def add_node(
        self,
        *,
        node_id: str,
        node_type: str,
        name: str,
        state: dict | None = None,
    ) -> GraphNode:
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            state=dict(state or {}),
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        *,
        from_node_id: str,
        relation: str,
        to_node_id: str,
        edge_kind: str = "fact",
        state: dict | None = None,
        edge_id: str = "",
    ) -> GraphEdge:
        if not edge_id:
            edge_id = self._stable_edge_id(
                from_node_id=from_node_id,
                relation=relation,
                to_node_id=to_node_id,
                edge_kind=edge_kind,
            )
        edge = GraphEdge(
            edge_id=edge_id,
            from_node_id=from_node_id,
            relation=relation,
            to_node_id=to_node_id,
            edge_kind=edge_kind,
            state=dict(state or {}),
        )
        self.edges[edge_id] = edge
        return edge

    def add_fact_edge(
        self,
        *,
        from_node_id: str,
        relation: str,
        to_node_id: str,
        state: dict | None = None,
        edge_id: str = "",
    ) -> GraphEdge:
        return self.add_edge(
            from_node_id=from_node_id,
            relation=relation,
            to_node_id=to_node_id,
            edge_kind="fact",
            state=state,
            edge_id=edge_id,
        )

    def remove_edges(
        self,
        *,
        from_node_id: str = "",
        relation: str = "",
        to_node_id: str = "",
        edge_kind: str = "",
    ) -> list[GraphEdge]:
        removed: list[GraphEdge] = []
        for edge_id, edge in list(self.edges.items()):
            if from_node_id and edge.from_node_id != from_node_id:
                continue
            if relation and edge.relation != relation:
                continue
            if to_node_id and edge.to_node_id != to_node_id:
                continue
            if edge_kind and edge.edge_kind != edge_kind:
                continue
            removed.append(self.edges.pop(edge_id))
        return removed

    def update_node_state(self, node_id: str, updates: dict) -> GraphNode | None:
        node = self.nodes.get(node_id)
        if node is None:
            return None
        node.state.update(updates)
        return node

    def append_node_history(
        self,
        node_id: str,
        entry: dict,
        *,
        limit: int = 8,
    ) -> None:
        node = self.nodes.get(node_id)
        if node is None or not isinstance(entry, dict):
            return
        node.recent_history.append(dict(entry))
        if limit > 0 and len(node.recent_history) > limit:
            node.recent_history = node.recent_history[-limit:]

    def recent_node_history(self, node_id: str, *, limit: int = 6) -> list[dict]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        history = list(node.recent_history)
        if limit > 0:
            history = history[-limit:]
        return [dict(item) for item in history if isinstance(item, dict)]

    def edges_for_node(self, node_id: str, *, edge_kind: str = "") -> list[GraphEdge]:
        return [
            edge
            for edge in self.edges.values()
            if (edge.from_node_id == node_id or edge.to_node_id == node_id)
            and (not edge_kind or edge.edge_kind == edge_kind)
        ]

    def temporary_elements_without_affiliation(self) -> list[GraphNode]:
        result: list[GraphNode] = []
        for node in self.nodes.values():
            if node.node_type != "temporary_element":
                continue
            has_affiliation = any(
                edge.from_node_id == node.node_id and edge.edge_kind == "affiliation"
                for edge in self.edges.values()
            )
            if not has_affiliation:
                result.append(node)
        return result

    def to_prompt_context(self) -> dict:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }

    def to_dict(self) -> dict:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }

    def summary(self) -> dict:
        node_counts: dict[str, int] = {}
        edge_counts: dict[str, int] = {}
        for node in self.nodes.values():
            node_counts[node.node_type] = node_counts.get(node.node_type, 0) + 1
        for edge in self.edges.values():
            edge_counts[edge.edge_kind] = edge_counts.get(edge.edge_kind, 0) + 1
        return {
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "temporary_elements_without_affiliation": [
                node.node_id for node in self.temporary_elements_without_affiliation()
            ],
        }

    def _stable_edge_id(
        self,
        *,
        from_node_id: str,
        relation: str,
        to_node_id: str,
        edge_kind: str,
    ) -> str:
        return f"{edge_kind}:{from_node_id}:{relation}:{to_node_id}"

    def _area_id_for_point(self, home: Home, point: tuple[float, float]) -> str:
        x, y = point
        for area in home.areas:
            x_min, y_min, x_max, y_max = area.bounds
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return area.node_id
        return ""

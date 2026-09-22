from __future__ import annotations

from agent.belief.spatial_memory.spatial_belief import SpatialBelief
from contextual_world.structure.scene_schema import Area, Element
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine

from .types import AreaInput, ElementChange, EnvironmentInput, VisualInput, VisibleElementInput


def initialize_spatial_belief(engine: PhysicsEngine) -> SpatialBelief:
    return SpatialBelief.from_home(engine.home, seen_step=0)


def build_visual_input(agent, engine: PhysicsEngine, current_area: Area) -> VisualInput:
    visible = engine.get_visible_elements(
        observer_center=tuple(agent.center),
        facing_degrees=float(agent.facing),
        field_of_view_degrees=float(agent.field_of_view_degrees),
        area_id=current_area.node_id,
    )
    return VisualInput(
        facing_degrees=float(agent.facing) % 360.0,
        field_of_view_degrees=float(agent.field_of_view_degrees),
        visible_elements=[_visible_element_from_world(item.element) for item in visible],
    )


def build_environment_input(
    agent,
    engine: PhysicsEngine,
    current_area: Area,
    *,
    changed_elements: list[ElementChange],
) -> EnvironmentInput:
    return EnvironmentInput(
        scene_id=str(getattr(engine.home, "node_id", "") or ""),
        scene_name=str(getattr(engine.home, "name", "") or getattr(engine.home, "node_id", "") or ""),
        current_area=AreaInput(
            area_id=current_area.node_id,
            area_name=current_area.name,
        ),
        changed_elements=changed_elements,
        previous_world_feedback=str(getattr(agent, "last_world_feedback_summary", "") or "").strip(),
    )


def detect_changed_elements(agent, current_area: Area, visible_element_ids: set[str]) -> list[ElementChange]:
    previous_snapshot = agent.belief.get_area(current_area.node_id) if getattr(agent, "belief", None) is not None else None
    if previous_snapshot is None:
        return []

    changed_elements: list[ElementChange] = []
    for element in current_area.elements:
        if element.node_id not in visible_element_ids:
            continue
        old = previous_snapshot.elements.get(element.node_id)
        if old is None:
            continue
        for status_kind, before_status, after_status in _element_status_changes(old, element):
            changed_elements.append(
                ElementChange(
                    element_id=element.node_id,
                    name=element.name,
                    before_status=before_status,
                    after_status=after_status,
                    status_kind=status_kind,
                )
            )
    return changed_elements


def _visible_element_from_world(element: Element) -> VisibleElementInput:
    return VisibleElementInput(
        element_id=element.node_id,
        name=element.name,
        physical_status=element.physical_status,
        evolution_status=element.evolution_status,
        interaction_status=element.interaction_status,
        semantic_type=element.semantic_type,
        state_details=dict(element.state_details),
        movable=element.movable,
        blocks_movement=element.blocks_movement,
    )


def _element_status_changes(old, element: Element) -> list[tuple[str, str, str]]:
    changes: list[tuple[str, str, str]] = []
    comparisons = [
        ("physical", old.physical_status, element.physical_status),
        ("evolution", old.evolution_status, element.evolution_status),
        ("interaction", old.interaction_status, element.interaction_status),
    ]
    for status_kind, before_status, after_status in comparisons:
        if before_status != after_status:
            changes.append((status_kind, before_status, after_status))
    detail_keys = set(old.state_details) | set(element.state_details)
    for key in sorted(detail_keys):
        before_status = old.state_details.get(key, "")
        after_status = element.state_details.get(key, "")
        if before_status != after_status:
            changes.append((f"detail:{key}", before_status or "unset", after_status or "unset"))
    return changes

from __future__ import annotations

from dataclasses import dataclass

from .physics_rules import get_physics_profile
from .physics_types import AABB, MoveResult
from .physics_utils import aabb_from_center_size, aabb_inside, aabb_overlap
from .agent_vision import bounds_intersect_view, normalize_field_of_view
from contextual_world.structure.scene_schema import Area, Element, Home


@dataclass
class QueryElement:
    area_id: str
    element: Element


class PhysicsEngine:
    def __init__(self, home: Home) -> None:
        self.home = home

    def reset(self) -> None:
        self.home.reset_all_statuses()

    def get_home_bounds(self) -> AABB:
        x_values: list[float] = []
        y_values: list[float] = []
        for area in self.home.areas:
            x_min, y_min, x_max, y_max = area.bounds
            x_values.extend([x_min, x_max])
            y_values.extend([y_min, y_max])
        return AABB(
            x_min=min(x_values),
            y_min=min(y_values),
            x_max=max(x_values),
            y_max=max(y_values),
        )

    def get_area(self, area_id: str) -> Area | None:
        return self.home.find_area(area_id)

    def get_element(self, node_id: str) -> Element | None:
        return self.home.find_element(node_id)

    def get_element_by_name(self, name: str) -> Element | None:
        return self.home.find_element_by_name(name)

    def get_area_bounds(self, area: Area) -> AABB:
        x_min, y_min, x_max, y_max = area.bounds
        return AABB(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def get_element_bounds(self, element: Element) -> AABB:
        return aabb_from_center_size(element.center, element.size)

    def get_area_for_element(self, node_id: str) -> Area | None:
        for area in self.home.areas:
            if area.find_element(node_id) is not None:
                return area
        return None

    def get_area_for_point(self, x: float, y: float) -> Area | None:
        for area in self.home.areas:
            x_min, y_min, x_max, y_max = area.bounds
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return area
        return None

    def iter_elements(self) -> list[QueryElement]:
        items: list[QueryElement] = []
        for area in self.home.areas:
            for element in area.elements:
                items.append(QueryElement(area_id=area.node_id, element=element))
        return items

    def get_blocking_elements(self, *, exclude_node_id: str | None = None) -> list[QueryElement]:
        blocking: list[QueryElement] = []
        for item in self.iter_elements():
            if exclude_node_id is not None and item.element.node_id == exclude_node_id:
                continue
            profile = get_physics_profile(item.element)
            if profile.is_blocking:
                blocking.append(item)
        return blocking

    def can_place_box(
        self,
        *,
        new_center: tuple[float, float],
        size: tuple[float, float],
        exclude_node_id: str | None = None,
    ) -> MoveResult:
        target_bounds = aabb_from_center_size(new_center, size)
        target_area = self.get_area_for_point(*new_center)
        if target_area is None:
            return MoveResult(False, "outside_known_areas", target_bounds)

        if not aabb_inside(target_bounds, self.get_home_bounds()):
            return MoveResult(
                False,
                "outside_home_bounds",
                target_bounds,
                target_area_id=target_area.node_id,
            )

        for blocking in self.get_blocking_elements(exclude_node_id=exclude_node_id):
            if aabb_overlap(target_bounds, self.get_element_bounds(blocking.element)):
                return MoveResult(
                    False,
                    "blocked_by_element",
                    target_bounds,
                    blocking_element_id=blocking.element.node_id,
                    target_area_id=blocking.area_id,
                )

        return MoveResult(
            True,
            "ok",
            target_bounds,
            target_area_id=target_area.node_id,
        )

    def get_nearby_elements(
        self,
        *,
        center: tuple[float, float],
        radius: float,
    ) -> list[QueryElement]:
        nearby: list[QueryElement] = []
        query_bounds = aabb_from_center_size(center, (radius * 2, radius * 2))
        for item in self.iter_elements():
            if aabb_overlap(query_bounds, self.get_element_bounds(item.element)):
                nearby.append(item)
        return nearby

    def get_visible_elements(
        self,
        *,
        observer_center: tuple[float, float],
        facing_degrees: float,
        field_of_view_degrees: float,
        area_id: str | None = None,
    ) -> list[QueryElement]:
        field_of_view = normalize_field_of_view(field_of_view_degrees)
        candidates = self.iter_elements()
        if area_id is not None:
            candidates = [item for item in candidates if item.area_id == area_id]
        return [
            item
            for item in candidates
            if bounds_intersect_view(
                observer=observer_center,
                facing_degrees=facing_degrees,
                field_of_view_degrees=field_of_view,
                bounds=self.get_element_bounds(item.element),
            )
        ]

    def can_place_element(
        self,
        *,
        node_id: str,
        new_center: tuple[float, float],
        allowed_area_id: str | None = None,
    ) -> MoveResult:
        element = self.get_element(node_id)
        if element is None:
            empty = AABB(0.0, 0.0, 0.0, 0.0)
            return MoveResult(False, "element_not_found", empty)
        if not element.movable:
            return MoveResult(False, "element_not_movable", self.get_element_bounds(element))

        area = self.get_area_for_element(node_id)
        if area is None:
            empty = AABB(0.0, 0.0, 0.0, 0.0)
            return MoveResult(False, "area_not_found", empty)

        target_bounds = aabb_from_center_size(new_center, element.size)
        target_area = self.get_area_for_point(*new_center)

        if target_area is None:
            return MoveResult(False, "outside_known_areas", target_bounds)

        if allowed_area_id is not None and target_area.node_id != allowed_area_id:
            return MoveResult(
                False,
                "wrong_target_area",
                target_bounds,
                target_area_id=target_area.node_id,
            )

        area_bounds = self.get_area_bounds(area)
        if not aabb_inside(target_bounds, area_bounds):
            return MoveResult(
                False,
                "outside_area_bounds",
                target_bounds,
                target_area_id=area.node_id,
            )

        for blocking in self.get_blocking_elements(exclude_node_id=node_id):
            if aabb_overlap(target_bounds, self.get_element_bounds(blocking.element)):
                return MoveResult(
                    False,
                    "blocked_by_element",
                    target_bounds,
                    blocking_element_id=blocking.element.node_id,
                    target_area_id=blocking.area_id,
                )

        return MoveResult(
            True,
            "ok",
            target_bounds,
            target_area_id=target_area.node_id,
        )

    def move_element(
        self,
        *,
        node_id: str,
        dx: float,
        dy: float,
        allowed_area_id: str | None = None,
    ) -> MoveResult:
        element = self.get_element(node_id)
        if element is None:
            empty = AABB(0.0, 0.0, 0.0, 0.0)
            return MoveResult(False, "element_not_found", empty)

        current_x, current_y = element.center
        new_center = (current_x + dx, current_y + dy)
        result = self.can_place_element(
            node_id=node_id,
            new_center=new_center,
            allowed_area_id=allowed_area_id,
        )
        if result.success:
            element.center = new_center
        return result

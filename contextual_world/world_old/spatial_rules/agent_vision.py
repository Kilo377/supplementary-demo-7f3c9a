from __future__ import annotations

from math import cos, radians, sin

from .physics_types import AABB


DEFAULT_FIELD_OF_VIEW_DEGREES = 160.0
_EPSILON = 1e-9


def bounds_intersect_view(
    *,
    observer: tuple[float, float],
    facing_degrees: float,
    field_of_view_degrees: float,
    bounds: AABB,
) -> bool:
    field_of_view = normalize_field_of_view(field_of_view_degrees)
    if field_of_view >= 360.0:
        return True
    if _point_in_bounds(observer, bounds):
        return True

    for point in _bounds_points(bounds):
        if _point_in_view(
            observer,
            point,
            facing_degrees=facing_degrees,
            half_view_degrees=field_of_view / 2.0,
        ):
            return True

    for boundary_degrees in (
        facing_degrees - field_of_view / 2.0,
        facing_degrees + field_of_view / 2.0,
    ):
        direction = (cos(radians(boundary_degrees)), sin(radians(boundary_degrees)))
        if _ray_intersects_bounds(observer, direction, bounds):
            return True
    return False


def normalize_field_of_view(value: float) -> float:
    return min(360.0, max(0.1, float(value)))


def _point_in_view(
    observer: tuple[float, float],
    point: tuple[float, float],
    *,
    facing_degrees: float,
    half_view_degrees: float,
) -> bool:
    dx = point[0] - observer[0]
    dy = point[1] - observer[1]
    if abs(dx) <= _EPSILON and abs(dy) <= _EPSILON:
        return True
    facing = (cos(radians(facing_degrees)), sin(radians(facing_degrees)))
    length = (dx * dx + dy * dy) ** 0.5
    cosine = (facing[0] * dx + facing[1] * dy) / length
    return cosine + _EPSILON >= cos(radians(half_view_degrees))


def _ray_intersects_bounds(
    origin: tuple[float, float],
    direction: tuple[float, float],
    bounds: AABB,
) -> bool:
    t_min = 0.0
    t_max = float("inf")
    for origin_value, direction_value, lower, upper in (
        (origin[0], direction[0], bounds.x_min, bounds.x_max),
        (origin[1], direction[1], bounds.y_min, bounds.y_max),
    ):
        if abs(direction_value) <= _EPSILON:
            if origin_value < lower or origin_value > upper:
                return False
            continue
        first = (lower - origin_value) / direction_value
        second = (upper - origin_value) / direction_value
        if first > second:
            first, second = second, first
        t_min = max(t_min, first)
        t_max = min(t_max, second)
        if t_min > t_max:
            return False
    return t_max >= 0.0


def _bounds_points(bounds: AABB) -> list[tuple[float, float]]:
    return [
        bounds.center,
        (bounds.x_min, bounds.y_min),
        (bounds.x_min, bounds.y_max),
        (bounds.x_max, bounds.y_min),
        (bounds.x_max, bounds.y_max),
    ]


def _point_in_bounds(point: tuple[float, float], bounds: AABB) -> bool:
    return bounds.x_min <= point[0] <= bounds.x_max and bounds.y_min <= point[1] <= bounds.y_max

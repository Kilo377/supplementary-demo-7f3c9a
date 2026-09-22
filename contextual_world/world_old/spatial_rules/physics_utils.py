from __future__ import annotations

from .physics_types import AABB


def aabb_from_center_size(
    center: tuple[float, float],
    size: tuple[float, float],
) -> AABB:
    center_x, center_y = center
    width, height = size
    return AABB(
        x_min=center_x - width / 2,
        y_min=center_y - height / 2,
        x_max=center_x + width / 2,
        y_max=center_y + height / 2,
    )


def aabb_overlap(a: AABB, b: AABB) -> bool:
    return not (
        a.x_max <= b.x_min
        or a.x_min >= b.x_max
        or a.y_max <= b.y_min
        or a.y_min >= b.y_max
    )


def aabb_inside(inner: AABB, outer: AABB) -> bool:
    return (
        inner.x_min >= outer.x_min
        and inner.y_min >= outer.y_min
        and inner.x_max <= outer.x_max
        and inner.y_max <= outer.y_max
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BodyType = Literal["static", "dynamic"]
ColliderType = Literal["box"]


@dataclass(frozen=True)
class AABB:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def moved(self, dx: float, dy: float) -> "AABB":
        return AABB(
            x_min=self.x_min + dx,
            y_min=self.y_min + dy,
            x_max=self.x_max + dx,
            y_max=self.y_max + dy,
        )


@dataclass(frozen=True)
class PhysicsProfile:
    body_type: BodyType
    collider: ColliderType = "box"
    is_blocking: bool = False
    is_movable: bool = False


@dataclass(frozen=True)
class MoveResult:
    success: bool
    reason: str
    target_bounds: AABB
    blocking_element_id: str | None = None
    target_area_id: str | None = None

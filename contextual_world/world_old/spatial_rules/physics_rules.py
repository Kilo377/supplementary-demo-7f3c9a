from __future__ import annotations

from contextual_world.structure.scene_schema import Element
from .physics_types import PhysicsProfile


def get_physics_profile(element: Element) -> PhysicsProfile:
    return PhysicsProfile(
        body_type="static",
        is_blocking=element.blocks_movement,
        is_movable=element.movable,
    )

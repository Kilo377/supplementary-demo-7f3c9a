from __future__ import annotations

from contextual_world.structure.scene_schema import Home
from contextual_world.structure.scene_tree import build_scene_tree

from .elements import AREA_ELEMENTS, UNITY_HOME_BLOCKING_ELEMENT_IDS
from .layout import AREA_DEFINITIONS, PORTAL_DEFINITIONS, RENDERING, UNITY_HOME_DEFAULT_AGENT_START


def build_unity_home_tree() -> Home:
    return build_scene_tree(
        scene_id="unity_home",
        scene_name="unity_imported_home",
        area_definitions=AREA_DEFINITIONS,
        area_elements=AREA_ELEMENTS,
        blocking_element_ids=UNITY_HOME_BLOCKING_ELEMENT_IDS,
        default_agent_start=UNITY_HOME_DEFAULT_AGENT_START,
        portal_definitions=PORTAL_DEFINITIONS,
        rendering=RENDERING,
    )

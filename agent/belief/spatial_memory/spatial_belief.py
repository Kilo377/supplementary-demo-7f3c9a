from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from contextual_world.structure.scene_schema import Area, Element, Home


@dataclass
class ElementBeliefSnapshot:
    element_id: str
    name: str
    area_id: str
    center: tuple[float, float]
    size: tuple[float, float]
    movable: bool
    physical_status: str
    evolution_status: str
    interaction_status: str
    state_details: dict[str, str] = field(default_factory=dict)
    last_seen_step: int = 0

    @property
    def status(self) -> str:
        return self.physical_status

    @status.setter
    def status(self, value: str) -> None:
        self.physical_status = value

    def to_resolution_dict(self) -> dict:
        return {
            "id": self.element_id,
            "name": self.name,
            "area_id": self.area_id,
            "physical_status": self.physical_status,
            "evolution_status": self.evolution_status,
            "interaction_status": self.interaction_status,
            "state_details": dict(self.state_details),
            "last_seen_step": self.last_seen_step,
        }


@dataclass
class AreaBeliefSnapshot:
    area_id: str
    area_name: str
    bounds: tuple[float, float, float, float]
    elements: dict[str, ElementBeliefSnapshot] = field(default_factory=dict)
    last_seen_step: int = 0

    def find_element_by_name(self, element_name: str) -> ElementBeliefSnapshot | None:
        for element in self.elements.values():
            if element.name == element_name:
                return element
        return None

    def to_resolution_dict(self, *, include_elements: bool = True) -> dict:
        data = {
            "area_id": self.area_id,
            "area_name": self.area_name,
            "last_seen_step": self.last_seen_step,
        }
        if include_elements:
            data["elements"] = [
                element.to_resolution_dict()
                for element in self.elements.values()
            ]
        return data


@dataclass
class SpatialBelief:
    home_id: str
    areas: dict[str, AreaBeliefSnapshot] = field(default_factory=dict)

    @classmethod
    def from_home(cls, home: Home, *, seen_step: int = 0) -> "SpatialBelief":
        belief = cls(home_id=home.node_id)
        for area in home.areas:
            belief.areas[area.node_id] = cls._area_snapshot(area, seen_step=seen_step)
        return belief

    @staticmethod
    def _element_snapshot(
        element: Element,
        area_id: str,
        *,
        seen_step: int,
    ) -> ElementBeliefSnapshot:
        return ElementBeliefSnapshot(
            element_id=element.node_id,
            name=element.name,
            area_id=area_id,
            center=element.center,
            size=element.size,
            movable=element.movable,
            physical_status=element.physical_status,
            evolution_status=element.evolution_status,
            interaction_status=element.interaction_status,
            state_details=dict(element.state_details),
            last_seen_step=seen_step,
        )

    @classmethod
    def _area_snapshot(cls, area: Area, *, seen_step: int) -> AreaBeliefSnapshot:
        snapshot = AreaBeliefSnapshot(
            area_id=area.node_id,
            area_name=area.name,
            bounds=area.bounds,
            last_seen_step=seen_step,
        )
        for element in area.elements:
            snapshot.elements[element.node_id] = cls._element_snapshot(
                element,
                area.node_id,
                seen_step=seen_step,
            )
        return snapshot

    def refresh_area_from_world(self, area: Area, *, seen_step: int) -> None:
        self.areas[area.node_id] = self._area_snapshot(area, seen_step=seen_step)

    def refresh_visible_elements_from_world(
        self,
        area: Area,
        *,
        visible_element_ids: set[str],
        seen_step: int,
    ) -> None:
        snapshot = self.areas.get(area.node_id)
        if snapshot is None:
            snapshot = AreaBeliefSnapshot(
                area_id=area.node_id,
                area_name=area.name,
                bounds=area.bounds,
            )
            self.areas[area.node_id] = snapshot
        snapshot.last_seen_step = seen_step
        for element in area.elements:
            if element.node_id not in visible_element_ids:
                continue
            snapshot.elements[element.node_id] = self._element_snapshot(
                element,
                area.node_id,
                seen_step=seen_step,
            )

    def get_area(self, area_id: str) -> AreaBeliefSnapshot | None:
        return self.areas.get(area_id)

    def get_element(self, element_id: str) -> ElementBeliefSnapshot | None:
        for element in self.iter_elements():
            if element.element_id == element_id:
                return element
        return None

    def iter_areas(self) -> Iterable[AreaBeliefSnapshot]:
        return self.areas.values()

    def iter_elements(self) -> Iterable[ElementBeliefSnapshot]:
        for area in self.areas.values():
            yield from area.elements.values()

    def find_area_by_name(self, area_name: str) -> AreaBeliefSnapshot | None:
        for area in self.areas.values():
            if area.area_name == area_name:
                return area
        return None

    def find_element_by_name(
        self,
        element_name: str,
        *,
        current_area_id: str | None = None,
    ) -> ElementBeliefSnapshot | None:
        if current_area_id:
            current_area = self.get_area(current_area_id)
            if current_area is not None:
                element = current_area.find_element_by_name(element_name)
                if element is not None:
                    return element
        for area in self.areas.values():
            element = area.find_element_by_name(element_name)
            if element is not None:
                return element
        return None

    def elements_mentioned_in(
        self,
        text: str,
        *,
        current_area_id: str | None = None,
    ) -> list[ElementBeliefSnapshot]:
        matches: list[tuple[int, int, int, ElementBeliefSnapshot]] = []
        for element in self.iter_elements():
            name_index = text.rfind(element.name)
            id_index = text.rfind(element.element_id)
            index = max(name_index, id_index)
            if index < 0:
                continue
            current_area_score = 1 if element.area_id == current_area_id else 0
            matches.append((current_area_score, len(element.name), index, element))
        matches.sort(reverse=True, key=lambda item: item[:3])
        return [item[3] for item in matches]

    def to_action_resolution_state(
        self,
        *,
        current_area_id: str,
        thought_text: str = "",
    ) -> dict:
        current_area = self.get_area(current_area_id)
        mentioned_elements = self.elements_mentioned_in(thought_text, current_area_id=current_area_id)
        current_ids = set(current_area.elements.keys()) if current_area is not None else set()
        cross_area_mentions = [
            element.to_resolution_dict()
            for element in mentioned_elements
            if element.element_id not in current_ids
        ]
        return {
            "source": "agent_spatial_belief",
            "refresh_policy": (
                "The agent knows the full home layout from the start. "
                "Only the current area is refreshed on each perception; "
                "other areas keep their last perceived states."
            ),
            "area_index": [
                area.to_resolution_dict(include_elements=False)
                for area in self.areas.values()
            ],
            "current_area": (
                current_area.to_resolution_dict(include_elements=True)
                if current_area is not None
                else None
            ),
            "cross_area_mentioned_elements": cross_area_mentions,
        }

    def to_target_resolution_state(
        self,
        *,
        current_area_id: str,
        thought_text: str = "",
    ) -> dict:
        return self.to_action_resolution_state(
            current_area_id=current_area_id,
            thought_text=thought_text,
        )

from __future__ import annotations

from dataclasses import dataclass, field

from .temporary_element import (
    TemporaryElement,
    TemporaryElementChange,
    TemporaryElementCreation,
    TemporaryElementUpdate,
)


@dataclass
class TemporaryElementRegistry:
    elements: dict[str, TemporaryElement] = field(default_factory=dict)
    next_index: int = 1

    def create(
        self,
        creation: TemporaryElementCreation,
        *,
        created_by_action: str = "",
    ) -> TemporaryElementChange | None:
        name = creation.name.strip()
        anchor_element_id = creation.anchor_element_id.strip()
        if not name or not anchor_element_id:
            return None
        element_id = self._next_id()
        element = TemporaryElement(
            temporary_element_id=element_id,
            name=name,
            anchor_element_id=anchor_element_id,
            anchor_area_id=creation.anchor_area_id.strip(),
            lifecycle=creation.lifecycle.strip() or "game",
            status=creation.status.strip() or "held",
            created_by_action=created_by_action,
        )
        self.elements[element_id] = element
        return TemporaryElementChange(
            temporary_element_id=element_id,
            name=element.name,
            change_type="created",
            new_status=element.status,
            new_anchor_element_id=element.anchor_element_id,
        )

    def apply_update(self, update: TemporaryElementUpdate) -> TemporaryElementChange | None:
        element = self.elements.get(update.temporary_element_id)
        if element is None:
            return None
        old_status = element.status
        old_anchor = element.anchor_element_id
        if update.status:
            element.status = update.status.strip()
        if update.anchor_element_id:
            element.anchor_element_id = update.anchor_element_id.strip()
        if update.anchor_area_id:
            element.anchor_area_id = update.anchor_area_id.strip()
        if old_status == element.status and old_anchor == element.anchor_element_id:
            return None
        return TemporaryElementChange(
            temporary_element_id=element.temporary_element_id,
            name=element.name,
            change_type="updated",
            old_status=old_status,
            new_status=element.status,
            old_anchor_element_id=old_anchor,
            new_anchor_element_id=element.anchor_element_id,
        )

    def apply_creations(
        self,
        creations: list[TemporaryElementCreation],
        *,
        created_by_action: str = "",
    ) -> list[TemporaryElementChange]:
        changes = []
        for creation in creations:
            change = self.create(creation, created_by_action=created_by_action)
            if change is not None:
                changes.append(change)
        return changes

    def apply_updates(self, updates: list[TemporaryElementUpdate]) -> list[TemporaryElementChange]:
        changes = []
        for update in updates:
            change = self.apply_update(update)
            if change is not None:
                changes.append(change)
        return changes

    def alive(self) -> list[TemporaryElement]:
        return [element for element in self.elements.values() if element.is_alive]

    def held(self) -> list[TemporaryElement]:
        return [element for element in self.alive() if element.status == "held"]

    def to_prompt_context(self) -> list[dict]:
        return [element.to_dict() for element in self.alive()]

    def held_prompt_context(self) -> list[dict]:
        return [element.to_dict() for element in self.held()]

    def reset(self) -> None:
        self.elements.clear()
        self.next_index = 1

    def _next_id(self) -> str:
        element_id = f"temporary_{self.next_index:03d}"
        self.next_index += 1
        return element_id

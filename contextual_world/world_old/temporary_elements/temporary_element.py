from __future__ import annotations

from dataclasses import dataclass


ENDED_STATUSES = {"consumed", "disposed", "discarded"}


@dataclass
class TemporaryElement:
    temporary_element_id: str
    name: str
    anchor_element_id: str
    anchor_area_id: str
    lifecycle: str
    status: str
    created_by_action: str = ""

    @property
    def is_alive(self) -> bool:
        return self.status not in ENDED_STATUSES

    def to_dict(self) -> dict:
        return {
            "temporary_element_id": self.temporary_element_id,
            "name": self.name,
            "anchor_element_id": self.anchor_element_id,
            "anchor_area_id": self.anchor_area_id,
            "lifecycle": self.lifecycle,
            "status": self.status,
            "created_by_action": self.created_by_action,
        }


@dataclass
class TemporaryElementCreation:
    name: str
    anchor_element_id: str
    lifecycle: str
    status: str
    anchor_area_id: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "anchor_element_id": self.anchor_element_id,
            "anchor_area_id": self.anchor_area_id,
            "lifecycle": self.lifecycle,
            "status": self.status,
        }


@dataclass
class TemporaryElementUpdate:
    temporary_element_id: str
    status: str
    anchor_element_id: str = ""
    anchor_area_id: str = ""

    def to_dict(self) -> dict:
        return {
            "temporary_element_id": self.temporary_element_id,
            "status": self.status,
            "anchor_element_id": self.anchor_element_id,
            "anchor_area_id": self.anchor_area_id,
        }


@dataclass
class TemporaryElementChange:
    temporary_element_id: str
    name: str
    change_type: str
    old_status: str = ""
    new_status: str = ""
    old_anchor_element_id: str = ""
    new_anchor_element_id: str = ""

    def to_dict(self) -> dict:
        return {
            "temporary_element_id": self.temporary_element_id,
            "name": self.name,
            "change_type": self.change_type,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "old_anchor_element_id": self.old_anchor_element_id,
            "new_anchor_element_id": self.new_anchor_element_id,
        }

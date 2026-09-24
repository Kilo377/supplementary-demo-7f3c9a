from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ElementChange:
    element_id: str
    name: str
    before_status: str
    after_status: str
    status_kind: str = "physical"

    @property
    def label(self) -> str:
        if self.status_kind.startswith("detail:"):
            return f"Detail status {self.status_kind.split(':', 1)[1]}"
        return {
            "physical": "Physical state",
            "evolution": "Evolutionary state",
            "interaction": "Interaction state",
        }.get(self.status_kind, "State")

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "name": self.name,
            "before_status": self.before_status,
            "after_status": self.after_status,
            "status_kind": self.status_kind,
        }


@dataclass
class VisibleElementInput:
    element_id: str
    name: str
    physical_status: str
    evolution_status: str
    interaction_status: str
    semantic_type: str = ""
    state_details: dict[str, str] = field(default_factory=dict)
    movable: bool = False
    blocks_movement: bool = False

    @property
    def status(self) -> str:
        return self.physical_status

    @property
    def description(self) -> str:
        if self.physical_status == "regular":
            return f"{self.name} remains unchanged."
        return f"{self.name} is now in a {self.physical_status} state."

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "name": self.name,
            "physical_status": self.physical_status,
            "evolution_status": self.evolution_status,
            "interaction_status": self.interaction_status,
            "semantic_type": self.semantic_type,
            "state_details": dict(self.state_details),
            "movable": self.movable,
            "blocks_movement": self.blocks_movement,
        }


@dataclass
class AreaInput:
    area_id: str
    area_name: str

    def to_dict(self) -> dict:
        return {
            "area_id": self.area_id,
            "area_name": self.area_name,
        }


@dataclass
class EnvironmentInput:
    scene_id: str
    scene_name: str
    current_area: AreaInput
    changed_elements: list[ElementChange] = field(default_factory=list)
    previous_world_feedback: str = ""

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "scene_name": self.scene_name,
            "current_area": self.current_area.to_dict(),
            "changed_elements": [item.to_dict() for item in self.changed_elements],
            "previous_world_feedback": self.previous_world_feedback,
        }


@dataclass
class PhysicalSelfInput:
    position: tuple[float, float]
    size: tuple[float, float]
    facing: float
    posture: str
    interaction_elements: list[dict] = field(default_factory=list)
    interaction_method: str = ""
    gaze_target: str = ""
    worn_items: list[str] = field(default_factory=list)
    body_surface: str = "dry_clean"

    def to_dict(self) -> dict:
        return {
            "position": list(self.position),
            "size": list(self.size),
            "facing": self.facing,
            "posture": self.posture,
            "interaction_elements": list(self.interaction_elements),
            "interaction_method": self.interaction_method,
            "gaze_target": self.gaze_target,
            "worn_items": list(self.worn_items),
            "body_surface": self.body_surface,
        }


@dataclass
class DesireFeelingInput:
    source: str
    text: str
    intensity: str = "normal"
    semantic_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "text": self.text,
            "intensity": self.intensity,
            "semantic_keys": list(self.semantic_keys),
        }


@dataclass
class PsychologicalSelfInput:
    desire_feelings: list[DesireFeelingInput] = field(default_factory=list)

    @property
    def feeling_lines(self) -> list[str]:
        return [item.text for item in self.desire_feelings if item.text.strip()]

    def feelings_for_sources(self, sources: set[str]) -> list[str]:
        return [
            item.text
            for item in self.desire_feelings
            if item.source in sources and item.text.strip()
        ]

    def to_dict(self) -> dict:
        return {
            "desire_feelings": [item.to_dict() for item in self.desire_feelings],
            "feeling_lines": self.feeling_lines,
        }


@dataclass
class SelfInput:
    physical: PhysicalSelfInput
    psychological: PsychologicalSelfInput

    def to_dict(self) -> dict:
        return {
            "physical": self.physical.to_dict(),
            "psychological": self.psychological.to_dict(),
        }


@dataclass
class VisualInput:
    facing_degrees: float
    field_of_view_degrees: float
    visible_elements: list[VisibleElementInput] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "facing_degrees": self.facing_degrees,
            "field_of_view_degrees": self.field_of_view_degrees,
            "visible_elements": [item.to_dict() for item in self.visible_elements],
        }


@dataclass
class AttentionItem:
    source: str
    text: str
    reason: str = ""
    salience: float = 0.5
    noteworthy: bool = True

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "text": self.text,
            "reason": self.reason,
            "salience": self.salience,
            "noteworthy": self.noteworthy,
        }


@dataclass
class AttentionResult:
    items: list[AttentionItem] = field(default_factory=list)

    @property
    def has_noteworthy(self) -> bool:
        return any(item.noteworthy for item in self.items)

    def to_dict(self) -> dict:
        return {
            "has_noteworthy": self.has_noteworthy,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass
class PerceiveResult:
    agent_id: str
    agent_name: str
    step_id: int
    first_time_visit: bool
    environment_input: EnvironmentInput
    self_input: SelfInput
    visual_input: VisualInput
    attention: AttentionResult = field(default_factory=AttentionResult)

    @property
    def world_input(self) -> EnvironmentInput:
        return self.environment_input

    @property
    def area_id(self) -> str:
        return self.environment_input.current_area.area_id

    @property
    def area_name(self) -> str:
        return self.environment_input.current_area.area_name

    @property
    def changed_elements(self) -> list[ElementChange]:
        return self.environment_input.changed_elements

    @property
    def perceived_elements(self) -> list[VisibleElementInput]:
        return self.visual_input.visible_elements

    @property
    def world_feedback_text(self) -> str:
        return self.environment_input.previous_world_feedback

    @property
    def agent_state_text(self) -> str:
        physical = self.self_input.physical
        parts = [f"{self.agent_name} is currently in {self.area_name}."]
        posture = _posture_label(physical.posture)
        if posture:
            parts.append(f"{self.agent_name}{posture}。")
        if physical.interaction_method:
            parts.append(f"{self.agent_name} is {physical.interaction_method}.")
        if physical.body_surface and physical.body_surface != "dry_clean":
            parts.append(f"{self.agent_name}'s body surface {_body_surface_label(physical.body_surface)}.")
        if physical.worn_items:
            parts.append(f"{self.agent_name} is wearing {_join_names(physical.worn_items)}.")
        if physical.gaze_target:
            parts.append(f"{self.agent_name}'s gaze is directed at {physical.gaze_target}.")
        return " ".join(parts).strip()

    @property
    def narration_facts(self) -> list[str]:
        facts = [f"{self.agent_name} has entered {self.area_name}."]
        if self.first_time_visit:
            facts.append("This is the first time visiting this area in this process.")
        else:
            facts.append(f"A comparison has been made between here and {self.agent_name}'s memory of home.")
        if self.changed_elements:
            for change in self.changed_elements:
                facts.append(f"{change.name}'s {change.label} changed from {change.before_status} to {change.after_status}.")
        else:
            facts.append("No obvious changes are observed in this area at present.")
        for item in self.perceived_elements:
            facts.append(item.description)
        return facts

    @property
    def environment_text(self) -> str:
        all_details = " ".join(item.description for item in self.perceived_elements)
        if self.changed_elements:
            change_text = "，".join(
                f"The {item.label} of {item.name} has changed from {item.before_status} to {item.after_status}"
                for item in self.changed_elements
            )
            return f"This place is not exactly as remembered, {change_text}. {all_details}".strip()
        mood = "Everything before me carries a familiar and quiet aura." if self.first_time_visit else "This place is almost identical to how I remember it; everything is as usual."
        return f"{mood} {all_details}".strip()

    @property
    def narration_text(self) -> str:
        intro = f"{self.agent_name} is in {self.area_name}."
        if self.changed_elements:
            change_text = "，".join(
                f"The {item.label} of {item.name} has changed from {item.before_status} to {item.after_status}"
                for item in self.changed_elements
            )
            return f"{intro} {self.agent_name} quickly notices that this place isn't exactly as remembered: {change_text}. {_element_details(self.perceived_elements)}".strip()
        mood = "Everything before me carries a familiar and quiet aura." if self.first_time_visit else f"This place is almost identical to how {self.agent_name} remembers it; everything is as usual."
        return f"{intro} {mood} {_element_details(self.perceived_elements)}".strip()

    @property
    def notice_text(self) -> str:
        if self.first_time_visit:
            return self.narration_text
        if self.changed_elements:
            change_text = "，".join(
                f"{item.name}'s {item.label} changed from {item.before_status} to {item.after_status}"
                for item in self.changed_elements
            )
            return f"{self.agent_name} notices a change in {self.area_name}: {change_text}."
        return ""

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "step_id": self.step_id,
            "first_time_visit": self.first_time_visit,
            "environment_input": self.environment_input.to_dict(),
            "self_input": self.self_input.to_dict(),
            "visual_input": self.visual_input.to_dict(),
            "attention": self.attention.to_dict(),
        }

    def format_for_prompt(
        self,
        *,
        include_physical: bool = True,
        include_visual: bool = True,
        include_psychological: bool = True,
        include_attention: bool = True,
        include_world_feedback: bool = False,
        include_changed_elements: bool = False,
    ) -> str:
        parts = [f"Just now, {self.agent_name} looked around."]
        if include_physical:
            physical_text = self.physical_prompt_text()
            if physical_text:
                parts.append(physical_text)
        if include_visual:
            visual_text = self.visual_prompt_text()
            if visual_text:
                parts.append(visual_text)
        if include_psychological:
            psychological_text = self.psychological_prompt_text()
            if psychological_text:
                parts.append(psychological_text)
        if include_attention:
            attention_text = self.attention_prompt_text()
            if attention_text:
                parts.append(attention_text)
        if include_world_feedback and self.world_feedback_text:
            parts.append(self.world_feedback_text)
        if include_changed_elements:
            changed_text = self.changed_elements_prompt_text()
            if changed_text:
                parts.append(changed_text)
        return " ".join(part for part in parts if part.strip()).strip()

    def physical_prompt_text(self) -> str:
        physical = self.self_input.physical
        posture = _posture_label(physical.posture)
        parts = [f"{self.agent_name} is currently in {self.area_name}{posture}." if posture else f"{self.agent_name} is currently in {self.area_name}."]
        if physical.interaction_method:
            parts.append(f"{self.agent_name} is {physical.interaction_method}.")
        if physical.body_surface and physical.body_surface != "dry_clean":
            parts.append(f"{self.agent_name}'s body surface {_body_surface_label(physical.body_surface)}.")
        if physical.worn_items:
            parts.append(f"{self.agent_name} is wearing {_join_names(physical.worn_items)}.")
        if physical.gaze_target:
            parts.append(f"{self.agent_name}'s gaze is directed at {physical.gaze_target}.")
        return " ".join(parts).strip()

    def psychological_prompt_text(self) -> str:
        return " ".join(self.self_input.psychological.feeling_lines).strip()

    def attention_prompt_text(self) -> str:
        fragments = []
        for item in self.attention.items:
            if not item.noteworthy or not item.text.strip():
                continue
            text = _strip_sentence_end(item.text.strip())
            if text.startswith("It is worth noting that"):
                fragments.append(text)
            else:
                fragments.append(f"It is worth noting that {text}")
        if not fragments:
            return ""
        return "，".join(fragments) + "。"

    def changed_elements_prompt_text(self) -> str:
        if not self.changed_elements:
            return ""
        fragments = [
            f"{item.name}'s {item.label} has changed from {item.before_status} to {item.after_status}"
            for item in self.changed_elements
        ]
        return "Environmental changes:" + "，".join(fragments) + "。"

    def visual_prompt_text(self) -> str:
        if not self.perceived_elements:
            return ""
        descriptions = "，".join(_visual_element_description(item) for item in self.perceived_elements if item.name)
        if not descriptions:
            return ""
        return f"{self.agent_name} sees in their field of view: {descriptions}."


def _element_details(elements: list[VisibleElementInput]) -> str:
    return " ".join(item.description for item in elements)


def _visual_element_description(element: VisibleElementInput) -> str:
    details = element.state_details
    contains = str(details.get("contains", "") or "").strip().lower()
    if contains == "water":
        return f"There is water in {element.name}"
    if contains in {"empty", "none"}:
        if element.semantic_type == "cup":
            return f"There is no water in {element.name}"
        return f"{element.name} is empty"
    if contains:
        contains_labels = {
            "fruit": "Fruit",
            "milk": "Milk",
            "food": "Food",
            "food_and_drinks": "Food and drinks",
            "seasoning": "Condiments",
            "books": "Books",
            "trash": "Trash",
        }
        return f"{element.name} contains {contains_labels.get(contains, contains)}"

    state_labels = {
        ("power_state", "on"): "Open",
        ("power_state", "off"): "Closed",
        ("door_state", "open"): "Open",
        ("door_state", "closed"): "Closed",
        ("curtain_state", "open"): "Pulled open",
        ("curtain_state", "closed"): "Pulled",
        ("plant_state", "normal"): "Normal status",
        ("clean_state", "usable"): "Usable",
    }
    for key, value in details.items():
        label = state_labels.get((key, str(value).lower()))
        if label:
            return f"{element.name}{label}"
    if element.physical_status != "regular":
        return f"{element.name} is currently in a {element.physical_status} state"
    return f"{element.name} remains unchanged"


def _posture_label(value: str) -> str:
    return {
        "standing": "standing",
        "sitting": "sitting",
        "lying": "lying down",
        "crouching": "crouching",
        "walking": "Walking",
    }.get(value, value)


def _body_surface_label(value: str) -> str:
    return {
        "wet_clean": "Moist but clean",
        "wet_dirty": "Moist and somewhat dirty",
        "soapy": "Foamy",
        "dirty": "A bit dirty",
        "sweaty": "A bit sweaty",
        "dry_dirty": "Dry but a bit dirty",
        "wet": "Wet",
    }.get(value, value)


def _join_names(items) -> str:
    return "、".join(str(item) for item in items if str(item).strip())


def _strip_sentence_end(text: str) -> str:
    return text.rstrip("。！？.!? ")

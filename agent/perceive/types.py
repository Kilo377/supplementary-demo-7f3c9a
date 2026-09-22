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
            return f"细节状态 {self.status_kind.split(':', 1)[1]}"
        return {
            "physical": "物理状态",
            "evolution": "演化状态",
            "interaction": "交互状态",
        }.get(self.status_kind, "状态")

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
            return f"{self.name}还保持着原来的样子。"
        return f"{self.name}现在呈现出{self.physical_status}的状态。"

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
        parts = [f"{self.agent_name}现在在{self.area_name}。"]
        posture = _posture_label(physical.posture)
        if posture:
            parts.append(f"{self.agent_name}{posture}。")
        if physical.interaction_method:
            parts.append(f"{self.agent_name}正在{physical.interaction_method}。")
        if physical.body_surface and physical.body_surface != "dry_clean":
            parts.append(f"{self.agent_name}身体表面{_body_surface_label(physical.body_surface)}。")
        if physical.worn_items:
            parts.append(f"{self.agent_name}穿戴着{_join_names(physical.worn_items)}。")
        if physical.gaze_target:
            parts.append(f"{self.agent_name}的注意方向落在{physical.gaze_target}。")
        return " ".join(parts).strip()

    @property
    def narration_facts(self) -> list[str]:
        facts = [f"{self.agent_name}进入了{self.area_name}。"]
        if self.first_time_visit:
            facts.append("这是这次流程里第一次来到这里。")
        else:
            facts.append(f"这里和{self.agent_name}记忆中的家进行了一次对照。")
        if self.changed_elements:
            for change in self.changed_elements:
                facts.append(f"{change.name}的{change.label}从{change.before_status}变成了{change.after_status}。")
        else:
            facts.append("这一带目前没有看出明显变化。")
        for item in self.perceived_elements:
            facts.append(item.description)
        return facts

    @property
    def environment_text(self) -> str:
        all_details = " ".join(item.description for item in self.perceived_elements)
        if self.changed_elements:
            change_text = "，".join(
                f"{item.name}的{item.label}已经从{item.before_status}变成了{item.after_status}"
                for item in self.changed_elements
            )
            return f"这里和记忆里并不完全一样，{change_text}。{all_details}".strip()
        mood = "眼前的一切都带着熟悉而安静的气息。" if self.first_time_visit else "这里和记忆中的样子几乎没有区别，一切如常。"
        return f"{mood} {all_details}".strip()

    @property
    def narration_text(self) -> str:
        intro = f"{self.agent_name}正在{self.area_name}。"
        if self.changed_elements:
            change_text = "，".join(
                f"{item.name}的{item.label}已经从{item.before_status}变成了{item.after_status}"
                for item in self.changed_elements
            )
            return f"{intro} {self.agent_name}很快察觉到这里和记忆里并不完全一样，{change_text}。{_element_details(self.perceived_elements)}".strip()
        mood = "眼前的一切都带着熟悉而安静的气息。" if self.first_time_visit else f"这里和{self.agent_name}记忆中的样子几乎没有区别，一切如常。"
        return f"{intro} {mood} {_element_details(self.perceived_elements)}".strip()

    @property
    def notice_text(self) -> str:
        if self.first_time_visit:
            return self.narration_text
        if self.changed_elements:
            change_text = "，".join(
                f"{item.name}的{item.label}从{item.before_status}变成{item.after_status}"
                for item in self.changed_elements
            )
            return f"{self.agent_name}注意到{self.area_name}里有变化：{change_text}。"
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
        parts = [f"就在刚刚，{self.agent_name}看了一下眼前。"]
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
        parts = [f"{self.agent_name}现在在{self.area_name}{posture}。" if posture else f"{self.agent_name}现在在{self.area_name}。"]
        if physical.interaction_method:
            parts.append(f"{self.agent_name}正在{physical.interaction_method}。")
        if physical.body_surface and physical.body_surface != "dry_clean":
            parts.append(f"{self.agent_name}身体表面{_body_surface_label(physical.body_surface)}。")
        if physical.worn_items:
            parts.append(f"{self.agent_name}穿戴着{_join_names(physical.worn_items)}。")
        if physical.gaze_target:
            parts.append(f"{self.agent_name}的注意方向落在{physical.gaze_target}。")
        return " ".join(parts).strip()

    def psychological_prompt_text(self) -> str:
        return " ".join(self.self_input.psychological.feeling_lines).strip()

    def attention_prompt_text(self) -> str:
        fragments = []
        for item in self.attention.items:
            if not item.noteworthy or not item.text.strip():
                continue
            text = _strip_sentence_end(item.text.strip())
            if text.startswith("值得注意的是"):
                fragments.append(text)
            else:
                fragments.append(f"值得注意的是，{text}")
        if not fragments:
            return ""
        return "，".join(fragments) + "。"

    def changed_elements_prompt_text(self) -> str:
        if not self.changed_elements:
            return ""
        fragments = [
            f"{item.name}的{item.label}从{item.before_status}变成了{item.after_status}"
            for item in self.changed_elements
        ]
        return "环境变化：" + "，".join(fragments) + "。"

    def visual_prompt_text(self) -> str:
        if not self.perceived_elements:
            return ""
        descriptions = "，".join(_visual_element_description(item) for item in self.perceived_elements if item.name)
        if not descriptions:
            return ""
        return f"{self.agent_name}的视线里看见了：{descriptions}。"


def _element_details(elements: list[VisibleElementInput]) -> str:
    return " ".join(item.description for item in elements)


def _visual_element_description(element: VisibleElementInput) -> str:
    details = element.state_details
    contains = str(details.get("contains", "") or "").strip().lower()
    if contains == "water":
        return f"{element.name}里有水"
    if contains in {"empty", "none"}:
        if element.semantic_type == "cup":
            return f"{element.name}里没有水"
        return f"{element.name}里面是空的"
    if contains:
        contains_labels = {
            "fruit": "水果",
            "milk": "牛奶",
            "food": "食物",
            "food_and_drinks": "食物和饮料",
            "seasoning": "调料",
            "books": "书",
            "trash": "垃圾",
        }
        return f"{element.name}里有{contains_labels.get(contains, contains)}"

    state_labels = {
        ("power_state", "on"): "开着",
        ("power_state", "off"): "关着",
        ("door_state", "open"): "开着",
        ("door_state", "closed"): "关着",
        ("curtain_state", "open"): "拉开着",
        ("curtain_state", "closed"): "拉着",
        ("plant_state", "normal"): "状态正常",
        ("clean_state", "usable"): "可以使用",
    }
    for key, value in details.items():
        label = state_labels.get((key, str(value).lower()))
        if label:
            return f"{element.name}{label}"
    if element.physical_status != "regular":
        return f"{element.name}现在呈现出{element.physical_status}的状态"
    return f"{element.name}还保持着原来的样子"


def _posture_label(value: str) -> str:
    return {
        "standing": "站着",
        "sitting": "坐着",
        "lying": "躺着",
        "crouching": "蹲着",
        "walking": "走动中",
    }.get(value, value)


def _body_surface_label(value: str) -> str:
    return {
        "wet_clean": "湿润但干净",
        "wet_dirty": "湿润且有些脏",
        "soapy": "有泡沫",
        "dirty": "有些脏",
        "sweaty": "有些出汗",
        "dry_dirty": "干燥但有些脏",
        "wet": "湿的",
    }.get(value, value)


def _join_names(items) -> str:
    return "、".join(str(item) for item in items if str(item).strip())


def _strip_sentence_end(text: str) -> str:
    return text.rstrip("。！？.!? ")

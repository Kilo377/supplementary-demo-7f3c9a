from __future__ import annotations

from dataclasses import dataclass, field

from agent.belief.short_time_memory import MemoryEpisode, ShortTermMemory
from agent.belief.spatial_memory.spatial_belief import (
    ElementBeliefSnapshot,
    SpatialBelief,
)
from agent.intent.state import IntentState
from agent.perceive import ElementChange, PerceiveResult, VisibleElementInput
from contextual_world.structure.scene_schema import Element
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine


@dataclass
class SensoryInput:
    agent_name: str
    scene_name: str
    area_id: str
    area_name: str
    prompt_text: str = ""
    agent_state_text: str = ""
    world_feedback_text: str = ""
    environment_text: str = ""
    notice_text: str = ""
    changed_elements: list[ElementChange] = field(default_factory=list)
    perceived_elements: list[VisibleElementInput] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        if self.prompt_text.strip():
            return self.prompt_text.strip()
        lines = []
        _append_unique_line(lines, f"{self.agent_name}现在正在{self.scene_name}的{self.area_name}。")
        for text in [
            self.agent_state_text,
            self.world_feedback_text,
            self.environment_text,
            self.notice_text,
        ]:
            cleaned = text.strip()
            _append_unique_line(lines, cleaned)
        return "\n".join(lines)


@dataclass
class AttentionItem:
    source: str
    text: str
    salience: float = 0.5
    noteworthy: bool = True

    def format_for_prompt(self) -> str:
        text = self.text.strip()
        if self.noteworthy and text and not text.startswith("值得注意的是"):
            text = f"值得注意的是，{text}"
        return f"- [{self.source}] {text}"


@dataclass
class SpatialRetrieval:
    scene_name: str
    current_area_id: str
    current_area_name: str
    current_element_texts: list[str] = field(default_factory=list)
    other_area_names: list[str] = field(default_factory=list)

    def format_for_prompt(self, *, agent_name: str) -> str:
        lines = [
            f"{agent_name}现在正在{self.scene_name}的{self.current_area_name}。",
        ]
        if self.current_element_texts:
            lines.append(f"这个房间里有：{_join_texts(self.current_element_texts)}。")
        else:
            lines.append("这个房间里暂时没有可见元素。")

        if self.other_area_names:
            lines.append(f"{agent_name}在{self.scene_name}里，还知道其他房间：{_join_texts(self.other_area_names)}。")
        else:
            lines.append(f"{agent_name}在{self.scene_name}里，暂时没有其他房间信息。")
        return "\n".join(lines)


@dataclass
class ExperienceRetrieval:
    agent_name: str
    biography_text: str = ""
    intent_text: str = ""
    intent_status: str = "active"
    intent_progress: list[str] = field(default_factory=list)
    episodes: list[MemoryEpisode] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        sections = []
        doing_lines = [f"Intent: {self.intent_text} [{self.intent_status}]"]
        if self.intent_progress:
            doing_lines.append("当前进度：")
            doing_lines.extend(f"- {item}" for item in self.intent_progress if item)
        sections.append(f"{self.agent_name}正在做的事：\n" + "\n".join(doing_lines))

        biography = self.biography_text.strip()
        if biography:
            sections.append(f"{self.agent_name}的人物传记：\n{biography}")

        episode_text = _format_episodes(self.episodes)
        sections.append(f"{self.agent_name}已经经历过的事情：\n{episode_text}")
        return "\n\n".join(section for section in sections if section.strip())


@dataclass
class WorkingMemoryFrame:
    agent_name: str
    sensory_input: SensoryInput
    attention_items: list[AttentionItem]
    spatial_retrieval: SpatialRetrieval
    experience_retrieval: ExperienceRetrieval
    time_belief: str = ""
    self_belief: str = ""

    def format_for_action_prompt(self) -> str:
        sections = [
            "Sensory Input：\n" + self.sensory_input.format_for_prompt(),
        ]
        if self.attention_items:
            sections.append(
                "Attention：\n"
                + "\n".join(item.format_for_prompt() for item in self.attention_items)
            )
        time_text = self.time_belief.strip()
        if time_text:
            sections.append(f"Time Belief：\n{time_text}")
        sections.append(
            "Spatial Retrieval：\n"
            + self.spatial_retrieval.format_for_prompt(agent_name=self.agent_name)
        )
        sections.append("Experience Retrieval：\n" + self.experience_retrieval.format_for_prompt())
        self_belief_text = self.self_belief.strip()
        if self_belief_text:
            sections.append(f"{self.agent_name}对自身状态的当前理解：\n{self_belief_text}")
        return "\n\n".join(section for section in sections if section.strip())


def build_working_memory(
    agent,
    engine: PhysicsEngine,
    perception: PerceiveResult,
    *,
    intent: IntentState | None = None,
) -> WorkingMemoryFrame:
    active_intent = intent or getattr(agent, "active_intent", None)
    spatial_retrieval = _build_spatial_retrieval(
        engine,
        getattr(agent, "belief", None),
        perception,
    )
    sensory_input = SensoryInput(
        agent_name=agent.name,
        scene_name=_scene_name(engine),
        area_id=perception.area_id,
        area_name=perception.area_name,
        prompt_text=perception.format_for_prompt(
            include_physical=True,
            include_visual=True,
            include_psychological=True,
            include_attention=True,
            include_world_feedback=False,
            include_changed_elements=False,
        ),
        agent_state_text=perception.agent_state_text,
        world_feedback_text=perception.world_feedback_text,
        environment_text=perception.environment_text,
        notice_text=perception.notice_text,
        changed_elements=list(perception.changed_elements),
        perceived_elements=list(perception.perceived_elements),
    )
    short_time_memory = getattr(agent, "short_time_memory", None)
    experience_retrieval = ExperienceRetrieval(
        agent_name=agent.name,
        biography_text=_biography_text(agent),
        intent_text=getattr(active_intent, "intent_text", "") or getattr(short_time_memory, "intent_text", ""),
        intent_status=getattr(active_intent, "status", "") or getattr(short_time_memory, "status", "active"),
        intent_progress=list(getattr(active_intent, "progress", []) or []),
        episodes=_episodes(short_time_memory),
    )
    return WorkingMemoryFrame(
        agent_name=agent.name,
        sensory_input=sensory_input,
        attention_items=[],
        spatial_retrieval=spatial_retrieval,
        experience_retrieval=experience_retrieval,
        time_belief=_time_belief_text(agent),
        self_belief=str(getattr(getattr(agent, "self_state", None), "self_belief", "") or ""),
    )


def _build_spatial_retrieval(
    engine: PhysicsEngine,
    belief: SpatialBelief | None,
    perception: PerceiveResult,
) -> SpatialRetrieval:
    current_area = belief.get_area(perception.area_id) if belief is not None else None
    if current_area is not None:
        current_area_id = current_area.area_id
        current_area_name = current_area.area_name
        current_element_texts = [_belief_element_text(element) for element in current_area.elements.values()]
        other_area_names = [
            area.area_name
            for area in belief.iter_areas()
            if area.area_id != current_area_id
        ]
    else:
        world_area = engine.get_area(perception.area_id) or engine.get_area_for_point(*getattr(perception.self_input.physical, "position", (0.0, 0.0)))
        current_area_id = world_area.node_id if world_area is not None else perception.area_id
        current_area_name = world_area.name if world_area is not None else perception.area_name
        current_element_texts = [_world_element_text(element) for element in (world_area.elements if world_area is not None else [])]
        other_area_names = [
            area.name
            for area in engine.home.areas
            if area.node_id != current_area_id
        ]
    return SpatialRetrieval(
        scene_name=_scene_name(engine),
        current_area_id=current_area_id,
        current_area_name=current_area_name,
        current_element_texts=current_element_texts,
        other_area_names=other_area_names,
    )


def _build_attention_items(agent, perception: PerceiveResult) -> list[AttentionItem]:
    _ = agent
    attention = getattr(perception, "attention", None)
    raw_items = list(getattr(attention, "items", []) or [])
    items = [
        AttentionItem(
            source=str(getattr(item, "source", "") or "attention"),
            text=str(getattr(item, "text", "") or ""),
            salience=float(getattr(item, "salience", 0.5) or 0.5),
            noteworthy=bool(getattr(item, "noteworthy", True)),
        )
        for item in raw_items
        if bool(getattr(item, "noteworthy", True)) and str(getattr(item, "text", "") or "").strip()
    ]
    items.sort(key=lambda item: item.salience, reverse=True)
    return items


def _time_belief_text(agent) -> str:
    time_belief = getattr(agent, "time_belief", None)
    formatter = getattr(time_belief, "format_for_cognition", None)
    if callable(formatter):
        return str(formatter() or "").strip()
    return ""


def _belief_element_text(element: ElementBeliefSnapshot) -> str:
    details = []
    if element.physical_status != "regular":
        details.append(f"物理状态={element.physical_status}")
    if element.evolution_status != "stable":
        details.append(f"演化状态={element.evolution_status}")
    if element.interaction_status != "idle":
        details.append(f"交互状态={element.interaction_status}")
    for key, value in sorted(element.state_details.items()):
        if str(value).strip():
            details.append(f"{key}={value}")
    if details:
        return f"{element.name}（{_join_texts(details)}）"
    return element.name


def _world_element_text(element: Element) -> str:
    details = []
    if element.physical_status != "regular":
        details.append(f"物理状态={element.physical_status}")
    if element.evolution_status != "stable":
        details.append(f"演化状态={element.evolution_status}")
    if element.interaction_status != "idle":
        details.append(f"交互状态={element.interaction_status}")
    for key, value in sorted(element.state_details.items()):
        if str(value).strip():
            details.append(f"{key}={value}")
    if details:
        return f"{element.name}（{_join_texts(details)}）"
    return element.name


def _biography_text(agent) -> str:
    memory = getattr(agent, "long_term_memory", None)
    if memory is None:
        return ""
    biography = str(getattr(memory, "biography_text", "") or "").strip()
    if biography:
        return biography
    retrieve = getattr(memory, "retrieve", None)
    if callable(retrieve):
        return str(retrieve(max_chars=0) or "").strip()
    return ""


def _episodes(memory: ShortTermMemory | None) -> list[MemoryEpisode]:
    if memory is None:
        return []
    return [
        episode
        for episode in list(getattr(memory, "episodes", []) or [])
        if getattr(episode, "recallable", True)
    ]


def _format_episodes(episodes: list[MemoryEpisode]) -> str:
    if not episodes:
        return "暂无。"
    blocks = []
    for episode in episodes:
        lines = [episode.header_text()]
        if episode.perceived_summary:
            lines.append(f"   notice: {episode.perceived_summary}")
        if episode.intended_action:
            lines.append(f"   intended: {episode.intended_action}")
        if episode.experienced_result:
            lines.append(f"   experienced: {episode.experienced_result}")
        if getattr(episode, "intuition_thought", ""):
            mode = f"{episode.intuition_mode}/" if getattr(episode, "intuition_mode", "") else ""
            route = f"[{mode}{episode.intuition_route}] " if getattr(episode, "intuition_route", "") or mode else ""
            lines.append(f"   intuition: {route}{episode.intuition_thought}")
        if getattr(episode, "arbiter_thought", ""):
            mode = f"[{episode.arbiter_mode}] " if getattr(episode, "arbiter_mode", "") else ""
            lines.append(f"   arbiter: {mode}{episode.arbiter_thought}")
        if episode.interacted_element_names:
            lines.append(f"   interacted: {_join_texts(episode.interacted_element_names)}")
        elif episode.interacted_element_ids:
            lines.append(f"   interacted: {_join_texts(episode.interacted_element_ids)}")
        for fact in episode.learned_facts:
            if fact:
                lines.append(f"   learned: {fact}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _scene_name(engine: PhysicsEngine) -> str:
    return str(getattr(engine.home, "name", "") or getattr(engine.home, "node_id", "") or "当前场景")


def _join_texts(items: list[str]) -> str:
    return "、".join(str(item).strip() for item in items if str(item).strip())


def _append_unique_line(lines: list[str], text: str) -> None:
    cleaned = text.strip()
    if not cleaned:
        return
    for line in lines:
        if cleaned == line or cleaned in line or line in cleaned:
            return
    lines.append(cleaned)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryEpisode:
    step_id: int
    area_id: str
    area_name: str
    perceived_summary: str
    intended_action: str
    experienced_result: str
    intuition_route: str = ""
    intuition_thought: str = ""
    intuition_mode: str = ""
    arbiter_mode: str = ""
    arbiter_thought: str = ""
    interacted_element_ids: list[str] = field(default_factory=list)
    interacted_element_names: list[str] = field(default_factory=list)
    learned_facts: list[str] = field(default_factory=list)
    time_text: str = ""
    recallable: bool = True
    encoding_probability: float = 1.0
    encoding_sample: float | None = None

    def concise_summary(self) -> str:
        intended = self.intended_action.strip()
        experienced = self.experienced_result.strip()
        if intended and experienced and intended != experienced:
            return f"{intended} 结果是，{experienced}"
        if self.experienced_result:
            return self.experienced_result
        if self.intended_action:
            return self.intended_action
        return self.perceived_summary

    def header_text(self) -> str:
        bracketed = self.bracket_text()
        return f"{self.step_id}. {bracketed}".strip()

    def bracket_text(self) -> str:
        labels = []
        if self.time_text.strip():
            labels.append(self.time_text.strip())
        location = (self.area_name or self.area_id).strip()
        if location:
            labels.append(location)
        return " ".join(f"[{label}]" for label in labels)


@dataclass
class ShortTermMemory:
    intent_text: str
    status: str = "active"
    episodes: list[MemoryEpisode] = field(default_factory=list)

    def append_episode(self, episode: MemoryEpisode) -> None:
        self.episodes.append(episode)

    def remember(
        self,
        *,
        step_id: int,
        area_id: str,
        area_name: str,
        perceived_summary: str,
        intended_action: str,
        experienced_result: str,
        intuition_route: str = "",
        intuition_thought: str = "",
        intuition_mode: str = "",
        arbiter_mode: str = "",
        arbiter_thought: str = "",
        time_text: str = "",
        interacted_element_ids: list[str] | None = None,
        interacted_element_names: list[str] | None = None,
        learned_facts: list[str] | None = None,
        recallable: bool = True,
        encoding_probability: float = 1.0,
        encoding_sample: float | None = None,
    ) -> MemoryEpisode:
        episode = MemoryEpisode(
            step_id=step_id,
            area_id=area_id,
            area_name=area_name,
            perceived_summary=perceived_summary,
            intended_action=intended_action,
            experienced_result=experienced_result,
            intuition_route=intuition_route,
            intuition_thought=intuition_thought,
            intuition_mode=intuition_mode,
            arbiter_mode=arbiter_mode,
            arbiter_thought=arbiter_thought,
            interacted_element_ids=interacted_element_ids or [],
            interacted_element_names=interacted_element_names or [],
            learned_facts=learned_facts or [],
            time_text=time_text,
            recallable=recallable,
            encoding_probability=encoding_probability,
            encoding_sample=encoding_sample,
        )
        self.append_episode(episode)
        return episode

    def recent(self, count: int = 5) -> list[MemoryEpisode]:
        if count <= 0:
            return []
        return self.episodes[-count:]

    def recent_recallable(self, count: int = 5) -> list[MemoryEpisode]:
        if count <= 0:
            return []
        return [episode for episode in self.episodes if episode.recallable][-count:]

    def has_episodes(self) -> bool:
        return bool(self.episodes)

    def clear(self) -> None:
        self.episodes.clear()

    def format_experience_for_prompt(
        self,
        count: int = 8,
        *,
        empty_text: str = "",
        exclude_experienced_result: str = "",
    ) -> str:
        recent_episodes = self.recent_recallable(count)
        if not recent_episodes:
            return empty_text

        excluded = exclude_experienced_result.strip()
        lines = []
        for episode in recent_episodes:
            if excluded and episode.experienced_result.strip() == excluded:
                continue
            summary = episode.concise_summary()
            if summary:
                prefix = episode.bracket_text()
                if prefix:
                    lines.append(f"- {prefix} {summary}")
                else:
                    lines.append(f"- {summary}")
            if episode.intuition_thought:
                mode = f"{episode.intuition_mode}/" if episode.intuition_mode else ""
                route = f"[{mode}{episode.intuition_route}] " if episode.intuition_route or mode else ""
                lines.append(f"- intuition: {route}{episode.intuition_thought}")
            if episode.arbiter_thought:
                mode = f"[{episode.arbiter_mode}] " if episode.arbiter_mode else ""
                lines.append(f"- arbiter: {mode}{episode.arbiter_thought}")
            for fact in episode.learned_facts:
                if fact:
                    lines.append(f"- {fact}")
        return "\n".join(lines) if lines else empty_text

    def format_for_prompt(self, count: int = 8) -> str:
        return self.format_experience_for_prompt(count, empty_text="暂无。")

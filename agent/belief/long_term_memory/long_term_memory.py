from __future__ import annotations

from dataclasses import dataclass, field

from agent.habitual_controller.cue_memory import CueMemory


@dataclass
class LongTermMemory:
    agent_name: str
    biography_text: str = ""
    cue_memory: CueMemory = field(default_factory=CueMemory)
    demonstrated_habit_memory: CueMemory = field(default_factory=CueMemory)

    def remember_biography(self, biography_text: str) -> None:
        self.biography_text = biography_text.strip()

    def retrieve(self, query: str = "", *, max_chars: int = 2400) -> str:
        _ = query
        text = self.biography_text.strip()
        if not text:
            return ""
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    def format_for_prompt(self, query: str = "", *, max_chars: int = 2400) -> str:
        text = self.retrieve(query, max_chars=max_chars)
        if not text:
            return ""
        return f"{self.agent_name} 的长期个人背景与自我记忆：\n{text}"

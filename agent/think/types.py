from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThinkResult:
    thought: str
    conclusion: str = ""
    raw_response: str = ""

    def format_for_prompt(self) -> str:
        lines = []
        if self.thought.strip():
            lines.append(f"First-person thought: {self.thought.strip()}")
        if self.conclusion.strip():
            lines.append(f"Tendency after thinking: {self.conclusion.strip()}")
        return "\n".join(lines)

    def format_for_memory(self) -> str:
        if self.conclusion.strip():
            return f"{self.thought.strip()} The conclusion is that {self.conclusion.strip()}"
        return self.thought.strip()


class ThinkError(Exception):
    def __init__(self, message: str, raw_response: str = "", prompt: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.prompt = prompt

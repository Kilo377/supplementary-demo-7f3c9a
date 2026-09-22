from __future__ import annotations

from dataclasses import dataclass

from agent.belief.time_belief import parse_duration_seconds


WAIT_DURATION_OPTIONS = (
    "30s",
    "1min",
    "3min",
    "5min",
    "10min",
    "15min",
    "20min",
    "30min",
)
_WAIT_DURATION_BY_SECONDS = {
    parse_duration_seconds(value): value
    for value in WAIT_DURATION_OPTIONS
}


@dataclass
class IntuitionResult:
    route: str
    thought: str
    target_area_id: str = ""
    target_area_name: str = ""
    chat_target: str = ""
    wait_duration: str = ""
    raw_response: str = ""

    @property
    def is_action(self) -> bool:
        return self.route == "action"


def normalize_wait_duration(value: object, *, default: str = "") -> str:
    seconds = parse_duration_seconds(str(value or ""))
    if seconds <= 0:
        return default
    exact = _WAIT_DURATION_BY_SECONDS.get(seconds)
    if exact:
        return exact
    nearest_seconds = min(_WAIT_DURATION_BY_SECONDS, key=lambda item: abs(item - seconds))
    return _WAIT_DURATION_BY_SECONDS[nearest_seconds]


class IntuitionError(Exception):
    def __init__(self, message: str, raw_response: str = "", prompt: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.prompt = prompt

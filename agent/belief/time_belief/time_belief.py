from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import re


DEFAULT_SIMULATION_DATETIME = datetime(2026, 1, 1, 8, 10)


@dataclass
class TimeBelief:
    current_datetime: datetime = field(default_factory=lambda: DEFAULT_SIMULATION_DATETIME)
    jitter_enabled: bool = True
    jitter_ratio: float = 0.08
    min_ratio: float = 0.75
    max_ratio: float = 1.25
    rng: random.Random = field(default_factory=random.Random, repr=False)

    def reset(self, start_datetime: datetime | None = None) -> None:
        self.current_datetime = start_datetime or DEFAULT_SIMULATION_DATETIME

    def advance(self, estimated_duration: str) -> int:
        base_seconds = parse_duration_seconds(estimated_duration)
        if base_seconds <= 0:
            return 0
        seconds = self._realistic_seconds(base_seconds)
        self.current_datetime += timedelta(seconds=seconds)
        return seconds

    def format_for_cognition(self) -> str:
        return f"It is now {self.format_short_label()}."

    def format_short_label(self) -> str:
        return self.format_datetime_label(self.current_datetime)

    def format_datetime_label(self, value: datetime) -> str:
        return f"{_period_name(value.hour)} {value.hour}:{value.minute:02d}"

    def format_elapsed_since(self, started_at: datetime | None) -> str:
        if started_at is None:
            return "A short while ago"
        elapsed_seconds = max(0, int((self.current_datetime - started_at).total_seconds()))
        elapsed_minutes = elapsed_seconds // 60
        started_label = self.format_datetime_label(started_at)
        if elapsed_minutes < 1:
            relative = "Less than 1 minute ago"
        elif elapsed_minutes < 60:
            relative = f"About {elapsed_minutes} minutes ago"
        else:
            hours, minutes = divmod(elapsed_minutes, 60)
            relative = f"About {hours} hours"
            if minutes:
                relative += f"{minutes} minutes ago"
            else:
                relative += "ago"
        return f"{relative}（{started_label}）"

    def to_dict(self) -> dict:
        return {
            "current_time": self.current_datetime.strftime("%H:%M"),
            "short_label": self.format_short_label(),
            "cognition_text": self.format_for_cognition(),
        }

    def _realistic_seconds(self, base_seconds: int) -> int:
        if not self.jitter_enabled:
            return base_seconds
        sigma = max(1.0, base_seconds * self.jitter_ratio)
        sampled = self.rng.gauss(base_seconds, sigma)
        sampled = max(base_seconds * self.min_ratio, min(base_seconds * self.max_ratio, sampled))
        return max(1, int(round(sampled)))


def parse_duration_seconds(duration_text: str) -> int:
    text = str(duration_text or "").strip().lower()
    if not text:
        return 0
    normalized = (
        text.replace("seconds", "second")
        .replace("minutes", "minute")
        .replace("mins", "min")
        .replace("minutes", "min")
        .replace("minute", "min")
        .replace("seconds", "s")
        .replace("second", "s")
        .replace("secs", "s")
        .replace("sec", "s")
    )
    match = re.search(r"(\d+(?:\.\d+)?)\s*(min|m|minute|seconds?|second)", normalized)
    if not match:
        return 0
    value = float(match.group(1))
    unit = match.group(2)
    if unit in {"min", "m", "minute"}:
        return int(round(value * 60))
    return int(round(value))


def _period_name(hour: int) -> str:
    if 5 <= hour < 12:
        return "AM"
    if 12 <= hour < 14:
        return "noon"
    if 14 <= hour < 18:
        return "PM"
    if 18 <= hour < 24:
        return "evening"
    return "early morning"

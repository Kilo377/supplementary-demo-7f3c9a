from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class IntentState:
    intent_text: str
    decided_at: datetime | None = None
    source_desire: str = ""
    candidate_reason: str = ""
    feasibility_reason: str = ""
    selection_reason: str = ""
    target_element_id: str = ""
    status: str = "active"
    step_count: int = 0
    last_feedback: str = ""
    progress: list[str] = field(default_factory=list)

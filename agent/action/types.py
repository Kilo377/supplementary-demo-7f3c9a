from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionProposalResult:
    action_text: str
    provider_name: str
    model: str | None = None

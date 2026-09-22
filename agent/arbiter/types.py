from dataclasses import dataclass


@dataclass(frozen=True)
class ArbiterResult:
    decision_mode: str
    action_text: str
    reason: str
    raw_response: str = ""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldTransitionCheckResult:
    check_status: str = "accepted"
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "check_status": self.check_status,
            "issues": list(self.issues),
        }

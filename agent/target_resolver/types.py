from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TargetResolution:
    target_element_id: str = ""
    target_element_name: str = ""
    navigation_anchor_element_id: str = ""
    navigation_anchor_element_name: str = ""
    secondary_target_element_id: str = ""
    arrival_completes_action: bool = False
    reason: str = ""
    raw_response: str = ""
    prompt: str = ""
    provider_name: str = ""
    model: str = ""
    duration_seconds: float | None = None
    error: str = ""


class TargetResolutionError(Exception):
    def __init__(
        self,
        message: str,
        raw_response: str = "",
        prompt: str = "",
        duration_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.prompt = prompt
        self.duration_seconds = duration_seconds

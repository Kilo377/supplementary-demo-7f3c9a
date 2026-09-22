from __future__ import annotations

from dataclasses import dataclass

from agent.perceive import PerceiveResult


@dataclass
class IntentAttentionAffect:
    notice_text: str
    attention_reason: str

    def to_dict(self) -> dict:
        return {
            "notice_text": self.notice_text,
            "attention_reason": self.attention_reason,
        }


def build_intent_attention_affect(perception: PerceiveResult | None) -> IntentAttentionAffect | None:
    if perception is None:
        return None

    notice_text = getattr(perception, "notice_text", "").strip()
    if not notice_text:
        return None
    if getattr(perception, "first_time_visit", False) and not getattr(perception, "changed_elements", []):
        return None

    return IntentAttentionAffect(
        notice_text=notice_text,
        attention_reason="当前观察中出现了值得注意的信息，可能临时改变接下来想做什么。",
    )

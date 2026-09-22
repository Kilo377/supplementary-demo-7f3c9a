from __future__ import annotations

from dataclasses import dataclass

from agent.desire.desire_state import DesireState


PHYSIOLOGICAL_INTERPRETATIONS = {
    "hunger": {
        0: "非常饱，吃撑了。",
        1: "不饿。",
        2: "稍微有点饿。",
        3: "饿了，有点想吃东西了。",
        4: "非常饿。很想吃东西。",
    },
    "thirst": {
        0: "一点也不渴。",
        1: "不太渴。",
        2: "稍微有点口渴。",
        3: "渴了，有点想喝水了。",
        4: "非常渴。很想喝水。",
    },
    "hygiene": {
        0: "很干净。",
        1: "还算干净。",
        2: "稍微有点不清爽。",
        3: "有点脏了，想清洁一下。",
        4: "非常脏。很想清洁一下。",
    },
}

INTERNAL_STATE_INTERPRETATIONS = {
    "stress": ["没有压力。", "压力不大。", "有一点压力。", "压力比较明显。", "压力非常大。"],
    "tension": ["很放松。", "不太紧张。", "有一点紧张。", "明显感到紧张。", "非常紧张。"],
    "fatigue": ["一点也不累。", "不太累。", "稍微有点累。", "累了，有点想休息了。", "非常累。很想休息。"],
}

@dataclass
class DesireSubjectiveSignal:
    source_desire: str
    subjective_interpretation: str

    def to_dict(self) -> dict:
        return {
            "source_desire": self.source_desire,
            "subjective_interpretation": self.subjective_interpretation,
        }


def build_desire_subjective_signals(
    desire_state: DesireState,
    *,
    agent_name: str = "Agent",
) -> list[DesireSubjectiveSignal]:
    signals: list[DesireSubjectiveSignal] = []
    physiological = desire_state.physiological_state
    for field_name in ("hunger", "thirst", "hygiene"):
        score = int(getattr(physiological, field_name))
        interpretation = PHYSIOLOGICAL_INTERPRETATIONS[field_name][_likert_band(score)]
        signals.append(
            DesireSubjectiveSignal(
                source_desire=f"physiological_state.{field_name}",
                subjective_interpretation=interpretation,
            )
        )

    internal = desire_state.internal_state
    for field_name in ("stress", "tension", "fatigue"):
        score = int(getattr(internal, field_name))
        signals.append(
            DesireSubjectiveSignal(
                source_desire=f"internal_state.{field_name}",
                subjective_interpretation=INTERNAL_STATE_INTERPRETATIONS[field_name][_likert_band(score)],
            )
        )

    mental = _strip_agent_subject(desire_state.mental.strip(), agent_name=agent_name)
    if mental:
        signals.append(
            DesireSubjectiveSignal(
                source_desire="mental",
                subjective_interpretation=mental,
            )
        )

    for goal in desire_state.work_goal:
        if goal.completed:
            continue
        goal_text = _strip_agent_subject(goal.text.strip(), agent_name=agent_name)
        if not goal_text:
            continue
        signals.append(
            DesireSubjectiveSignal(
                source_desire="work_goal",
                subjective_interpretation=f"未完成的正事目标：{goal_text}",
            )
        )

    return signals


def _likert_band(score: int) -> int:
    if score <= 1:
        return 0
    if score <= 3:
        return 1
    if score <= 6:
        return 2
    if score <= 8:
        return 3
    return 4


def _strip_agent_subject(text: str, *, agent_name: str) -> str:
    stripped = text.strip()
    for prefix in (
        f"{agent_name}现在",
        f"{agent_name}想要",
        f"{agent_name}想",
        f"{agent_name}要",
        agent_name,
    ):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
            break
    return stripped

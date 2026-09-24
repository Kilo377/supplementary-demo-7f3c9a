from __future__ import annotations

from dataclasses import dataclass

from agent.desire.desire_state import DesireState


PHYSIOLOGICAL_INTERPRETATIONS = {
    "hunger": {
        0: "Very full, stuffed.",
        1: "Not hungry at all.",
        2: "A little bit hungry.",
        3: "Hungry, feeling like eating something.",
        4: "Very hungry. Really want to eat.",
    },
    "thirst": {
        0: "Not thirsty at all.",
        1: "Not very thirsty.",
        2: "A little bit thirsty.",
        3: "Thirsty, feeling like drinking some water.",
        4: "Very thirsty. Really want to drink water.",
    },
    "hygiene": {
        0: "Very clean.",
        1: "It's fairly clean.",
        2: "A bit uncomfortable.",
        3: "It's a bit dirty; I want to clean it up.",
        4: "Very dirty. I really want to clean it up.",
    },
}

INTERNAL_STATE_INTERPRETATIONS = {
    "stress": ["No pressure.", "Not much pressure.", "A little pressure.", "Noticeable pressure.", "Very high pressure."],
    "tension": ["Very relaxed.", "Not very tense.", "A little tense.", "Clearly feeling nervous.", "Very nervous."],
    "fatigue": ["Not tired at all.", "Not very tired.", "Slightly tired.", "Tired, and a bit like to rest.", "Very tired. Really want to rest."],
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
                subjective_interpretation=f"Unfinished goal: {goal_text}",
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
        f"{agent_name} now",
        f"{agent_name} wants to",
        f"{agent_name} wants to",
        f"{agent_name} is going to",
        agent_name,
    ):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
            break
    return stripped

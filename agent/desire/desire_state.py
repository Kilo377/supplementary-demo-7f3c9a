from __future__ import annotations

from dataclasses import dataclass, field

from agent.desire.default_desire import (
    DEFAULT_INTERNAL_STATE,
    DEFAULT_MENTAL,
    DEFAULT_PHYSIOLOGICAL_STATE,
    DEFAULT_WORK_GOALS,
    default_mental_for,
    default_work_goals_for,
)


def clamp_likert_score(value: int) -> int:
    return max(0, min(10, int(value)))


@dataclass
class PhysiologicalState:
    hunger: int = DEFAULT_PHYSIOLOGICAL_STATE["hunger"]
    thirst: int = DEFAULT_PHYSIOLOGICAL_STATE["thirst"]
    hygiene: int = DEFAULT_PHYSIOLOGICAL_STATE["hygiene"]

    def __post_init__(self) -> None:
        self.hunger = clamp_likert_score(self.hunger)
        self.thirst = clamp_likert_score(self.thirst)
        self.hygiene = clamp_likert_score(self.hygiene)

    def to_dict(self) -> dict:
        return {
            "hunger": self.hunger,
            "thirst": self.thirst,
            "hygiene": self.hygiene,
        }


@dataclass
class InternalState:
    stress: int = DEFAULT_INTERNAL_STATE["stress"]
    tension: int = DEFAULT_INTERNAL_STATE["tension"]
    fatigue: int = DEFAULT_INTERNAL_STATE["fatigue"]
    depletion: int = 3
    cognitive_load: int = 3

    def __post_init__(self) -> None:
        self.stress = clamp_likert_score(self.stress)
        self.tension = clamp_likert_score(self.tension)
        self.fatigue = clamp_likert_score(self.fatigue)
        self.depletion = clamp_likert_score(self.depletion)
        self.cognitive_load = clamp_likert_score(self.cognitive_load)

    def to_dict(self) -> dict:
        return {
            "stress": self.stress,
            "tension": self.tension,
            "fatigue": self.fatigue,
            "depletion": self.depletion,
            "cognitive_load": self.cognitive_load,
        }


@dataclass
class WorkGoalDesire:
    text: str
    completed: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "completed": self.completed,
        }


@dataclass
class DesireState:
    physiological_state: PhysiologicalState = field(default_factory=PhysiologicalState)
    internal_state: InternalState = field(default_factory=InternalState)
    mental: str = DEFAULT_MENTAL
    work_goal: list[WorkGoalDesire] = field(
        default_factory=lambda: [WorkGoalDesire(text=text) for text in DEFAULT_WORK_GOALS]
    )

    def to_dict(self) -> dict:
        return {
            "physiological_state": self.physiological_state.to_dict(),
            "internal_state": self.internal_state.to_dict(),
            "mental": self.mental,
            "work_goal": [goal.to_dict() for goal in self.work_goal],
        }

    @classmethod
    def for_agent(cls, agent_name: str) -> "DesireState":
        return cls(
            physiological_state=PhysiologicalState(),
            internal_state=InternalState(),
            mental=default_mental_for(agent_name),
            work_goal=[
                WorkGoalDesire(text=text)
                for text in default_work_goals_for(agent_name)
            ],
        )

    @classmethod
    def from_dict(cls, data: dict) -> "DesireState":
        physiological_data = data.get("physiological_state", {}) or {}
        internal_data = data.get("internal_state", {}) or {}
        work_goal_data = data.get("work_goal", []) or []
        return cls(
            physiological_state=PhysiologicalState(
                hunger=physiological_data.get("hunger", DEFAULT_PHYSIOLOGICAL_STATE["hunger"]),
                thirst=physiological_data.get("thirst", DEFAULT_PHYSIOLOGICAL_STATE["thirst"]),
                hygiene=physiological_data.get("hygiene", DEFAULT_PHYSIOLOGICAL_STATE["hygiene"]),
            ),
            internal_state=InternalState(
                stress=internal_data.get("stress", DEFAULT_INTERNAL_STATE["stress"]),
                tension=internal_data.get("tension", DEFAULT_INTERNAL_STATE["tension"]),
                fatigue=internal_data.get("fatigue", DEFAULT_INTERNAL_STATE["fatigue"]),
                depletion=internal_data.get("depletion", 3),
                cognitive_load=internal_data.get("cognitive_load", 3),
            ),
            mental=data.get("mental", "") or DEFAULT_MENTAL,
            work_goal=_work_goals_from_data(work_goal_data),
        )

    def all_work_goals_completed(self) -> bool:
        return bool(self.work_goal) and all(goal.completed for goal in self.work_goal)


def _work_goals_from_data(items) -> list[WorkGoalDesire]:
    goals: list[WorkGoalDesire] = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text", "")
            completed = bool(item.get("completed", False))
        else:
            text = str(item)
            completed = False
        text = str(text).strip()
        if text:
            goals.append(WorkGoalDesire(text=text, completed=completed))
    return goals

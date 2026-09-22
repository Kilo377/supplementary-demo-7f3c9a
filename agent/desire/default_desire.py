from __future__ import annotations

# Default Desire configuration for experiments.
#
# This is the quick-edit location for the agent's starting Desire state.
# Keep independent work goals as separate strings. Do not merge multiple goals
# into one sentence, because desire settlement marks each WorkGoalDesire as
# completed independently.

DEFAULT_PHYSIOLOGICAL_STATE = {
    "hunger": 5,
    "thirst": 3,
    "hygiene": 0,
}

DEFAULT_INTERNAL_STATE = {
    "stress": 5,
    "tension": 3,
    "fatigue": 8,
}

DEFAULT_AGENT_NAME = "Agent"

DEFAULT_MENTAL_TEMPLATE = "{agent_name}现在心情有点焦虑。"

DEFAULT_WORK_GOAL_TEMPLATES = [
    "",

]


def default_mental_for(agent_name: str) -> str:
    return DEFAULT_MENTAL_TEMPLATE.format(agent_name=agent_name or DEFAULT_AGENT_NAME)


def default_work_goals_for(agent_name: str) -> list[str]:
    name = agent_name or DEFAULT_AGENT_NAME
    return [template.format(agent_name=name) for template in DEFAULT_WORK_GOAL_TEMPLATES]


DEFAULT_MENTAL = default_mental_for(DEFAULT_AGENT_NAME)
DEFAULT_WORK_GOALS = default_work_goals_for(DEFAULT_AGENT_NAME)

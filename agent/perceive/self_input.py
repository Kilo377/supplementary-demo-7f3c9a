from __future__ import annotations

from agent.desire import DesireState

from .types import DesireFeelingInput, PhysicalSelfInput, PsychologicalSelfInput, SelfInput


def build_self_input(agent, desire_state: DesireState | None = None) -> SelfInput:
    desire = desire_state or getattr(agent, "desire_state", None) or DesireState.for_agent(getattr(agent, "name", "Agent"))
    return SelfInput(
        physical=PhysicalSelfInput(
            position=tuple(getattr(agent, "center", (0.0, 0.0))),
            size=tuple(getattr(agent, "size", (0.0, 0.0))),
            facing=float(getattr(agent, "facing", 0.0)),
            posture=str(getattr(agent, "posture", "") or ""),
            interaction_elements=list(getattr(agent, "interaction_elements", []) or []),
            interaction_method=str(getattr(agent, "interaction_method", "") or ""),
            gaze_target=str(getattr(agent, "gaze_target", "") or ""),
            worn_items=list(getattr(agent, "worn_items", []) or []),
            body_surface=str(getattr(agent, "body_surface", "") or ""),
        ),
        psychological=_build_psychological_input(getattr(agent, "name", "Agent"), desire),
    )


def _build_psychological_input(agent_name: str, desire_state: DesireState) -> PsychologicalSelfInput:
    feelings = _physiological_feelings(agent_name, desire_state.physiological_state.to_dict())
    feelings.extend(_internal_state_feelings(agent_name, desire_state.internal_state.to_dict()))
    mental = str(desire_state.mental or "").strip()
    if mental:
        feelings.append(
            DesireFeelingInput(
                source="mental",
                text=mental,
                intensity=_mental_intensity(mental),
                semantic_keys=_mental_semantic_keys(mental),
            )
        )
    for goal in desire_state.work_goal:
        if not goal.completed:
            feelings.append(
                DesireFeelingInput(
                    source="work_goal",
                    text=f"{agent_name} still has unresolved motivational pressure: {goal.text}",
                    intensity="normal",
                )
            )

    return PsychologicalSelfInput(desire_feelings=feelings)


def _physiological_feelings(agent_name: str, state: dict[str, int]) -> list[DesireFeelingInput]:
    feelings: list[DesireFeelingInput] = []
    mappings = [
        ("hunger", "Hungry", "Eat something"),
        ("thirst", "Thirsty", "Drink something"),
        ("hygiene", "Unrefreshed", "Cleanse the body or clothing"),
    ]
    for key, label, action_hint in mappings:
        score = int(state.get(key, 0))
        if score >= 7:
            feelings.append(
                DesireFeelingInput(
                    source=f"physiological_state:{key}",
                    text=f"{agent_name} clearly feels {label}, which will drive {agent_name} to {action_hint}.",
                    intensity="strong",
                    semantic_keys=[key],
                )
            )
        elif score >= 4:
            feelings.append(
                DesireFeelingInput(
                    source=f"physiological_state:{key}",
                    text=f"{agent_name} has a slight sense of {label}.",
                    intensity="mild",
                    semantic_keys=[key],
                )
            )
    return feelings


def _internal_state_feelings(agent_name: str, state: dict[str, int]) -> list[DesireFeelingInput]:
    feelings: list[DesireFeelingInput] = []
    labels = {
        "stress": "Stress",
        "tension": "nervous",
        "fatigue": "Fatigue",
    }
    for key, label in labels.items():
        score = int(state.get(key, 0))
        if score < 4:
            continue
        intensity = "strong" if score >= 7 else "mild"
        text = (
            f"{agent_name} clearly feels {label}."
            if intensity == "strong"
            else f"{agent_name} has a slight sense of {label}."
        )
        feelings.append(
            DesireFeelingInput(
                source=f"internal_state:{key}",
                text=text,
                intensity=intensity,
                semantic_keys=[key],
            )
        )
    return feelings


def _mental_intensity(text: str) -> str:
    high_pressure_markers = [
        "Intense",
        "Severe",
        "Very",
        "Panic",
        "Breakdown",
        "Anger",
        "Frustration",
        "Despair",
        "Under great stress",
        "Unable to calm down",
        "Unable to focus",
    ]
    if any(marker in text for marker in high_pressure_markers):
        return "strong"
    if any(marker in text for marker in ["A bit", "A little", "Mild", "Slightly", "Faintly", "Somewhat"]):
        return "mild"
    return "normal"


def _mental_semantic_keys(text: str) -> list[str]:
    mappings = [
        ("anxiety", ["Anxiety", "nervous", "Unease", "Worry"]),
        ("frustration", ["Frustration", "Frustration", "Helplessness", "Irritation", "Annoyance"]),
        ("stress", ["Stress", "Oppressive", "Burden"]),
        ("calm", ["Calm", "relaxed", "easygoing"]),
        ("satisfaction", ["Satisfied", "Sense of achievement", "Gratified"]),
    ]
    keys = [key for key, markers in mappings if any(marker in text for marker in markers)]
    return keys or ["general"]

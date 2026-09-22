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
                    text=f"{agent_name}还有未完成的动机压力：{goal.text}",
                    intensity="normal",
                )
            )

    return PsychologicalSelfInput(desire_feelings=feelings)


def _physiological_feelings(agent_name: str, state: dict[str, int]) -> list[DesireFeelingInput]:
    feelings: list[DesireFeelingInput] = []
    mappings = [
        ("hunger", "饥饿", "吃点东西"),
        ("thirst", "口渴", "喝点东西"),
        ("hygiene", "不清爽", "清洁身体或衣物"),
    ]
    for key, label, action_hint in mappings:
        score = int(state.get(key, 0))
        if score >= 7:
            feelings.append(
                DesireFeelingInput(
                    source=f"physiological_state:{key}",
                    text=f"{agent_name}明显感到{label}，这会推动{agent_name}{action_hint}。",
                    intensity="strong",
                    semantic_keys=[key],
                )
            )
        elif score >= 4:
            feelings.append(
                DesireFeelingInput(
                    source=f"physiological_state:{key}",
                    text=f"{agent_name}有一点{label}感。",
                    intensity="mild",
                    semantic_keys=[key],
                )
            )
    return feelings


def _internal_state_feelings(agent_name: str, state: dict[str, int]) -> list[DesireFeelingInput]:
    feelings: list[DesireFeelingInput] = []
    labels = {
        "stress": "压力",
        "tension": "紧张",
        "fatigue": "疲劳",
    }
    for key, label in labels.items():
        score = int(state.get(key, 0))
        if score < 4:
            continue
        intensity = "strong" if score >= 7 else "mild"
        text = (
            f"{agent_name}明显感到{label}。"
            if intensity == "strong"
            else f"{agent_name}有一点{label}感。"
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
        "强烈",
        "严重",
        "非常",
        "恐慌",
        "崩溃",
        "愤怒",
        "沮丧",
        "绝望",
        "压力很大",
        "无法平静",
        "无法集中",
    ]
    if any(marker in text for marker in high_pressure_markers):
        return "strong"
    if any(marker in text for marker in ["有点", "一点", "轻微", "略", "隐隐", "稍微"]):
        return "mild"
    return "normal"


def _mental_semantic_keys(text: str) -> list[str]:
    mappings = [
        ("anxiety", ["焦虑", "紧张", "不安", "担心"]),
        ("frustration", ["挫败", "沮丧", "无奈", "烦躁", "恼火"]),
        ("stress", ["压力", "压迫", "负担"]),
        ("calm", ["平静", "放松", "轻松"]),
        ("satisfaction", ["满足", "成就感", "欣慰"]),
    ]
    keys = [key for key, markers in mappings if any(marker in text for marker in markers)]
    return keys or ["general"]

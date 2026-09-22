from __future__ import annotations

import json

from agent.belief.spatial_memory.spatial_belief import SpatialBelief

from .types import StateTransitionPromptContext


def build_state_transition_context(
    agent,
    *,
    action_text: str,
    intent_text: str = "",
    recent_experience_count: int = 8,
) -> StateTransitionPromptContext:
    return StateTransitionPromptContext(
        agent_name=str(getattr(agent, "name", "Agent") or "Agent"),
        action_text=str(action_text or "").strip(),
        intent_text=_intent_text(agent, explicit_text=intent_text),
        spatial_belief_text=format_complete_spatial_belief(
            getattr(agent, "belief", None)
        ),
        time_text=_time_text(agent),
        physical_state_text=_physical_state_text(agent),
        internal_state_text=_internal_state_text(agent),
        recent_experience_text=_recent_experience_text(
            agent,
            count=recent_experience_count,
        ),
    )


def format_complete_spatial_belief(belief: SpatialBelief | None) -> str:
    if belief is None:
        return ""
    lines = [f"这个家的空间记忆编号是 {belief.home_id}。"]
    for area in belief.iter_areas():
        lines.append(
            f"- {area.area_name}（area_id={area.area_id}，"
            f"bounds={_number_tuple(area.bounds)}，"
            f"最近在第{area.last_seen_step}次感知时更新）"
        )
        for element in area.elements.values():
            properties = [
                f"element_id={element.element_id}",
                f"center={_number_tuple(element.center)}",
                f"size={_number_tuple(element.size)}",
                f"movable={str(element.movable).lower()}",
                f"physical_status={element.physical_status}",
                f"evolution_status={element.evolution_status}",
                f"interaction_status={element.interaction_status}",
            ]
            if element.state_details:
                properties.append(
                    "state_details="
                    + json.dumps(
                        element.state_details,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            properties.append(f"最近在第{element.last_seen_step}次感知时确认")
            lines.append(f"  - {element.name}（{'，'.join(properties)}）")
    return "\n".join(lines)


def _physical_state_text(agent) -> str:
    name = str(getattr(agent, "name", "Agent") or "Agent")
    current_area_id = str(getattr(agent, "current_area_id", "") or "")
    area_name = _area_name(getattr(agent, "belief", None), current_area_id)
    location = area_name or current_area_id
    center = tuple(getattr(agent, "center", (0.0, 0.0)))
    lines = [
        f"{name}位于{location or '尚未确认的区域'}，坐标是{_number_tuple(center)}。",
        f"{name}当前姿态是{getattr(agent, 'posture', '') or '未知'}，"
        f"朝向{_number(getattr(agent, 'facing', 0.0))}度。",
    ]
    interaction_elements = list(getattr(agent, "interaction_elements", []) or [])
    if interaction_elements:
        lines.append(
            f"{name}当前接触或持有：{_interaction_text(interaction_elements)}。"
        )
    interaction_method = str(getattr(agent, "interaction_method", "") or "").strip()
    if interaction_method:
        lines.append(f"{name}当前的交互方式是{interaction_method}。")
    gaze_target = str(getattr(agent, "gaze_target", "") or "").strip()
    if gaze_target:
        lines.append(f"{name}正在注视{gaze_target}。")
    worn_items = [
        str(item).strip()
        for item in list(getattr(agent, "worn_items", []) or [])
        if str(item).strip()
    ]
    if worn_items:
        lines.append(f"{name}穿戴着{'、'.join(worn_items)}。")
    body_surface = str(getattr(agent, "body_surface", "") or "").strip()
    if body_surface:
        lines.append(f"{name}的身体表面状态是{body_surface}。")
    return "\n".join(lines)


def _internal_state_text(agent) -> str:
    desire = getattr(agent, "desire_state", None)
    if desire is None:
        return ""
    name = str(getattr(agent, "name", "Agent") or "Agent")
    physiological = desire.physiological_state.to_dict()
    internal = desire.internal_state.to_dict()
    lines = [
        "生理需求采用0到10的程度："
        f"饥饿{physiological['hunger']}，口渴{physiological['thirst']}，"
        f"清洁需求{physiological['hygiene']}。",
        "内部状态采用0到10的程度："
        f"压力{internal['stress']}，紧张{internal['tension']}，"
        f"疲劳{internal['fatigue']}。",
    ]
    mental = str(getattr(desire, "mental", "") or "").strip()
    if mental:
        lines.append(mental)
    active_goals = [
        _strip_terminal_punctuation(str(goal.text))
        for goal in list(getattr(desire, "work_goal", []) or [])
        if not bool(getattr(goal, "completed", False))
        and _strip_terminal_punctuation(str(goal.text))
    ]
    if active_goals:
        lines.append(f"{name}仍然在意：{'；'.join(active_goals)}。")
    return "\n".join(lines)


def _recent_experience_text(agent, *, count: int) -> str:
    memory = getattr(agent, "short_time_memory", None)
    formatter = getattr(memory, "format_experience_for_prompt", None)
    if not callable(formatter):
        return ""
    return str(formatter(count=max(0, count), empty_text="") or "").strip()


def _time_text(agent) -> str:
    time_belief = getattr(agent, "time_belief", None)
    formatter = getattr(time_belief, "format_short_label", None)
    if not callable(formatter):
        return ""
    return str(formatter() or "").strip()


def _intent_text(agent, *, explicit_text: str) -> str:
    explicit = str(explicit_text or "").strip()
    if explicit:
        return explicit
    active_intent = getattr(agent, "active_intent", None)
    active_text = str(getattr(active_intent, "intent_text", "") or "").strip()
    if active_text:
        return active_text
    short_time_memory = getattr(agent, "short_time_memory", None)
    return str(getattr(short_time_memory, "intent_text", "") or "").strip()


def _area_name(belief: SpatialBelief | None, area_id: str) -> str:
    if belief is None or not area_id:
        return ""
    area = belief.get_area(area_id)
    return area.area_name if area is not None else ""


def _interaction_text(items: list[dict]) -> str:
    texts = []
    for item in items:
        if isinstance(item, dict):
            text = str(
                item.get("name")
                or item.get("element_name")
                or item.get("id")
                or item.get("element_id")
                or ""
            ).strip()
        else:
            text = str(item).strip()
        if text:
            texts.append(text)
    return "、".join(texts) or "一个尚未识别的对象"


def _number_tuple(values) -> str:
    return "(" + ", ".join(_number(value) for value in values) + ")"


def _number(value) -> str:
    try:
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _strip_terminal_punctuation(text: str) -> str:
    return str(text or "").strip().rstrip("。！？；.!?; ")

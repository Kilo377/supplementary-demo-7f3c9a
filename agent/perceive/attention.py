from __future__ import annotations

from agent.belief.spatial_memory.spatial_belief import SpatialBelief

from .types import AttentionItem, AttentionResult, PerceiveResult


def build_attention(
    result: PerceiveResult,
    *,
    belief: SpatialBelief | None = None,
) -> AttentionResult:
    items: list[AttentionItem] = []
    items.extend(_environment_attention(result, belief=belief))
    items.extend(_physical_attention(result))
    items.extend(_psychological_attention(result))
    items.sort(key=lambda item: item.salience, reverse=True)
    return AttentionResult(items=items)


def _environment_attention(result: PerceiveResult, *, belief: SpatialBelief | None) -> list[AttentionItem]:
    items: list[AttentionItem] = []
    change_reason = _belief_change_reason(result, belief)
    for change in result.environment_input.changed_elements:
        items.append(
            AttentionItem(
                source="environment_change",
                text=f"{change.name}的{change.label}从{change.before_status}变成了{change.after_status}。",
                reason=change_reason,
                salience=0.9,
            )
        )

    feedback = result.environment_input.previous_world_feedback.strip()
    if feedback and _looks_like_problem_feedback(feedback):
        items.append(
            AttentionItem(
                source="world_feedback",
                text=feedback,
                reason="上一轮动作反馈包含失败、缺失、无法完成或解析异常。",
                salience=0.85,
            )
        )
    return items


def _belief_change_reason(result: PerceiveResult, belief: SpatialBelief | None) -> str:
    if belief is not None and belief.get_area(result.area_id) is not None:
        return "当前环境状态和 spatial belief 中的旧快照不一致。"
    return "当前环境状态出现变化。"


def _physical_attention(result: PerceiveResult) -> list[AttentionItem]:
    physical = result.self_input.physical
    items: list[AttentionItem] = []
    if physical.body_surface and physical.body_surface != "dry_clean":
        items.append(
            AttentionItem(
                source="body_surface",
                text=f"{result.agent_name}身体表面状态是{physical.body_surface}。",
                reason="身体表面状态不是默认的干净干燥状态。",
                salience=0.7,
            )
        )
    if physical.posture in {"lying", "crouching"}:
        items.append(
            AttentionItem(
                source="posture",
                text=f"{result.agent_name}当前姿态是{physical.posture}。",
                reason="当前姿态可能限制下一步动作。",
                salience=0.65,
            )
        )
    return items


def _psychological_attention(result: PerceiveResult) -> list[AttentionItem]:
    psychological = result.self_input.psychological
    items: list[AttentionItem] = []
    for feeling in psychological.desire_feelings:
        if feeling.intensity == "strong":
            items.append(
                AttentionItem(
                    source=feeling.source,
                    text=feeling.text,
                    reason="这个内在 feeling 足够强，可能影响行动选择。",
                    salience=_psychological_salience(feeling.source),
                )
            )
    return items


def _looks_like_problem_feedback(text: str) -> bool:
    lowered = text.lower()
    problem_markers = [
        "无法",
        "不能",
        "没有",
        "失败",
        "错误",
        "异常",
        "不完整",
        "无法完成",
        "无法判断",
        "无法解析",
        "failed",
        "failure",
        "error",
        "unsupported",
        "rejected",
        "not found",
        "cannot",
        "can't",
        "could not",
        "incomplete",
        "parse",
    ]
    return any(marker in lowered or marker in text for marker in problem_markers)


def _psychological_salience(source: str) -> float:
    if source.startswith(("physiological_state:", "internal_state:")):
        return 0.82
    if source == "mental":
        return 0.72
    return 0.65


def build_attention_todo(result: PerceiveResult) -> list:
    return build_attention(result).items

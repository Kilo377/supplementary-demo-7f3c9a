from __future__ import annotations

from .types import PreparedHabitualResponse


def build_goal_conflict_prompt(
    *,
    intent_text: str,
    habitual_response: PreparedHabitualResponse,
) -> str:
    return f"""
心理学上，习惯性行为受到特定情境触发，但并不需要一个目标来驱动。这个习惯性行为可能正好符合一个人当前的目标，可能与目标无关，也可能与目标冲突。

有一个人当前的目标是：
{intent_text or "没有明确的想法。"}

这个人受到了特定情境的触发，准备产生的习惯性行为是：
{habitual_response.response_text}

请判断这个习惯性行为是否与这个人的目标冲突。
与目标相符合，或者与目标无关，都不算冲突。
这里只判断是否冲突，不判断这个人最终应该选择哪个行为。

只输出 JSON：
{{
  "conflict": true,
  "reason": "第三人称的简短判断理由"
}}
""".strip()

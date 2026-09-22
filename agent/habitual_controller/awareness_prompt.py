from __future__ import annotations

from .types import PreparedHabitualResponse


def build_habitual_awareness_prompt(
    *,
    trigger_context_text: str,
    current_state_text: str,
    habitual_response: PreparedHabitualResponse,
) -> str:
    return f"""
有一个人受到当前情境的触发，准备产生一个习惯性行为。

触发这个行为的情境是：
{trigger_context_text or "没有提供明确的情境线索。"}

这个人的当前状态是：
{current_state_text or "没有提供特别的身体或情绪状态。"}

准备产生的习惯性行为是：
{habitual_response.response_text}

请判断在这个习惯性行为开始发生之前，这个人是否通常会注意到自己产生了这种行为倾向。

这里的“有意识”是指，这个人会形成可以被自己察觉的想法、冲动或行动意图。
这里的“无意识”是指，行为主要由情境、情绪或身体状态直接触发，在发生前没有形成清楚、可察觉的行动意图。

不要根据这个行为是否合理、是否符合目标来判断。
不要判断这个人最终是否应该执行该行为。
不要仅仅因为压力、紧张或焦虑程度较高就判定为有意识。有意识的习惯是否经过目标导向评估，由后续独立控制门控处理。

只输出 JSON：
{{
  "conscious": true,
  "reason": "第三人称的简短判断理由"
}}
""".strip()

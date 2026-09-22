from __future__ import annotations


def build_cumulative_reward_prompt(
    *,
    agent_name: str,
    intent_text: str,
    final_intent_status: str,
    actual_trajectory_text: str,
    final_state_text: str,
) -> str:
    return f"""
你正在评估一次已经结束的真实行为模拟，不是在预测未来。

这个人叫 {agent_name}。

这次模拟中的 Intent 是：
{intent_text}

模拟结束时记录的 Intent 状态是：
{final_intent_status or "active"}

实际发生的完整执行过程是：
{actual_trajectory_text or "没有实际执行动作。"}

模拟结束后的真实状态是：
{final_state_text or "没有提供最终状态。"}

请只判断模拟结束时，这个 Intent 在实践中得到了多大程度的满足。

使用0到10的连续量表：
- 0：完全没有满足，或者结果与 Intent 明显相反。
- 2：只完成了很小的前置动作，几乎没有形成实际结果。
- 5：完成了重要的一部分，但 Intent 仍明显没有完成。
- 8：已经基本满足，只有不影响实际结果的小部分没有完成。
- 10：Intent 已经在实践中充分完成。

要求：
- 只能依据实际执行结果和最终真实状态，不能假设没有发生的后续行动。
- Intent状态可以作为证据，但不能代替对实际过程的判断。
- 不要因为执行步数多或少而改变满足度；执行效率由程序另行计算。
- 不要评价人格、习惯强度、动作风格或World Model预测是否准确。
- 不要自己计算 cumulative reward。

只返回 JSON：
{{
  "intent_satisfaction": 0,
  "reason": "对最终满足度的简短客观说明"
}}
""".strip()

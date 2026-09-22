from __future__ import annotations


def build_transition_intent_satisfaction_prompt(
    *,
    agent_name: str,
    intent_text: str,
    action_text: str,
    world_feedback: str,
    next_state_text: str,
) -> str:
    name = str(agent_name or "Agent").strip()
    sections = [
        f"{name} 当前想要：\n\n{str(intent_text or '').strip()}",
        f"{name} 刚刚尝试：\n\n{str(action_text or '').strip()}",
        f"World 实际反馈：\n\n{str(world_feedback or '').strip()}",
    ]
    state = str(next_state_text or "").strip()
    if state:
        sections.append(f"动作后的可观察状态：\n\n{state}")
    sections.append(
        """
判断这个动作后的状态，是否已经在实践上满足或近似满足当前 Intent。

判断可以宽松：基本达到目的即可，不要求穷举所有细节。动作只是推进了一步、仍然需要明确后续步骤时，应为 not_satisfied。动作失败时也应为 not_satisfied。

请严格输出 JSON：

{
  "status": "satisfied | approximately_satisfied | not_satisfied",
  "reason": "判断依据"
}
""".strip()
    )
    return "\n\n".join(sections)

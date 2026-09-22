from __future__ import annotations


def build_reward_evaluation_prompt(
    *,
    agent_name: str,
    intent_text: str,
    current_state_text: str,
    action_chains_text: str,
    personality_text: str = "",
) -> str:
    name = str(agent_name or "Agent").strip()
    sections = [
        f"你是 {name}。",
        f"你当前想要：\n{intent_text.strip()}",
    ]
    _append_section(sections, "你觉得自己是这样的人：", personality_text)
    _append_section(sections, "你现在的状态是：", current_state_text)
    sections.append(
        "下面是从同一个当前状态出发，对几个初始动作分别进行前向推演后得到的动作链：\n\n"
        + action_chains_text.strip()
    )
    sections.append(
        f"""
请一次性比较这些动作链，并判断每个初始动作对于 {name} 当前 Intent 的整体价值。

被评分的是每条链的初始动作。后续动作和预测状态，是这个初始动作可能带来的后果。

请把目标满足、目标推进、最终状态、身体和心理变化、耗时、努力、失败风险、不确定性、信息价值以及是否符合 {name} 的实际偏好混合在一起，直接形成一个0到10的综合分数。不要输出任何子项分数。

评分锚点：
- 0：明显失败、严重偏离 Intent，或者产生显著负面结果。
- 2：几乎没有推进，成本、风险或不确定性明显。
- 5：有所推进，但距离目标仍远，或者代价和不确定性较大。
- 8：已经基本满足目标，过程合理，代价可以接受。
- 10：充分满足目标，而且结果、过程和人物状态都非常理想。

要求：
- 所有动作必须使用同一套尺度，在一次比较中完成评分。
- 分数是绝对价值，不是排名。即使某个动作是候选中最好的，如果它本身很差，也应该得到低分。
- 不必强行拉开分数，多个动作可以同分。
- 不要因为某条动作链更长、文字更多或者描述更详细而给更高分。
- 如果推演因为达到迭代上限而结束，应根据已经实现的推进程度评分，不能假设后续必然成功。
- 如果推演依赖未经确认的物体或状态，应把这种不确定性计入总分。
- 每个 action_id 必须且只能输出一次，不要添加新的动作。

只返回 JSON，不要解释。

JSON schema:
{{
  "evaluations": [
    {{
      "action_id": "动作链编号",
      "reward": 0,
      "reason": "形成这个综合价值分数的简短理由"
    }}
  ]
}}
""".strip()
    )
    return "\n\n".join(section for section in sections if section.strip()).strip()


def _append_section(sections: list[str], heading: str, content: str) -> None:
    cleaned = str(content or "").strip()
    if cleaned:
        sections.append(f"{heading}\n\n{cleaned}")

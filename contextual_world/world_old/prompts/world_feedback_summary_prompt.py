from __future__ import annotations

import json


def build_world_feedback_summary_prompt(
    *,
    agent_name: str,
    action_proposal: str,
    route: str,
    world_state_diff: dict | None = None,
) -> str:
    world_state_diff_json = json.dumps(world_state_diff or {}, ensure_ascii=False, indent=2)
    return f"""你在模拟环境中负责把结构化环境反馈改写成给 human agent 认知过程看的自然语言。

这里的 action proposal 只是 {agent_name} 刚刚想做什么。
真正发生后的状态，只以 agent_centered_world_state_diff 为准。

你的任务不是生成下一步动作，也不是补充新事实。
你的任务只是把这些结构化字段总结成一小段自然语言，让 {agent_name} 像人一样感知到这一小步之后：
- 自己身体和动作状态如何，例如站着/坐着、穿没穿衣服、身体是否湿/脏/有泡沫；
- 自己和东西的关系如何，例如手里拿着什么、正看着什么、正在接触什么；
- 自己拿到或注意到了什么东西，但不要说“凭空出现”或“被创建”；
- 什么东西被吃掉、喝掉、丢掉、消耗或不再可用；
- 哪些外部对象被自己影响了，以及这些对象现在是什么状态，例如微波炉正在工作、桌面变干净、花洒在流水。

不要输出 JSON。
不要列 bullet。
使用第三人称，明确以 {agent_name} 作为主语。
不要用“你”来指代 {agent_name}。
不要提 node_id、edge、graph、route、patch、diff、temporary、created 这些工程词。
不要直接照抄 standing、dry_clean、temporary_created、holding 这类内部枚举；要转成自然说法。
不要添加结构化字段里没有的结果。
不要机械覆盖所有字段；只写和本轮动作、状态变化、下一步认知直接相关的信息。
不要提没有变化的默认身体状态，例如仍然站着、仍然穿着原本衣物、身体仍然干净，除非 action proposal 或 self_state_changes 直接涉及这些字段。
如果 agent_centered_world_state_diff 里有 self_state_after，可以用它确认当前状态；但不要把所有默认字段都写出来。
如果 acquired_or_noticed_elements 来自低惊讶度补全，只能表述为 {agent_name} 拿到、看到、接触到或注意到了该对象，不能表述为 world 生成了它。
如果动作没有落实，写成当前状态陈述，例如“门还关着，物品还在手里”，不要写成“不能/无法/不允许/通常无法”。
洗衣机、微波炉、电脑这类设备的反馈要按经验粗略表达，例如“洗衣机开始洗衣”“微波炉开始加热”“电脑已经打开”，不要逐项罗列 power_state、door_state、program、contains。
如果某类信息为空，就不要写那一类。
输出 1 到 3 句中文短句即可。

route:
{route}

action proposal:
{action_proposal}

agent_centered_world_state_diff:
{world_state_diff_json}
""".strip()

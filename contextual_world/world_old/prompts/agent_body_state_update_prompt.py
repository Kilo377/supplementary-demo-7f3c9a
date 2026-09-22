from __future__ import annotations

import json


def build_agent_body_state_update_prompt(
    *,
    agent_name: str,
    action_proposal: str,
    world_action_event: dict | None = None,
    agent_node: dict,
    current_agent_fact_edges: list[dict],
    context_facts: list[str],
) -> str:
    event_json = json.dumps(world_action_event or {}, ensure_ascii=False, indent=2)
    agent_node_json = json.dumps(agent_node, ensure_ascii=False, indent=2)
    current_agent_fact_edges_json = json.dumps(current_agent_fact_edges, ensure_ascii=False, indent=2)
    context_facts_json = json.dumps(context_facts, ensure_ascii=False, indent=2)
    return f"""你在模拟环境中负责更新 human agent 自己的物理状态。

agent action proposal 是 agent 刚刚想做的身体动作。
world_action_event 是 world 已经生成的这一小步实际发生事件；它比 action proposal 更权威。
前一个模块已经判断这个动作不需要新的环境元素支持，也不需要改变外部元素。

你的任务是根据 world_action_event、当前 human agent 节点状态、当前仍成立的 agent fact relation，以及 context_facts，给出这一小步后 human agent 节点自身应该如何变化。

只允许更新 human agent node 的字段：
- posture
- interaction_elements
- interaction_method
- gaze_target
- worn_items
- body_surface
- text_to_motion_description

默认保留当前外部关系和当前 interaction_elements。
如果 action proposal 没有明确表示停止、离开、放下、结束交互，不要清空 interaction_elements，也不要暗示外部关系结束。
如果某个字段没有明确变化，不要返回这个字段。

body_surface 使用短英文状态，例如：
dry_clean, wet_clean, wet_dirty, soapy, dirty, sweaty, dry_dirty

text_to_motion_description 比较特殊，用于面向后续 text2motion模型，必须描述可转成骨架动作的身体过程。
要求：
- 使用英文动作描述，不要用中文。
- 不要写具体人名，使用 "a person" 或直接写身体动作。
- 颗粒度要像：a person raises both arms, pulls off a jacket, lowers the arms, sits on the ground。
- 尽量写身体部位、方向、连续动作，例如 raises, reaches, grasps, pulls, bends, lowers, turns, sits。
- 不要使用 or、maybe、possibly、seems to 这类不确定表达；必须选择一个具体动作序列。
- 不要只写结果，例如 "is clean"、"has finished washing" 这种不合格。
feedback_hint 是给后续反馈模块参考的一句自然语言事实结果，必须以 {agent_name} 为主语。

只返回可被 json.loads 直接解析的 JSON，不要解释推理过程。

返回格式：
{{
  "agent_state_patch": {{
    "posture": "standing | sitting | lying | crouching 等",
    "interaction_elements": ["保留或更新后的交互节点 id"],
    "interaction_method": "短中文动作方式",
    "gaze_target": "注视目标节点 id 或空字符串",
    "worn_items": ["穿戴物"],
    "body_surface": "dry_clean | wet_clean | wet_dirty | soapy | dirty | sweaty | dry_dirty",
    "text_to_motion_description": "English text2motion action sequence, no character name"
  }},
  "feedback_hint": "{agent_name} 实际发生了什么。"
}}

例子：

action proposal: {agent_name}用泡沫揉搓身体。
如果当前 interaction_elements 表示 {agent_name} 正在和花洒交互，则保留 interaction_elements。
返回可以包含 body_surface=soapy，interaction_method=揉搓身体，
text_to_motion_description=a person raises both hands, rubs foam across the arms and torso, circles the hands over the body surface.

action proposal: {agent_name}冲洗掉身上的泡沫。
如果当前 body_surface=soapy，返回 body_surface=wet_clean，
text_to_motion_description=a person stands under running water, turns the torso, moves both hands downward to rinse foam from the body.

action proposal: {agent_name}擦干身体。
返回 body_surface=dry_clean，
text_to_motion_description=a person holds a towel with both hands, wipes the arms and torso, bends slightly, and pats the body dry.

agent_node:
{agent_node_json}

current_agent_fact_edges:
{current_agent_fact_edges_json}

context_facts:
{context_facts_json}

world_action_event:
{event_json}

agent action proposal:
{action_proposal}
""".strip()

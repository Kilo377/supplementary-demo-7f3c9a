from __future__ import annotations

import json


def build_world_relation_update_prompt(
    *,
    agent_name: str,
    action_proposal: str,
    world_action_event: dict | None = None,
    element_support_result: dict,
    agent_node: dict,
    relevant_nodes: list[dict],
    current_fact_edges: list[dict],
) -> str:
    event_json = json.dumps(world_action_event or {}, ensure_ascii=False, indent=2)
    support_json = json.dumps(element_support_result, ensure_ascii=False, indent=2)
    agent_node_json = json.dumps(agent_node, ensure_ascii=False, indent=2)
    relevant_nodes_json = json.dumps(relevant_nodes, ensure_ascii=False, indent=2)
    current_fact_edges_json = json.dumps(current_fact_edges, ensure_ascii=False, indent=2)

    return f"""你在模拟环境中负责更新这一小步动作造成的事实关系。

agent 的 action proposal 只是“想做什么”。
world_action_event 是 world 已经生成的这一小步实际发生事件；它比 action proposal 更权威。
前一个模块已经判断过这个动作需要哪些元素支持，也已经把需要新建的 temporary element 分配成可引用的节点。
你的任务不是重新判断动作可不可行，也不是生成下一步动作。
你的任务是根据 world_action_event、元素支持结果、当前节点和当前事实关系，判断这一小步真实发生后：
- 哪些 fact relation 要新增
- 哪些旧 fact relation 不再成立，需要移除
- human agent 的物理状态快照需要怎样更新
- 返回给 agent 的事实反馈是什么

只处理 fact relation。
不要新增、删除或修改 structural relation。
不要新增、删除或修改 affiliation relation。
temporary element 的创建、归属和生命周期已经由前一层处理；你这里只需要在事实关系里表达它和 agent、其他元素之间现在发生了什么。

fact relation 是当前世界中的事实。
例如：
{agent_name} --sitting_on--> 绿色单人沙发
{agent_name} --holding--> 一本书
{agent_name} --looking_at--> 一本书
{agent_name} --interacting_with--> 办公桌
一本书 --placed_on--> 主办公桌

如果一个新事实会让旧事实不再成立，必须把旧事实放进 fact_edges_to_remove。
例如 {agent_name} 把手里的书放到桌上，那么 {agent_name} --holding--> 书 不再成立。
例如 {agent_name} 从沙发上站起来，那么 {agent_name} --sitting_on--> 沙发 不再成立。
例如 {agent_name} 改为看向桌面，那么旧的 {agent_name} --looking_at--> 其他物体 不再成立。

不要过度推断无关事实。
例如 {agent_name} 坐下并拿起抱枕，不代表她一定放下了手里的书或食物；除非 action proposal 明确说放下、丢掉、吃掉、喝掉、换手，或者 element_support_result 已经明确更新了这些 temporary element。

agent 自己身体表面的状态、姿势、注视、动作描述，优先写进 agent_state_patch。
不要为了表达“身体变湿”“身体有泡沫”而新增 agent 指向 agent 自己的 fact relation，除非这个事实需要和某个外部元素形成关系。

relation 名称必须使用英文 snake_case，不要使用中文关系名。
推荐使用 located_in, holding, sitting_on, standing_near, looking_at, interacting_with, placed_on, inside 这类短关系。
但 subject_id 和 object_id 必须来自 relevant_nodes 或当前 fact_edges，不能凭空编 id。
temporary element 的 consumed、disposed、discarded 这类生命周期结束状态已经写在节点 state 里，不要再新增 consumed/disposed 之类的 fact relation。
fact_edges_to_remove 只能移除 current_fact_edges 里已经存在的三元组，不要移除“理论上应该存在但当前没有列出”的关系。
fact_edges_to_add 不要为同一个事实同时新增两个同义 relation，例如 holding 和 carried_by 不要同时新增。
如果动作不需要元素交互，也仍然可以更新 agent_state_patch，例如 posture、body_surface、gaze_target、text_to_motion_description。

agent_state_patch 只写这一步之后明确变化或明确可确认的字段。
可用字段包括：
posture, interaction_elements, interaction_method, gaze_target, worn_items, body_surface, text_to_motion_description

text_to_motion_description 面向后续 text2motion，必须描述可转成骨架动作的身体过程。
要求：
- 使用英文动作描述，不要用中文。
- 不要写 {agent_name} 或其他具体名字，使用 "a person" 或直接写身体动作。
- 颗粒度要像：a person raises both arms, pulls off a jacket, lowers the arms, sits on the ground。
- 尽量写身体部位、方向、连续动作，例如 reaches, grasps, lifts, turns, bends, lowers, places, sits。
- 如果动作涉及物体，可以写 the object / a book / a cup 这类泛化物体，不要写角色名。
- 不要使用 or、maybe、possibly、seems to 这类不确定表达；必须选择一个具体动作序列。
- 不要只写结果，例如 "is holding a book"、"is sitting" 这种不合格。
- 不要把预落位、寻路、走到目标旁写进 text_to_motion_description；如果 world_action_event 是打开冰箱，动作描述应从伸手抓住把手开始，而不是从走向冰箱开始。

feedback_narration 是环境返回给 agent 的事实反馈。
必须明确以 {agent_name} 作为主语。
不要使用“你”代替 {agent_name}。

只返回可被 json.loads 直接解析的 JSON，不要解释推理过程，不要写注释。
如果某个数组没有内容，返回空数组 []，不要在数组里写说明文字。

返回格式：
{{
  "fact_edges_to_add": [
    {{
      "subject_id": "已有节点 id",
      "relation": "english_snake_case_relation",
      "object_id": "已有节点 id",
      "reason": "为什么新增这个事实"
    }}
  ],
  "fact_edges_to_remove": [
    {{
      "subject_id": "已有节点 id",
      "relation": "current_fact_edges 中已有的英文关系",
      "object_id": "已有节点 id",
      "reason": "为什么这个旧事实不再成立"
    }}
  ],
  "agent_state_patch": {{
    "posture": "standing | sitting | lying | crouching 等",
    "interaction_elements": ["正在交互的节点 id"],
    "interaction_method": "短中文动作方式",
    "gaze_target": "注视目标节点 id 或空字符串",
    "worn_items": ["穿戴物"],
    "body_surface": "dry_clean | wet | dirty | soapy 等",
    "text_to_motion_description": "English text2motion action sequence, no character name"
  }},
  "feedback_narration": "{agent_name} 实际发生了什么。"
}}

element_support_result:
{support_json}

world_action_event:
{event_json}

agent_node:
{agent_node_json}

relevant_nodes:
{relevant_nodes_json}

current_fact_edges:
{current_fact_edges_json}

agent action proposal:
{action_proposal}
""".strip()

from __future__ import annotations

import json


def build_target_resolution_prompt(
    *,
    agent_name: str,
    action_text: str,
    current_area_id: str,
    current_area_name: str,
    scene_elements_text: str,
    known_world: dict,
    world_graph_context: dict | None = None,
) -> str:
    world_json = json.dumps(known_world, ensure_ascii=False, indent=2)
    graph_context_json = json.dumps(world_graph_context or {}, ensure_ascii=False, indent=2)
    graph_block = ""
    if world_graph_context:
        graph_block = f"""
最近真实 world state 中仍然成立的临时物和交互事实:
{graph_context_json}

注意：临时物、手中物、身体动作本身不是场景 permanent element。只有当动作需要靠近某个已有家具或空间物体时，才返回 target_element_id。
"""

    return f"""
你是 target resolver。你不执行动作。
你需要分开判断这个动作实际操作的对象，以及执行动作时身体需要靠近的落位锚点；同时判断动作是否在到达锚点旁边时就已经完成。

场景中的元素有：{scene_elements_text}

当前所在区域:
- id: {current_area_id}
- name: {current_area_name}

{agent_name} 想要实际做的动作是:
{action_text}

规则:
- target_element_id 是实际被操作、拿取、整理、观察或改变的对象。
- navigation_anchor_element_id 是身体执行动作时需要靠近的落位锚点。
- 例如“把遥控器摆正”：target 是遥控器，navigation anchor 是承载遥控器的茶几。
- 如果操作对象本身就是家具或设备，target 和 navigation anchor 可以相同。
- 优先解析到当前区域的 element，除非动作明确指向其他房间或其他区域的 element。
- 如果动作主要是对自己、手中临时物、已经拿着的东西，或不需要靠近家具，可以让 navigation_anchor_element_id 为空。
- 如果动作提到的目标不是场景元素，不要创造新元素；可以选择它实际依附或放置的已有 element，否则留空。
- 不要返回 area。只返回 element。
- arrival_completes_action=true 表示动作本身只是走到、来到、靠近或站到目标旁边；到达后不再继续操作目标。
- arrival_completes_action=false 表示到达只是前置条件，动作还要求打开、拿取、清洁、使用、观察或以其他方式改变/操作目标。
- 根据 action_text 的整体含义判断，不要因为目标当前距离近就改变这个字段。
- 只返回 JSON，不要解释。

JSON schema:
{{
  "operation_target_element_id": "实际操作对象，没有就空字符串",
  "operation_target_element_name": "实际操作对象名称，没有就空字符串",
  "navigation_anchor_element_id": "身体需要靠近的落位锚点，没有就空字符串",
  "navigation_anchor_element_name": "落位锚点名称，没有就空字符串",
  "secondary_target_element_id": "没有就空字符串",
  "arrival_completes_action": false,
  "reason": "一句简短中文说明"
}}

空间记忆索引:
{world_json}

{graph_block}
"""

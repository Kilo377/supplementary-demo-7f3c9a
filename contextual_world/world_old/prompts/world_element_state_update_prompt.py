from __future__ import annotations

import json


def build_world_element_state_update_prompt(
    *,
    agent_name: str,
    action_proposal: str,
    world_action_event: dict | None = None,
    element_support_result: dict,
    relation_update_result: dict,
    relevant_nodes: list[dict],
    element_recent_history: dict | None = None,
) -> str:
    event_json = json.dumps(world_action_event or {}, ensure_ascii=False, indent=2)
    support_json = json.dumps(element_support_result, ensure_ascii=False, indent=2)
    relation_json = json.dumps(relation_update_result, ensure_ascii=False, indent=2)
    nodes_json = json.dumps(relevant_nodes, ensure_ascii=False, indent=2)
    history_json = json.dumps(element_recent_history or {}, ensure_ascii=False, indent=2)
    return f"""你在模拟环境中负责更新 element 节点自己的状态。

agent 的 action proposal 是 {agent_name} 刚刚想做什么。
world_action_event 是 world 已经生成的这一小步实际发生事件；它比 action proposal 更权威。
前面的模块已经判断了这个动作需要哪些元素支持，并决定了 agent 和元素之间的事实关系如何变化。

你的任务不是判断动作是否可行，不是创建 temporary element，也不是更新 agent 自己的身体状态。
你的任务只是根据 world_action_event 判断这一小步真实发生后，相关 element node 自身的状态字段是否应该变化。

element_recent_history 是相关 element 最近几步发生过的局部历史。
更新状态时要参考当前 relevant_nodes 中的状态和 element_recent_history，不要把元素无原因地回退到更早阶段。
如果某个设备已经进入 running、washing、heating、filling、locked 等明确阶段，状态更新应保持阶段连续。
只有 world_action_event 明确给出暂停、停止、打开、取出、关闭等因果动作时，才改变该设备的关键状态或内容。
例如洗衣机已经启动运行后，后续含糊动作通常只保持 power_state 和 contains 原样；如果事件只是状态陈述，element_state_updates 可以为空。

element node 自身状态包括：
- physical_status：粗略物理状态，例如 regular、clean、dirty、moved、open、closed。
- evolution_status：随时间演化状态，例如 stable、running、heating、cooling。
- interaction_status：当前交互状态，例如 idle、in_use、inspected。
- state_details：元素自己的具体细节，例如 door_state=open、flow_state=on、surface_state=clean、power_state=running、contains=food。

只更新会影响后续行动判断的元素状态。
不要重复表达 agent 和元素之间的关系；holding、looking_at、sitting_on、placed_on 这类关系已经由 relation update 处理。
不要为了泡沫、气味、水汽、灰尘飞起、香味这类短暂效果创建或更新 element state；这类内容如果没有稳定影响，就不写。
如果 world_action_event 明确表示设备被打开、启动、运行、关闭、停止，就要更新该设备 element 的 state_details，例如 power_state=on/off/running，并在需要时更新 evolution_status。
设备内部状态只需要足够支持后续判断，不要过度拆分成多个微小程序状态；例如洗衣机可以粗略记录为 running / contains=dirty_clothes，微波炉可以粗略记录为 heating，电脑可以粗略记录为 power_state=on。
不要推进后续步骤，只处理 action proposal 这一小步直接造成的状态变化。

element_id 必须来自 relevant_nodes 中已有的 permanent_element 或 temporary_element。
优先更新 permanent_element 的状态，例如冰箱门、花洒水流、微波炉运行、桌面清洁程度。
temporary_element 的生命周期状态通常已经由 support 层处理；除非 temporary_element 自身有明确状态细节变化，否则不要重复更新它。

state_details 的 key 要短、稳定、可复用，例如 door_state、flow_state、power_state、surface_state、temperature、contains、recently_removed。
state_details 的 value 必须是短状态，不要写完整句子。
如果 element 已经有合适的 state_details key，优先复用它。
如果没有明确 element 自身状态变化，element_state_updates 返回空数组。

只返回 JSON，不要解释推理过程。

返回格式：
{{
  "element_state_updates": [
    {{
      "element_id": "已有 element node id",
      "physical_status": null 或 "短状态",
      "evolution_status": null 或 "短状态",
      "interaction_status": null 或 "短状态",
      "state_details": {{
        "短key": "短value"
      }}
    }}
  ]
}}

例子：

action proposal: {agent_name}打开冰箱。
返回：
{{
  "element_state_updates": [
    {{
      "element_id": "fridge_01",
      "physical_status": null,
      "evolution_status": null,
      "interaction_status": "in_use",
      "state_details": {{"door_state": "open"}}
    }}
  ]
}}

action proposal: {agent_name}从冰箱里拿出一瓶牛奶。
返回：
{{
  "element_state_updates": [
    {{
      "element_id": "fridge_01",
      "physical_status": null,
      "evolution_status": null,
      "interaction_status": "in_use",
      "state_details": {{"door_state": "open", "recently_removed": "milk"}}
    }}
  ]
}}

action proposal: {agent_name}擦拭办公桌。
返回：
{{
  "element_state_updates": [
    {{
      "element_id": "desk_01",
      "physical_status": null,
      "evolution_status": null,
      "interaction_status": "in_use",
      "state_details": {{"surface_state": "clean"}}
    }}
  ]
}}

action proposal: {agent_name}打开电脑。
返回：
{{
  "element_state_updates": [
    {{
      "element_id": "computer_01",
      "physical_status": null,
      "evolution_status": "running",
      "interaction_status": "in_use",
      "state_details": {{"power_state": "on"}}
    }}
  ]
}}

action proposal: {agent_name}关闭电脑。
返回：
{{
  "element_state_updates": [
    {{
      "element_id": "computer_01",
      "physical_status": null,
      "evolution_status": "stable",
      "interaction_status": "idle",
      "state_details": {{"power_state": "off"}}
    }}
  ]
}}

element_support_result:
{support_json}

world_action_event:
{event_json}

relation_update_result:
{relation_json}

relevant_nodes:
{nodes_json}

element_recent_history:
{history_json}

agent action proposal:
{action_proposal}
""".strip()

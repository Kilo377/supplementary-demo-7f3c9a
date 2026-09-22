from __future__ import annotations

import json


def build_world_action_event_prompt(
    *,
    agent_name: str,
    action_proposal: str,
    route: str,
    element_support_result: dict,
    agent_node: dict,
    relevant_nodes: list[dict],
    current_fact_edges: list[dict],
    element_recent_history: dict | None = None,
) -> str:
    support_json = json.dumps(element_support_result, ensure_ascii=False, indent=2)
    agent_node_json = json.dumps(agent_node, ensure_ascii=False, indent=2)
    relevant_nodes_json = json.dumps(relevant_nodes, ensure_ascii=False, indent=2)
    current_fact_edges_json = json.dumps(current_fact_edges, ensure_ascii=False, indent=2)
    history_json = json.dumps(element_recent_history or {}, ensure_ascii=False, indent=2)
    return f"""你是模拟环境 world 本身，负责根据当前 world state 和 agent 的 action proposal 生成这一小步实际发生的世界事件。

agent 的 action proposal 只是想做什么，不等于已经发生。
前一层已经给出 route 和 element_support_result，说明这个动作大致涉及哪些节点、是否需要 temporary element。

你的任务是在世界层面决定这一小步实际发生了什么。
你不直接生成 agent_state_patch、element_state_updates 或 fact_edges。
你只生成一个权威的 world_action_event，后续模块会基于这个 event 分别投影 agent 状态、element 状态和关系变化。

element_recent_history 是相关 element 最近几步发生过的局部历史，用来帮助判断当前动作是否顺着该元素已有状态推进。
如果某个元素已经进入明确阶段，例如 running、washing、heating、filling、locked，world_action_event 应保持这个阶段的连续性。
如果 action proposal 与当前状态没有形成完整的因果链，actual_event 写成这一刻真实的状态陈述。
例如洗衣机已经 close door、start washer、running 后，agent 想直接再放衣服进去时，actual_event 可以写成“洗衣机仍在运转，衣物保持在原来的位置”，accepted=false。
如果 action proposal 要把物品放入某个有门/盖的容器或设备，而 relevant_nodes 显示 door_state=closed，且 action proposal 没有明确打开门/盖，则 actual_event 写成“门仍关着，物品保持在原位置”，accepted=false。
如果 action proposal 要调整正在 running / washing / heating 的设备内部程序或内容，而 action proposal 没有明确暂停、停止或重新打开设备，则 actual_event 写成设备仍按当前状态运行，accepted=false。

要求：
- actual_event 必须描述这一小步真实发生的事件，以 {agent_name} 为主语。
- event_effects 写这一事件造成的直接事实后果，短句即可。
- involved_node_ids 只能使用 relevant_nodes、agent_node、current_fact_edges 中出现过的 node_id。
- 如果 element_support_result 里已经创建/引用了 temporary element，要把它纳入事件理解，但不要说“凭空创建”。
- 如果 action proposal 明确提到某个目标物，例如洗衣篮、微波炉、冰箱、书架，不要把目标物替换成另一个相近物。目标物不存在时，可以根据 element_support_result 使用 temporary element，或把事件写成当前状态没有改变。
- 如果 agent_node 显示 {agent_name} 已经在相关元素附近或同一区域，不要把“走到/来到/移动到目标旁”写进 actual_event。
- 导航或预落位是动作执行前的外部步骤，不属于这个 world_action_event；除非 action proposal 本身就是移动，否则 actual_event 只描述真正的交互或身体动作。
- 对普通、低惊讶度、元素支持充分的动作，默认让动作成功落实，不要凭空制造失败。
- 只有当前 world state 或 element_support_result 明确显示坏了、断电、缺少必要目标、不可达、被阻挡时，才把事件写成失败或无效。
- 例如“按下电脑电源按钮/打开电脑”在电脑存在且没有故障事实时，应产生“电脑被启动/电源打开”的直接后果，不要写“电脑仍处于关闭状态”。
- 不要推进下一步，只描述 action proposal 这一小步。
- 如果动作无法发生，accepted=false，并说明 reason；否则 accepted=true。
- accepted=false 时，actual_event 和 event_effects 仍然写世界状态陈述，不要写成规则宣告。
- 避免在 actual_event、event_effects、reason 里使用“不能”“无法”“不允许”“通常无法”这类裁判式说法；改写成“门仍关着”“设备仍在运行”“物品仍在手里”等状态陈述。
- estimated_duration 是这一小步动作的粗略持续时间，可以用 10s、20s、30s 或 1min、5min、10min。
- 不要输出 JSON 之外的解释。

返回格式：
{{
  "accepted": true 或 false,
  "route": "{route}",
  "actual_event": "{agent_name} 实际发生了什么。",
  "event_effects": [
    "直接事实后果"
  ],
  "involved_node_ids": [
    "已有 node id"
  ],
  "estimated_duration": "10s | 20s | 30s | 1min | 5min | 10min",
  "reason": "如果 accepted=false，说明为什么；否则可以为空"
}}

route:
{route}

element_support_result:
{support_json}

agent_node:
{agent_node_json}

relevant_nodes:
{relevant_nodes_json}

current_fact_edges:
{current_fact_edges_json}

element_recent_history:
{history_json}

agent action proposal:
{action_proposal}
""".strip()

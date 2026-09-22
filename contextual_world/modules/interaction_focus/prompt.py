from __future__ import annotations

import json

from .types import WorldInteractionFocusVariables


def build_world_interaction_focus_prompt(variables: WorldInteractionFocusVariables) -> str:
    return f"""你是 world_interaction_focus 模块。

我们在模拟一个人类 agent 在环境中的一小步动作。环境是一个图，human_agent、permanent_element、temporary_element 都是 node。

你的任务是根据 action proposal、节点目录和当前事实关系，选择这一步可能参与交互或可能发生变化的最小节点集合，并给出动作关系提示 interaction_frame。

请优先阅读 graph_state_text，它是工程层从节点目录和事实关系组装的人类可读输入。
如果需要核对 node_id、node_type 或已有 fact edge，可以参考 raw JSON。
如果自然语言和 raw JSON 冲突，以 raw JSON 为准。

你不判断动作是否成功，也不判断节点状态如何变化。
你只选择后续 state transition 需要查看和可能更新的节点。
不要推断门是否打开、食物是否吃完、电脑是否开机、身体是否变湿等状态变化。

focused_node_ids 必须包含 human_agent node。
focused_node_ids 可以包含多个节点，例如 agent 坐在沙发上抱抱枕时，应包含 agent、沙发、抱枕。
如果 action proposal 没有明说对象，但当前事实关系说明 agent 正 holding、sitting_on、looking_at 或 using 某个对象，也可以把这个对象选入 focused_node_ids。
interaction_frame 是动作意图的关系提示，不是已经成立的事实。真正的事实关系由后续 state transition 决定。
interaction_frame 中每一条关系只能有一个 subject_id 和一个 object_id，二者都必须是单个 node_id 字符串，不能写成数组。
如果一个动作涉及多个对象，请拆成多条 interaction_frame，不要把多个 node_id 塞进同一个字段。
relation_hint 也必须是一个英文 snake_case 字符串，不要写数组。
不要选择整张图，只选择这一步最可能相关的节点。

只返回 JSON，不要解释推理过程。

返回格式：
{{
  "focused_node_ids": ["已有 node_id"],
  "interaction_frame": [
    {{
      "subject_id": "已有 node_id",
      "relation_hint": "english_snake_case_hint",
      "object_id": "已有 node_id",
      "reason": "为什么这个动作关系和本轮动作相关"
    }}
  ],
  "focus_reason": "一句中文说明为什么选这些节点"
}}

action_proposal:
{variables.action_proposal}

support_result:
{json.dumps(variables.support_result, ensure_ascii=False, indent=2)}

graph_state_text:
{variables.graph_state_text}

node_reference:
{json.dumps(variables.node_reference, ensure_ascii=False, indent=2)}

raw_nodes:
{json.dumps(variables.nodes, ensure_ascii=False, indent=2)}

raw_fact_edges:
{json.dumps(variables.fact_edges, ensure_ascii=False, indent=2)}
""".strip()

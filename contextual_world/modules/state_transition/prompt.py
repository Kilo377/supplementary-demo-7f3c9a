from __future__ import annotations

import json

from .types import WorldStateTransitionVariables


def build_world_state_transition_prompt(variables: WorldStateTransitionVariables) -> str:
    return f"""你是 world_state_transition 模块。

我们在模拟一个人类 agent 在环境中的一小步动作。环境被表示成一个图：
- human_agent、permanent_element、temporary_element 都是 node。
- fact edge 表示当前成立的事实关系，例如 agent holding milk、book placed_on desk。
- node state 表示节点自己的状态，例如冰箱 door_state=open，agent posture=sitting，牛奶 status=held。

你收到的 action proposal 是 agent 想要做的一小步动作。
它不是已经发生的事实。
你的任务是根据当前子图状态，判断这一小步之后 world graph 应该如何从 st 变成 st+1。

非常重要：
你必须先决定 st+1 的状态变化，再根据状态变化总结动作结果。
不要先写自然语言结果再反推状态。
如果状态变化没有落实 action proposal 的核心目标，accepted=false。
如果状态变化落实了核心目标，accepted=true。

你会收到两类输入：

1. 工程层组装好的自然语言状态：
- subgraph_state_text
- recent_state_history_text
- transition_constraints_text
- support_context_facts

这些是你主要阅读的内容，已经把节点、状态、关系组装成人类可理解的形式。
support_context_facts 是上一模块判断出的非 node 背景，例如邮件、网页、办公软件、屏幕内容这类不应建模成独立 node 的内容。

2. 原始结构化数据：
- node_reference
- focused_nodes
- focused_fact_edges
- interaction_frame

这些用于核对 node_id、节点类型、已有事实关系和动作关系提示。
如果自然语言状态与原始数据冲突，以原始结构化数据为准。
interaction_frame 只是动作关系提示，不是当前已经成立的事实，也不是状态更新结果。

accepted=true 的含义：
next_state_delta 让 action proposal 的核心目标在 st+1 中成立。
例如打开冰箱时，冰箱 door_state 变成 open；吃掉食物时，食物 status 变成 consumed/visible=false，agent 不再 holding 食物。
如果当前 st 中食物/牛奶/酸奶是 held 或 visible=true，而 action proposal 是吃掉/喝掉它，那么 st+1 可以把它改成 consumed/visible=false，这表示这一小步成功完成。
这种情况下 actual_event 应该写 agent 吃掉/喝掉了它，不要写“发现它已经被吃完/喝完”。

accepted=false 的含义：
当前状态不支持核心目标落实。
你仍然可以更新 agent 的尝试动作、注视、姿态，但不要把外部对象改成完成状态。
只有当当前 st 已经明确显示目标对象已经 consumed、disposed、discarded、visible=false 或关键对象缺失时，才可以因为“已经结束/不存在”而 rejected。
accepted=false 时，不要把 temporary object 改成 consumed/disposed/discarded，不要把门改成 open，不要把电脑改成 logged_in，不要把任何外部对象改成 action proposal 的完成态。

node_updates:
- human_agent 可以更新 posture、interaction_elements、interaction_method、gaze_target、worn_items、body_surface、text_to_motion_description。
- permanent_element 可以更新 physical_status、evolution_status、interaction_status、state_details。
- temporary_element 可以更新 status、visible、lifecycle、temperature、cooking_state、anchor_element_id 等短状态。

不要把 fact relation 写进 node state。
例如 holding、inside、placed_on、sitting_on 要写到 fact_edges_to_add/remove，不要写进 state_patch。

text_to_motion_description 必须是英文身体动作描述，不要写人名，不要使用 or/maybe/possibly。
例如：a person reaches forward, grasps the refrigerator handle, pulls the door open, and looks inside.

estimated_duration 要根据完整 execution_result 估计：如果 actual_event 和 event_effects 在现实世界真实发生，大概需要多长时间。
这里估计的是 actual_event 所描述的整件事情，不是最后一次手部动作，也不是一次游戏动画的时长。
必须按现实生活标尺选择：
- 看一眼、转身、按一次按钮、拿起或放下一个已在手边的物品：10s 到 30s。
- 打开柜门、检查一处小区域、处理一个简单物品：1min 到 3min。
- 收拾一张小桌面、整理少量杂物、简单擦拭一处表面：3min 到 5min。
- 整理一件完整家具、床面、沙发、地毯、带抽屉的柜子，或对多件物品归类和清洁：10min 到 20min。
- 整理书架、衣柜、大量杂物，完整清洁一个局部区域：15min 到 30min。
- 打扫整个房间或完成多个连续家务：30min 到 60min。
- 等待：严格按照 action proposal 中要求等待的时长。
如果 actual_event 使用了“整理好、收拾干净、清理完成、归类完毕”等完成式表达，就必须计入完成该结果所需的全部现实时间，不能只估计最后一下动作。
只输出下面给出的粗略档位，不要输出小数秒。

只返回 JSON，不要解释推理过程。

返回格式：
{{
  "accepted": true,
  "next_state_delta": {{
    "node_updates": [
      {{
        "node_id": "已有 node_id",
        "node_type": "human_agent | permanent_element | temporary_element",
        "state_patch": {{}}
      }}
    ],
    "fact_edges_to_add": [
      {{
        "subject_id": "已有 node_id",
        "relation": "english_snake_case_relation",
        "object_id": "已有 node_id",
        "reason": "可选"
      }}
    ],
    "fact_edges_to_remove": [
      {{
        "subject_id": "已有 node_id",
        "relation": "focused_fact_edges 中已有 relation",
        "object_id": "已有 node_id",
        "reason": "可选"
      }}
    ],
    "state_reasoning_summary": "一句中文 debug 摘要，只说明为什么这些状态变化合理"
  }},
  "execution_result": {{
    "actual_event": "根据 st -> st+1，总结真实发生了什么",
    "event_effects": ["直接事实后果"],
    "estimated_duration": "10s | 20s | 30s | 1min | 3min | 5min | 10min | 15min | 20min | 30min | 45min | 60min",
    "rejection": {{
      "reason": "",
      "state_statement": ""
    }}
  }}
}}

agent_name:
{variables.agent_name}

action_proposal:
{variables.action_proposal}

subgraph_state_text:
{variables.subgraph_state_text}

recent_state_history_text:
{variables.recent_state_history_text}

transition_constraints_text:
{variables.transition_constraints_text}

support_context_facts:
{json.dumps(variables.support_context_facts, ensure_ascii=False, indent=2)}

node_reference:
{json.dumps(variables.node_reference, ensure_ascii=False, indent=2)}

focused_nodes:
{json.dumps(variables.focused_nodes, ensure_ascii=False, indent=2)}

focused_fact_edges:
{json.dumps(variables.focused_fact_edges, ensure_ascii=False, indent=2)}

interaction_frame:
{json.dumps(variables.interaction_frame, ensure_ascii=False, indent=2)}
""".strip()

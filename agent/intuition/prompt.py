from __future__ import annotations

import json

from agent.intent.state import IntentState
from agent.working_memory import WorkingMemoryFrame
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine


def build_intuition_prompt(
    *,
    agent_name: str,
    intent: IntentState,
    working_memory: WorkingMemoryFrame,
    engine: PhysicsEngine,
    action_sample_window: int = 1,
    include_think: bool = True,
) -> str:
    sample_window = min(12, max(1, int(action_sample_window)))
    areas = [
        {"area_id": area.node_id, "area_name": area.name}
        for area in engine.home.areas
    ]
    areas_json = json.dumps(areas, ensure_ascii=False, indent=2)
    route_options = "action | wait | walk | chat | think" if include_think else "action | wait | walk | chat"
    think_example = "\n- 储物间好像去不了，我得先想想不用那些工具怎么开始。" if include_think else ""
    think_route = "\n- think：遇到复杂、卡住、目标不清或需要先形成解决策略的问题，第一反应是慢下来想想。不要把普通犹豫当成 think。" if include_think else ""
    if sample_window == 1:
        output_block = f"""只返回 JSON，不要解释。

JSON schema:
{{
  "route": "{route_options}",
  "thought": "一句第一人称中文内心想法",
  "target_area_id": "walk 时填写，没有就空字符串",
  "target_area_name": "walk 时填写，没有就空字符串",
  "chat_target": "chat 时填写，没有就空字符串",
  "wait_duration": "wait 时填写：30s | 1min | 3min | 5min | 10min | 15min | 20min | 30min；其他路由为空字符串"
}}"""
    else:
        output_block = f"""请给出 {sample_window} 种可能的 Intuition reply。

只返回 JSON，不要解释。

JSON schema:
{{
  "replies": [
    {{
      "route": "{route_options}",
      "thought": "一句第一人称中文内心想法",
      "target_area_id": "walk 时填写，没有就空字符串",
      "target_area_name": "walk 时填写，没有就空字符串",
      "chat_target": "chat 时填写，没有就空字符串",
      "wait_duration": "wait 时填写：30s | 1min | 3min | 5min | 10min | 15min | 20min | 30min；其他路由为空字符串"
    }}
  ]
}}"""
    return f"""
你要模拟 {agent_name} 在读入 Working Memory 后，最先冒出来的直觉反应。

这不是完整计划，也不是事后反思。请把自己当成 {agent_name}，用第一人称想一句话。
这句话应该像内心反应，例如：
- 微波炉还没好，我最好等一下。
- 电脑坏了，我打算检查下看看怎么回事。
- 这里没什么想要的东西，我先回卧室看看。
- 我想和某个人聊聊。{think_example}

必须考虑当前 Intent，不要忘掉 {agent_name} 原本正在做什么。
如果环境没有强烈打断，就顺着当前 Intent 往前反应。
如果要行动，只说第一反应，不要展开多步计划。

route 只能是：
- action：想在当前位置或附近做一个具体动作；后面会交给 Action Proposal 细化。
- wait：第一反应是先等一下、观察一下、暂时不动。必须根据正在等待的事情填写 wait_duration，并在 thought 里自然说出准备等多久。
- walk：第一反应是去另一个房间或区域。必须尽量填写 target_area_id 或 target_area_name。
- chat：第一反应是和某个人说话。尽量填写 chat_target。{think_route}

{output_block}

当前 Intent:
{intent.intent_text} [{intent.status}]

已知房间:
{areas_json}

Working Memory:
{working_memory.format_for_action_prompt()}
""".strip()

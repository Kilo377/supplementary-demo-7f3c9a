from __future__ import annotations

import json

from agent.intent.state import IntentState
from agent.intuition import IntuitionResult
from agent.working_memory import WorkingMemoryFrame
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine

from .types import ThinkResult


def build_think_prompt(
    *,
    agent_name: str,
    intent: IntentState,
    working_memory: WorkingMemoryFrame,
    intuition: IntuitionResult,
) -> str:
    return f"""
你要模拟 {agent_name} 的慢思考过程。

这不是旁白，不是行动计划书，也不是对外解释。请把自己当成 {agent_name}，用第一人称想一小段。
必须根据 {agent_name} 的人物传记、欲望状态、身体状态、当前位置、当前 Intent、刚才的直觉和已经发生过的事来想。
不同的人思考方式应该不同：谨慎的人会更看重风险、稳定和后果，冲动的人会更快试错，疲惫的人会更保守。

思考内容应该帮助 {agent_name} 把当前问题想清楚一点，但不要直接写成多步行动清单。

只返回 JSON，不要解释。

JSON schema:
{{
  "thought": "第一人称中文内心思考",
  "conclusion": "一句话概括思考后的倾向"
}}

当前 Intent:
{intent.intent_text} [{intent.status}]

刚才的 Intuition:
[{intuition.route}] {intuition.thought}

Working Memory:
{working_memory.format_for_action_prompt()}
""".strip()


def build_post_think_route_prompt(
    *,
    agent_name: str,
    intent: IntentState,
    working_memory: WorkingMemoryFrame,
    think_result: ThinkResult,
    engine: PhysicsEngine,
) -> str:
    areas = [
        {"area_id": area.node_id, "area_name": area.name}
        for area in engine.home.areas
    ]
    areas_json = json.dumps(areas, ensure_ascii=False, indent=2)
    return f"""
{agent_name} 刚才仔细想了一下：
{think_result.format_for_prompt()}

请根据这个思考结果、当前 Intent 和 Working Memory，判断 {agent_name} 接下来应该进入哪条路由。
这仍然是第一人称内心反应，不是完整计划。

route 只能是：
- action：想在当前位置或附近做一个具体动作；后面会交给 Action Proposal 细化。
- wait：决定先等一下、观察一下、暂时不动。必须结合刚才的思考填写 wait_duration，并在 thought 里自然说出准备等多久；例如想休息十几分钟就填写 15min。
- walk：决定去另一个房间或区域。必须尽量填写 target_area_id 或 target_area_name。
- chat：决定和某个人说话。尽量填写 chat_target。

只返回 JSON，不要解释。

JSON schema:
{{
  "route": "action | wait | walk | chat",
  "thought": "一句第一人称中文内心想法",
  "target_area_id": "walk 时填写，没有就空字符串",
  "target_area_name": "walk 时填写，没有就空字符串",
  "chat_target": "chat 时填写，没有就空字符串",
  "wait_duration": "wait 时填写：30s | 1min | 3min | 5min | 10min | 15min | 20min | 30min；其他路由为空字符串"
}}

当前 Intent:
{intent.intent_text} [{intent.status}]

已知房间:
{areas_json}

Working Memory:
{working_memory.format_for_action_prompt()}
""".strip()

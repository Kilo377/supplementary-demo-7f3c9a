from __future__ import annotations

import json

from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine


def build_transition_iteration_intuition_prompt(
    *,
    agent_name: str,
    intent_text: str,
    transition_state_text: str,
    failure_context_text: str = "",
    action_sample_window: int = 1,
    engine: PhysicsEngine,
) -> str:
    areas = [
        {"area_id": area.node_id, "area_name": area.name}
        for area in engine.home.areas
    ]
    areas_json = json.dumps(areas, ensure_ascii=False, indent=2)
    sections = [
        f"你是 {agent_name}。",
        f"你当前想要：\n{intent_text}",
        (
            "你刚才在心里向前推演了一步。下面是根据你的经验，你认为自己下一刻会处于的状态。"
            "在这次继续推演中，把它当作你现在相信的状态：\n\n"
            f"{transition_state_text}"
        ),
    ]
    failure = str(failure_context_text or "").strip()
    if failure:
        sections.append(
            "刚才的推演暴露了一个需要处理的问题：\n\n" + failure
        )
    sample_count = max(1, min(12, int(action_sample_window)))
    if sample_count == 1:
        output_instruction = """
只返回 JSON，不要解释。

JSON schema:
{
  "route": "action | wait | walk | chat",
  "thought": "一句第一人称中文内心想法",
  "target_area_id": "walk 时填写，没有就空字符串",
  "target_area_name": "walk 时填写，没有就空字符串",
  "chat_target": "chat 时填写，没有就空字符串",
  "wait_duration": "wait 时填写：30s | 1min | 3min | 5min | 10min | 15min | 20min | 30min；其他路由为空字符串"
}
""".strip()
    else:
        output_instruction = f"""
给出 {sample_count} 种当前可能采取的不同后续动作。不同回复应当是有实际行为差异的方案，而不是同一句话的改写。

只返回 JSON，不要解释。

JSON schema:
{{
  "replies": [
    {{
      "route": "action | wait | walk | chat",
      "thought": "一句第一人称中文内心想法",
      "target_area_id": "walk 时填写，没有就空字符串",
      "target_area_name": "walk 时填写，没有就空字符串",
      "chat_target": "chat 时填写，没有就空字符串",
      "wait_duration": "wait 时填写；其他路由为空字符串"
    }}
  ]
}}
""".strip()
    sections.extend([
        f"你知道的房间是：\n{areas_json}",
        f"""
根据这个新的状态，{agent_name} 接下来打算怎么做，以努力满足你的目标？


route 只能是：
- action：做一个具体动作。
- wait：等待、观察或暂时不动。必须填写 wait_duration，并在 thought 中自然说出准备等多久。
- walk：去另一个房间或区域。必须尽量填写 target_area_id 或 target_area_name。
- chat：和某个人说话。尽量填写 chat_target。

{output_instruction}
""".strip(),
    ])
    return "\n\n".join(section for section in sections if section.strip()).strip()

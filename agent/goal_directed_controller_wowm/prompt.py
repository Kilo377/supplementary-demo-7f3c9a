from __future__ import annotations

from agent.action.prompt import (
    format_short_time_memory_for_action_prompt,
    format_spatial_belief_for_action_prompt,
)


def build_without_world_model_action_prompt(
    *,
    agent_name: str,
    intent,
    short_time_memory,
    spatial_belief,
    current_area_id: str,
    scene_name: str,
    action_sample_window: int = 1,
) -> str:
    sample_window = min(12, max(1, int(action_sample_window)))
    intent_text = str(getattr(intent, "intent_text", "") or "").strip()
    memory_text = format_short_time_memory_for_action_prompt(short_time_memory)
    spatial_text = format_spatial_belief_for_action_prompt(
        spatial_belief,
        current_area_id=current_area_id,
        scene_name=scene_name,
        agent_name=agent_name,
    )
    return f"""
你正在为 {agent_name} 生成紧接着要实际执行的动作。

【当前任务目标】
{intent_text or "（没有明确 Intent。）"}

【近期已经做过的事情】
{memory_text}

【当前空间认知】
{spatial_text}

【生成任务】
生成 {sample_window} 个 {agent_name} 下一刻可能实际执行的候选动作。
候选动作必须服务于当前任务目标，并按照“最符合目标且最适合现在执行”到“相对次优”的顺序排列。

【动作要求】
- 每个候选项只描述一个具体动作。
- 动作用 {agent_name} 开头。
- 这是实际要交给 world 执行的动作，不要解释理由。
- 优先选择环境中真实存在的家具或物品。
- 不要重复已经失败或没有产生实际进展的动作。
- 不要编造 Spatial Belief 里没有的房间、家具或工具。
- 不要输出思考过程或对后续步骤的规划。
- 如果下一步需要接触其他房间中的真实物品，直接描述与该物品交互的具体动作。

【输出格式】
只输出以下 JSON，不要输出 Markdown 或其他文字：
{{
  "actions": [
    {{"action": "{agent_name}执行的一个具体动作。"}}
  ]
}}

actions 必须正好包含 {sample_window} 项；每项只能描述一个动作。
""".strip()

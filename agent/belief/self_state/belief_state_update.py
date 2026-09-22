from __future__ import annotations

import json

from llm.api_manager import APIManager


def build_belief_state_update_prompt(
    *,
    agent_name: str,
    state_fields: dict,
) -> str:
    return f"""你在更新一个 human agent 的 self belief。

输入是 {agent_name} 刚刚一轮 action 结束后的四个字段：
- action_proposal: {agent_name} 刚刚想做什么
- movement: {agent_name} 从哪到哪；如果 moved=false，就是原地
- interacted_elements: 系统成功绑定到场景 schema 的交互元素
- environment_feedback: 环境反馈里实际发生了什么

你的任务：
根据这四个字段，归纳 {agent_name} 现在对自己状态的 belief。

要求：
- 1 句简短中文
- 不要提出下一步动作，只进行总结。
- 不要复述完整日志
- 如果 action_proposal 和 environment_feedback 不一致，以 environment_feedback 和 movement 为准
- environment_feedback 是 {agent_name} 形成 self belief 的主要依据
- interacted_elements 为空，只表示这次动作没有绑定到显式 world element；不代表动作没有发生，也不代表 {agent_name} 没有实际交互
- 如果 environment_feedback 描述了未显式建模的手中物、食物、包装、餐具或其他低惊讶度对象，就承认该动作实际发生，并把它写入 self belief
- 必须覆盖三点：刚刚想做什么；从哪到哪或是否原地；实际做了什么

四个字段：
{json.dumps(state_fields, ensure_ascii=False, indent=2)}

输出格式：
{agent_name} 刚刚想……；{agent_name}从……到……/{agent_name}留在原地；{agent_name}实际……。
"""


def update_belief_state(
    *,
    agent_name: str,
    state_fields: dict,
    provider_name: str = "ollama",
    model: str | None = None,
) -> str:
    try:
        api = APIManager(
            provider_name=provider_name,
            task_name="agent.self_belief_update",
        )
        raw = api.generate(
            build_belief_state_update_prompt(
                agent_name=agent_name,
                state_fields=state_fields,
            ),
            model=model,
        )
        return clean_belief_state_update(raw, agent_name=agent_name)
    except Exception:
        return fallback_belief_state_update(agent_name=agent_name, state_fields=state_fields)


def clean_belief_state_update(text: str, *, agent_name: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1].replace("json", "", 1).strip()
        else:
            cleaned = cleaned.replace("```", "").strip()
    lines = [line.strip(" -") for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    cleaned = " ".join(lines[:3]).strip()
    if agent_name not in cleaned:
        cleaned = f"{agent_name}相信{cleaned}"
    return cleaned


def fallback_belief_state_update(*, agent_name: str, state_fields: dict) -> str:
    proposal = state_fields.get("action_proposal", "") or "不明确的动作"
    feedback = state_fields.get("environment_feedback", "") or "环境没有给出明确反馈"
    movement = state_fields.get("movement", {}) or {}
    from_text = movement.get("from", "")
    to_text = movement.get("to", "")
    moved = movement.get("moved", False)
    if moved:
        movement_text = f"从{from_text}到了{to_text}"
    else:
        movement_text = f"留在{to_text or from_text or '原地'}"
    return f"{agent_name}刚刚想{proposal}；{agent_name}{movement_text}；{agent_name}实际经历的是：{feedback}"

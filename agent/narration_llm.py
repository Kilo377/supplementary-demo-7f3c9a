from __future__ import annotations

from agent.perceive import PerceiveResult
from llm.api_manager import APIManager


def build_perception_narration_prompt(result: PerceiveResult) -> str:
    changed_lines = []
    for item in result.changed_elements:
        changed_lines.append(
            f"- {item.name}: {item.before_status} -> {item.after_status}"
        )
    if not changed_lines:
        changed_lines.append("- 无明显变化")

    perceived_lines = [f"- {item.name}: {item.status}" for item in result.perceived_elements]
    if not perceived_lines:
        perceived_lines.append("- 无")

    fact_lines = [f"- {fact}" for fact in result.narration_facts]

    return f"""请根据以下感知结果，写一段1到3句的中文小说式旁白。

要求：
- 只使用给定事实，不要编造新信息
- 风格自然、克制、像轻小说旁白
- 如果环境基本正常，突出熟悉感与静态感
- 如果有变化，突出“与记忆不同”的感觉
- 不要输出项目符号，不要解释规则

Agent: {result.agent_name}
Area: {result.area_name}
Step: {result.step_id}
First visit in this run: {result.first_time_visit}

Changed elements:
{chr(10).join(changed_lines)}

Perceived elements in current area:
{chr(10).join(perceived_lines)}

Facts:
{chr(10).join(fact_lines)}

Agent self state:
{result.agent_state_text or "无特别身体状态"}
"""


def generate_narration_with_llm(
    result: PerceiveResult,
    *,
    provider_name: str = "ollama",
    model: str | None = None,
) -> str:
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.narration",
    )
    prompt = build_perception_narration_prompt(result)
    return api.generate(prompt, model=model).strip()

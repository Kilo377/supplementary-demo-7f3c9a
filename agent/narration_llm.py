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
        changed_lines.append("- No significant changes")

    perceived_lines = [f"- {item.name}: {item.status}" for item in result.perceived_elements]
    if not perceived_lines:
        perceived_lines.append("- None")

    fact_lines = [f"- {fact}" for fact in result.narration_facts]

    return f"""Based on the following perception results, write a 1-to-3 sentence Chinese novelistic narration.

Requirements:
- Use only the given facts; do not fabricate new information.
- Style should be natural, restrained, like light novel narration.
- If the environment is basically normal, emphasize familiarity and stillness.
- If there are changes, emphasize the feeling of "being different from memory."
- Do not output bullet points or explain rules.

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
{result.agent_state_text or "No special physical state"}
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

from __future__ import annotations

from .types import PreparedHabitualResponse


def build_goal_conflict_prompt(
    *,
    intent_text: str,
    habitual_response: PreparedHabitualResponse,
) -> str:
    return f"""
In psychology, habitual behaviors are triggered by specific situations but do not require a goal to drive them. This habitual behavior may coincidentally align with a person's current goal, be unrelated to the goal, or conflict with the goal.

The person's current goal is:
{intent_text or "No clear idea."}

This person has been triggered by a specific situation and is about to generate the following habitual behavior:
{habitual_response.response_text}

Please judge whether this habitual behavior conflicts with this person's goal.
Alignment with the goal or irrelevance to the goal does not count as conflict.
Here, only judge whether there is a conflict; do not judge which behavior the person should ultimately choose.

Output only JSON:
{{
  "conflict": true,
  "reason": "A brief third-person judgment reason"
}}
""".strip()

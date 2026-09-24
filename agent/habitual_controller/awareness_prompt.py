from __future__ import annotations

from .types import PreparedHabitualResponse


def build_habitual_awareness_prompt(
    *,
    trigger_context_text: str,
    current_state_text: str,
    habitual_response: PreparedHabitualResponse,
) -> str:
    return f"""
A person is triggered by the current situation and is about to generate a habitual behavior.

The situation triggering this behavior is:
{trigger_context_text or "No clear situational cues provided."}

The person's current state is:
{current_state_text or "No special physical or emotional state provided."}

The habitual behavior about to be generated is:
{habitual_response.response_text}

Please judge whether this person would typically notice that they are developing a tendency toward this behavior before the habitual behavior begins.

"Conscious" here means the person forms thoughts, impulses, or action intentions that can be perceived by themselves.
"Unconscious" here means the behavior is primarily triggered directly by the situation, emotion, or physical state, without forming a clear, perceivable intention to act beforehand.

Do not judge based on whether the behavior is reasonable or aligns with goals.
Do not judge whether the person should ultimately execute this behavior.
Do not judge it as conscious solely due to high stress, tension, or anxiety. Whether conscious habits undergo goal-directed evaluation is handled by subsequent independent control gating.

Output only JSON:
{{
  "conscious": true,
  "reason": "A brief third-person judgment reason"
}}
""".strip()

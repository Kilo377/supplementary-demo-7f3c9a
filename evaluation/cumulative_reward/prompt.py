from __future__ import annotations


def build_cumulative_reward_prompt(
    *,
    agent_name: str,
    intent_text: str,
    final_intent_status: str,
    actual_trajectory_text: str,
    final_state_text: str,
) -> str:
    return f"""
You are evaluating a real behavior simulation that has already ended, not predicting the future.

This person is named {agent_name}.

The Intent in this simulation is:
{intent_text}

The Intent status recorded at the end of the simulation is:
{final_intent_status or "active"}

The complete execution process that actually occurred is:
{actual_trajectory_text or "No actual execution actions."}

The true state after the simulation ended is:
{final_state_text or "Final state not provided."}

Please only judge to what extent this Intent was practically satisfied at the end of the simulation.

Use a continuous scale from 0 to 10:
- 0: Completely unsatisfied, or the result is clearly opposite to the Intent.
- 2: Only very small prerequisite actions were completed, with almost no actual results formed.
- 5: An important part was completed, but the Intent is still clearly not finished.
- 8: Basically satisfied, only a small part that does not affect the actual result is unfinished.
- 10: The Intent has been fully completed in practice.

Requirements:
- Only base judgments on actual execution results and final true states; do not assume subsequent actions that did not occur.
- Intent status can serve as evidence but cannot replace judgment of the actual process.
- Do not change satisfaction based on the number of execution steps; execution efficiency is calculated separately by the program.
- Do not evaluate personality, habit strength, action style, or whether the World Model prediction was accurate.
- Do not calculate cumulative reward yourself.

Only return JSON:
{{
  "intent_satisfaction": 0,
  "reason": "Brief objective explanation of final satisfaction"
}}
""".strip()

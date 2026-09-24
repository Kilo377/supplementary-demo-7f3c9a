from __future__ import annotations


def build_transition_intent_satisfaction_prompt(
    *,
    agent_name: str,
    intent_text: str,
    action_text: str,
    world_feedback: str,
    next_state_text: str,
) -> str:
    name = str(agent_name or "Agent").strip()
    sections = [
        f"{name} currently wants:\n\n{str(intent_text or '').strip()}",
        f"{name} just tried:\n\n{str(action_text or '').strip()}",
        f"World's actual feedback:\n\n{str(world_feedback or '').strip()}",
    ]
    state = str(next_state_text or "").strip()
    if state:
        sections.append(f"Observable state after the action:\n\n{state}")
    sections.append(
        """
Determine whether the state after this action practically satisfies or approximately satisfies the current Intent.

The judgment can be lenient: as long as the main goal is achieved, there is no need to enumerate every detail. If the action only advances things by one step and subsequent steps are still needed, it should be 'not_satisfied'. It should also be 'not_satisfied' if the action fails.

Please output strictly in JSON format:

{
  "status": "satisfied | approximately_satisfied | not_satisfied",
  "reason": "Basis for judgment"
}
""".strip()
    )
    return "\n\n".join(sections)

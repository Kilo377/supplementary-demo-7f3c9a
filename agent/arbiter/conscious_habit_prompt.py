from __future__ import annotations

from agent.habitual_controller import PreparedHabitualResponse


def build_conscious_habit_arbiter_prompt(
    *,
    agent_name: str,
    personality: str,
    trigger_context_text: str,
    habitual_response: PreparedHabitualResponse,
    intent_text: str,
    intuition_text: str,
    think_text: str,
    goal_directed_action: str,
    current_state_text: str,
) -> str:
    think_section = f"\nI thought about it again carefully:\n{think_text}\n" if think_text.strip() else ""
    trigger = _first_person(trigger_context_text, agent_name)
    habitual = _first_person(habitual_response.response_text, agent_name)
    intent = _first_person(intent_text, agent_name)
    state = _first_person(current_state_text, agent_name)
    return f"""
My name is {agent_name}.

I consider myself a {personality or "ordinary"} person.

Just now, the situation I noticed or felt was:
{trigger or "Some changes have occurred in the current situation."}

This made me realize that I had a very natural thought:
{habitual}

My current main task is:
{intent or "No clear main task."}

To continue with this main task, I originally felt:
{intuition_text or "Act in the most appropriate way for the current situation first."}
{think_section}
So I originally intended to:
{goal_directed_action}

This naturally arising thought does not hinder my main task, but I still need to decide whether to do it now, save it for later, combine it with the original action, or temporarily ignore it.

Now, my state is:
{state or "No particularly obvious physical or emotional changes."}

What do I ultimately intend to do?

Please provide a brief first-person weighing of options, but ultimately decide on only one action to execute now.

Output only JSON:
{{
  "decision_mode": "goal_directed | habitual | sequence | combine | inhibit_habit",
  "thought": "Brief first-person weighing",
  "action": "A specific action starting with {agent_name} to be executed by the environment"
}}
""".strip()


def _first_person(text: str, agent_name: str) -> str:
    return str(text or "").strip().replace(agent_name, "I") if agent_name else str(text or "").strip()

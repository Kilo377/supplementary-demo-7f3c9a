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
    think_section = f"\n我又仔细想了一下：\n{think_text}\n" if think_text.strip() else ""
    trigger = _first_person(trigger_context_text, agent_name)
    habitual = _first_person(habitual_response.response_text, agent_name)
    intent = _first_person(intent_text, agent_name)
    state = _first_person(current_state_text, agent_name)
    return f"""
我叫{agent_name}。

我觉得我是一个{personality or "普通"}的人。

刚才，我注意到或感受到的情境是：
{trigger or "眼前的情境发生了一些变化。"}

这让我意识到自己有一个很自然的念头：
{habitual}

我现在的正事是：
{intent or "没有明确的正事。"}

为了继续做这件正事，我本来觉得：
{intuition_text or "先按照眼前最合适的方式行动。"}
{think_section}
所以我本来打算：
{goal_directed_action}

这个自然产生的念头并不妨碍我的正事，但我仍然需要决定现在是否顺手去做、留到稍后、和原来的行动结合，或者暂时不理会它。

现在，我的状态是：
{state or "没有特别明显的身体或情绪变化。"}

我最终打算怎么做？

请以第一人称进行简短权衡，但最终只决定一个现在实际执行的动作。

只输出 JSON：
{{
  "decision_mode": "goal_directed | habitual | sequence | combine | inhibit_habit",
  "thought": "第一人称的简短权衡",
  "action": "以{agent_name}开头、交给环境执行的一个具体动作"
}}
""".strip()


def _first_person(text: str, agent_name: str) -> str:
    return str(text or "").strip().replace(agent_name, "我") if agent_name else str(text or "").strip()

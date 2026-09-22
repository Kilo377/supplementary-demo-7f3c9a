from __future__ import annotations

from agent.habitual_controller import PreparedHabitualResponse


def build_goal_conflict_arbiter_prompt(
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

这让我下意识地觉得：
{habitual}

不过，我现在的正事是：
{intent or "没有明确的正事。"}

为了继续做这件正事，我本来觉得：
{intuition_text or "先按照眼前最合适的方式行动。"}
{think_section}
所以我本来打算：
{goal_directed_action}

但是，下意识产生的行为和我为了正事准备做的行为有些冲突，我不能同时按照原来的方式完成它们。

现在，我的状态是：
{state or "没有特别明显的身体或情绪变化。"}

我最终打算怎么做？

我可以继续原本为了正事准备做的行为；改为执行下意识产生的行为；调整两个行为的先后顺序；把两个行为自然地结合起来；或者明确压下这个习惯反应。

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

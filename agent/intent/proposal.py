from __future__ import annotations

from dataclasses import dataclass

from agent.perceive import PerceiveResult
from llm.api_manager import APIManager


@dataclass
class IntentProposalResult:
    intent_text: str
    provider_name: str
    model: str | None = None


def build_intent_proposal_prompt(
    result: PerceiveResult,
    *,
    agent_name: str,
    previous_feedback: str = "",
) -> str:
    feedback_block = ""
    if previous_feedback:
        feedback_block = f"""
上一轮环境反馈：
{previous_feedback}
"""
    return f"""下面是一段日常生活场景的旁白。
一个人刚刚置身其中时，身体、记忆和注意力自然形成的片刻心念。

{result.narration_text}{feedback_block}

现在，把镜头从环境轻轻转到 {agent_name} 的内心。
这个念头刚刚在 {agent_name} 心里浮起来。

此刻进入 {agent_name} 脑子的，只有眼前的东西、身体正在感到的东西，以及记忆里浮起的片段。
这个瞬间短到只够一个念头浮上来。
它还不是马上发生的动作，而是一个可能牵引后续几步行动的短期倾向。


这种念头听起来大概像：
“{agent_name}想先在客厅里找个舒服的位置安静待一会儿。”
“{agent_name}忽然有些口渴，想去找点水喝。”
“{agent_name}想确认电视还能不能像平时一样打开。”

{agent_name}心里先出现的是：
"""


def propose_new_intent(
    result: PerceiveResult,
    *,
    agent_name: str,
    provider_name: str = "ollama",
    model: str | None = None,
    previous_feedback: str = "",
) -> IntentProposalResult:
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.intent_proposal",
    )
    prompt = build_intent_proposal_prompt(
        result,
        agent_name=agent_name,
        previous_feedback=previous_feedback,
    )
    text = api.generate(prompt, model=model).strip()
    if text.startswith("你"):
        text = f"{agent_name}{text[1:]}"
    elif text.startswith(("他", "她")):
        text = f"{agent_name}{text[1:]}"
    return IntentProposalResult(
        intent_text=text,
        provider_name=api.provider_name,
        model=api.route.model,
    )

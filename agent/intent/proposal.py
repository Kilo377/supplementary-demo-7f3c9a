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
Previous environmental feedback:
{previous_feedback}
"""
    return f"""Below is a narration of a daily life scene.
When a person first enters this scene, their body, memories, and attention naturally form a fleeting thought.

{result.narration_text}{feedback_block}

Now, gently shift the camera from the environment to {agent_name}'s inner world.
This thought has just surfaced in {agent_name}'s mind.

At this moment, what enters {agent_name}'s mind is only what is before their eyes, what their body is currently sensing, and fragments floating up from memory.
This instant is so brief that it only allows one thought to surface.
It is not an action that will happen immediately, but a short-term inclination that may lead to subsequent steps of action.


Such thoughts might sound like:
"{agent_name} wants to find a comfortable spot in the living room and stay quietly for a while."
"{agent_name} suddenly feels thirsty and wants to go find some water to drink."
"{agent_name} wants to confirm whether the TV can be turned on as usual."

What first appears in {agent_name}'s mind is:
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
    if text.startswith("You"):
        text = f"{agent_name}{text[1:]}"
    elif text.startswith(("He", "She")):
        text = f"{agent_name}{text[1:]}"
    return IntentProposalResult(
        intent_text=text,
        provider_name=api.provider_name,
        model=api.route.model,
    )

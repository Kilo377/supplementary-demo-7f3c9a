from __future__ import annotations

import json
from dataclasses import dataclass

from agent.intent.state import IntentState
from llm.api_manager import APIManager


@dataclass
class IntentLifecycleResult:
    intent_status: str
    reason: str
    updated_intent: str
    raw_response: str = ""


def build_intent_lifecycle_prompt(
    *,
    agent_name: str,
    personality: str,
    current_state_belief: str,
    intent: IntentState,
    intent_decided_time: str,
    current_time: str,
    short_time_memory_text: str,
    force_reflect: bool = False,
) -> str:
    force_reflect_text = ""
    if force_reflect:
        force_reflect_text = "\nYou have already taken several steps toward this. Now carefully judge whether it has reached a point you can accept in real life. Do not remain active simply because you could theoretically continue doing similar actions."

    personality_text = personality.strip().rstrip("。.!！") or "Will make judgments based on the current situation"
    state_text = current_state_belief.strip() or "You have just completed this immediate step."

    return f"""Your name is {agent_name}, and your current state is:
{state_text}

You consider yourself to be a person with the personality of {personality_text}.

At {intent_decided_time}, you decided:
{intent.intent_text}

It is now {current_time}, and you have done the following:
{short_time_memory_text}
{force_reflect_text}

Do you feel that you have completed what you intended to do?
If not, do you still want to continue?

Please provide your status and reasoning:

- active: There is currently a clear, specific, unfinished next step supported by the existing environment or recent experiences.
  Do not remain active merely because more actions are theoretically possible.

- satisfied: Based on the currently visible environment, completed actions, and recent feedback,
you feel that this Intent has been sufficiently met. This also depends on your personality. For example, if you are someone who settles for less, you might finish things hastily. If you are more responsible, you may hold yourself to higher standards.

- deferred: The Intent is not yet fully satisfied, but due to fatigue, time constraints, missing environmental conditions,
  blocked paths, or other real-world limitations, it is suitable to continue later.

- abandoned: You have explicitly decided no longer to pursue this Intent.
  A long task, a temporary break, or insufficient environmental information does not itself constitute abandonment.

Requirements:
- Use the first person for 'reason', providing a brief Chinese sentence explaining your true judgment.
- First judge within the scope of the original Intent, then see if the actions already taken are sufficient to satisfy it.
- If you have only completed a room, a piece of furniture, or a partial step, it usually should not be judged as 'satisfied', unless the original Intent only required that specific part.
- If you still retain this Intent but wish to adjust its expression, fill in 'updated_intent'; otherwise, leave it empty.
- Return only JSON; do not explain your reasoning process.

Return format:
{{
  "intent_status": "active | satisfied | deferred | abandoned",
  "reason": "A brief Chinese reason",
  "updated_intent": "Fill in if adjusting the intent; otherwise, an empty string"
}}
"""


def evaluate_intent_lifecycle(
    *,
    agent_name: str,
    personality: str,
    current_state_belief: str,
    intent: IntentState,
    intent_decided_time: str,
    current_time: str,
    short_time_memory_text: str,
    force_reflect: bool = False,
    provider_name: str = "ollama",
    model: str | None = None,
) -> IntentLifecycleResult:
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.intent_lifecycle",
    )
    prompt = build_intent_lifecycle_prompt(
        agent_name=agent_name,
        personality=personality,
        current_state_belief=current_state_belief,
        intent=intent,
        intent_decided_time=intent_decided_time,
        current_time=current_time,
        short_time_memory_text=short_time_memory_text,
        force_reflect=force_reflect,
    )
    raw = api.generate(prompt, model=model)
    parsed = json.loads(_extract_json_text(raw))
    status = str(parsed.get("intent_status", "active") or "active").strip().lower()
    if status not in {"active", "satisfied", "deferred", "abandoned"}:
        status = "active"
    return IntentLifecycleResult(
        intent_status=status,
        reason=parsed.get("reason", "") or "",
        updated_intent=parsed.get("updated_intent", "") or "",
        raw_response=raw,
    )


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    if "```json" in stripped:
        after = stripped.split("```json", 1)[1]
        return after.split("```", 1)[0].strip()
    if "```" in stripped:
        after = stripped.split("```", 1)[1]
        return after.split("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    raise ValueError("No JSON object found in intent reflection response.")

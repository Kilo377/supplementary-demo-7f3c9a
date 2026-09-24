from __future__ import annotations

import json
from dataclasses import dataclass

from agent.intent.state import IntentState
from llm.api_manager import APIManager


@dataclass
class IntentProgressResult:
    progress_item: str
    raw_response: str = ""


def build_intent_progress_prompt(
    *,
    agent_name: str,
    intent: IntentState,
    action_proposal: str,
    execution_feedback: str,
) -> str:
    progress_text = "\n".join(f"- {item}" for item in intent.progress) or "None yet."
    return f"""These are the short-term actions taken by {agent_name}.
It only records action progress that is directly related to this intent and has already occurred.

What {agent_name} wants to do at this stage is:
{intent.intent_text}

To accomplish this, the existing progress is:
{progress_text}

The action just proposed:
{action_proposal}

Environmental feedback:
{execution_feedback}

Please compress the part that truly advanced the intent into a single short-term progress item.

Requirements:
- Only record what has already happened.
- Only record things related to the active intent.
- If the intent was not advanced just now, return an empty string for 'progress_item'.
- Do not write plans, reasons, or reasoning.
- Do not repeat content already expressed in existing progress.
- Return only JSON.

Return format:
{{
  "progress_item": "A brief progress item; empty string if no progress"
}}
"""


def summarize_intent_progress(
    *,
    agent_name: str,
    intent: IntentState,
    action_proposal: str,
    execution_feedback: str,
    provider_name: str = "ollama",
    model: str | None = None,
) -> IntentProgressResult:
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.intent_progress",
    )
    prompt = build_intent_progress_prompt(
        agent_name=agent_name,
        intent=intent,
        action_proposal=action_proposal,
        execution_feedback=execution_feedback,
    )
    raw = api.generate(prompt, model=model)
    parsed = json.loads(_extract_json_text(raw))
    return IntentProgressResult(
        progress_item=(parsed.get("progress_item", "") or "").strip(),
        raw_response=raw,
    )


def fallback_progress_item(
    *,
    agent_name: str,
    intent: IntentState,
    action_proposal: str,
    execution_feedback: str,
) -> str:
    text = execution_feedback.strip() or action_proposal.strip()
    if not text:
        return ""
    if any(word in text for word in ["Cannot find", "Cannot", "Unable to", "None", "Failed", "Mistaken record", "Fall"]):
        return ""
    if not _looks_related_to_intent(intent.intent_text, text):
        return ""
    return text.rstrip("。") + "。"


def append_progress(intent: IntentState, item: str, *, max_items: int = 12) -> None:
    cleaned = item.strip()
    if not cleaned:
        return
    if not cleaned.endswith(("。", "！", "？")):
        cleaned = f"{cleaned}。"
    if _is_duplicate_progress(intent.progress, cleaned):
        return
    intent.progress.append(cleaned)
    if len(intent.progress) > max_items:
        intent.progress[:] = intent.progress[-max_items:]


def format_progress_for_prompt(intent: IntentState) -> str:
    if not intent.progress:
        return "None yet."
    return "\n".join(f"- {item}" for item in intent.progress)


def _looks_related_to_intent(intent_text: str, text: str) -> bool:
    if any(word in intent_text for word in ["Cooking", "Fill my stomach", "Eat"]):
        return any(word in text for word in ["kitchen", "Refrigerator", "Ingredients", "Workbench", "Stove", "Pot", "Sink", "Cutting board", "Chop", "Wash", "Heat", "Boil", "Stir-fry", "Eat"])
    if any(word in intent_text for word in ["Drink water", "Thirsty", "Find some water"]):
        return any(word in text for word in ["Water", "Cup", "Refrigerator", "Sink", "Drink"])
    return True


def _is_duplicate_progress(existing: list[str], item: str) -> bool:
    normalized = _normalize(item)
    return any(normalized == _normalize(old) for old in existing)


def _normalize(text: str) -> str:
    return text.replace("Already", "").replace("(completed action marker)", "").replace("。", "").replace(" ", "")


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
    raise ValueError("No JSON object found in intent progress response.")

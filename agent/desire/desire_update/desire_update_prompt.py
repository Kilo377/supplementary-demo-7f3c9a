from __future__ import annotations

import json
from dataclasses import dataclass

from agent.desire.desire_state import DesireState
from llm.api_manager import APIManager


@dataclass
class DesireUpdateResult:
    benefit: str
    cost: str
    desire_state: DesireState
    physiological_reason: str = ""
    internal_state_reason: str = ""
    mental_reason: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "benefit": self.benefit,
            "cost": self.cost,
            "desire_state": self.desire_state.to_dict(),
            "physiological_reason": self.physiological_reason,
            "internal_state_reason": self.internal_state_reason,
            "mental_reason": self.mental_reason,
        }


def build_desire_update_prompt(
    *,
    agent_name: str,
    desire_state: dict,
    intent_text: str,
    intent_status: str,
    intent_cycle_context: str,
    self_belief: str = "",
    personal_context: str = "",
    update_mode: str = "intent_end",
) -> str:
    personal_context_block = ""
    if personal_context.strip():
        personal_context_block = f"""
Long-term personal background of {agent_name}:
{personal_context.strip()}
"""
    is_step_update = update_mode == "every_step"
    update_timing_text = (
        "This update occurs immediately after completing one action; the intent may still be in an active state."
        if is_step_update
        else "This update occurs after the end of an intent lifecycle."
    )
    context_label = "This action and its latest result" if is_step_update else "The context of this intent cycle"
    self_belief_label = "Assessment of one's own state after this action" if is_step_update else "Assessment of one's own state at the end of the intent"
    work_goal_rule = (
        "- Only when the actual result of this action clearly satisfies a work_goal can it be changed to true;"
        "Do not complete the goal prematurely just because the intent is still ongoing or merely moving toward the goal"
        if is_step_update
        else "- If the behavior and intent state corresponding to the current intent cycle satisfy a work_goal, change completed to true"
    )
    single_goal_rule = (
        "- If this action only satisfies one of the work_goals, change only that goal to true and keep others false"
        if is_step_update
        else "- If the just-ended intent only satisfies one of the work_goals, change only that goal to true and keep others false"
    )
    return f"""You are updating the Desire state of a human agent.

Desire is not intent. Desire indicates whether {agent_name}'s needs, mood, and primary goals are being met.
{update_timing_text}

Current Desire state:
{json.dumps(desire_state, ensure_ascii=False, indent=2)}

Just-ended intent:
{intent_text}

Intent lifecycle status:
{intent_status}

{context_label}:
{intent_cycle_context}

{agent_name} {self_belief_label}:
{self_belief or "None."}
{personal_context_block}

Update rules:
1. physiological_state is a 0-10 scale of physiological need pressure; higher scores mean stronger need pressure.
   - hunger: 0=not hungry at all, 10=very hungry
   - thirst: 0=not thirsty at all, 10=very thirsty
   - hygiene: 0=clean/no need to clean, 10=dirty/strong need to clean
   - Updates should be conservative: keep unrelated items unchanged as much as possible
   - When eating, drinking, or cleaning is completed, the corresponding pressure can drop significantly

2. internal_state is a calculable 0-10 scale of internal state.
   - stress: 0=no stress, 10=extreme stress
   - tension: 0=completely relaxed, 10=very tense
   - fatigue: 0=not tired at all, 10=very fatigued
   - depletion: ego depletion of self-control resources, not equal to physical fatigue; 3=normal baseline, 7=noticeably depleted, 10=extremely depleted
   - cognitive_load: current cognitive load, not equal to tension; 3=normal baseline, 7=noticeably elevated, 10=extreme load
   - In internal state gating, stress uses 3 as normal baseline and 7 as noticeably elevated; do not significantly increase it solely due to a single ordinary action.
   - Continuously suppressing impulses or exerting effortful self-control may increase depletion; parallel processing or complex thinking may increase cognitive_load. Keep original values when there is no relevant evidence; reduce them during rest or when burdens are lifted.
   - Ordinary actions typically change by at most 1; only strong, sustained, or clearly restorative experiences allow more significant changes
   - Physical labor usually increases fatigue; rest usually decreases fatigue
   - Failure and obstruction may increase stress or tension; successfully completing things may decrease them

3. mental is a pure natural language psychological state.
   - Output a new natural language mental state
   - A brief reason can be provided
   - Maintain continuity; do not suddenly change personality, long-term values, or new goals
   - Can naturally reflect stress, tension, and fatigue, but do not expose scale values

4. work_goal represents the primary goals in Desire, not intent.
   - Each work_goal has only text and completed
   - There may be multiple work_goals simultaneously; they are independent of each other and must be evaluated item by item
   - The returned work_goal array must retain all original goals from the current Desire state without merging, splitting, rewriting, or omitting any text
   - Only judge whether this desire has been satisfied
   - Do not add reason, status, active, blocked, or abandoned to work_goals
   - If a work_goal is already completed=true, keep it true
   {work_goal_rule}
   {single_goal_rule}
   - If all work_goals are completed=true, the outer simulation can end

5. Do not invent new Desire dimensions.
6. Do not directly copy intent lifecycle status into work_goal status.
7. Return only JSON; do not explain reasoning.

Return JSON format:
{{
  "benefit": "The benefit brought by this intent cycle, in one Chinese sentence",
  "cost": "The cost brought by this intent cycle, in one Chinese sentence",
  "physiological_state_update": {{
    "after": {{
      "hunger": 0,
      "thirst": 0,
      "hygiene": 0
    }},
    "reason": "Why the physiological state is updated this way"
  }},
  "internal_state_update": {{
    "after": {{
      "stress": 0,
      "tension": 0,
      "fatigue": 0,
      "depletion": 3,
      "cognitive_load": 3
    }},
    "reason": "Why the internal state is updated this way"
  }},
  "mental_update": {{
    "after": "New natural language mood state",
    "reason": "Why the mental state is updated this way"
  }},
  "work_goal": [
    {{
      "text": "Original work goal text",
      "completed": false
    }}
  ]
}}
"""


def update_desire_state(
    *,
    agent_name: str,
    desire_state: DesireState,
    intent_text: str,
    intent_status: str,
    intent_cycle_context: str,
    self_belief: str = "",
    provider_name: str = "ollama",
    model: str | None = None,
    personal_context: str = "",
    update_mode: str = "intent_end",
) -> DesireUpdateResult:
    prompt = build_desire_update_prompt(
        agent_name=agent_name,
        desire_state=desire_state.to_dict(),
        intent_text=intent_text,
        intent_status=intent_status,
        intent_cycle_context=intent_cycle_context,
        self_belief=self_belief,
        personal_context=personal_context,
        update_mode=update_mode,
    )
    try:
        raw = APIManager(
            provider_name=provider_name,
            task_name="agent.desire_update",
        ).generate(prompt, model=model)
        parsed = json.loads(_extract_json_text(raw))
        updated = _desire_state_from_update(
            before=desire_state,
            parsed=parsed,
        )
        return DesireUpdateResult(
            benefit=parsed.get("benefit", "") or "",
            cost=parsed.get("cost", "") or "",
            desire_state=updated,
            physiological_reason=_update_reason(parsed, "physiological_state_update"),
            internal_state_reason=_update_reason(parsed, "internal_state_update"),
            mental_reason=_update_reason(parsed, "mental_update"),
            raw_response=raw,
        )
    except Exception as error:
        return DesireUpdateResult(
            benefit="",
            cost=f"Desire update failed: {error}",
            desire_state=desire_state,
            raw_response="",
        )


def _desire_state_from_update(*, before: DesireState, parsed: dict) -> DesireState:
    before_data = before.to_dict()
    physiological_after = (
        parsed.get("physiological_state_update", {}).get("after", {})
        if isinstance(parsed.get("physiological_state_update", {}), dict)
        else {}
    )
    internal_state_after = (
        parsed.get("internal_state_update", {}).get("after", {})
        if isinstance(parsed.get("internal_state_update", {}), dict)
        else {}
    )
    mental_after = (
        parsed.get("mental_update", {}).get("after", "")
        if isinstance(parsed.get("mental_update", {}), dict)
        else ""
    )
    work_goal_after = parsed.get("work_goal", before_data.get("work_goal", []))
    merged_work_goal = _merge_work_goal_updates(
        before_goals=before_data.get("work_goal", []),
        updated_goals=work_goal_after,
    )
    return DesireState.from_dict(
        {
            "physiological_state": _merge_numeric_state(
                before_data.get("physiological_state", {}),
                physiological_after,
            ),
            "internal_state": _merge_numeric_state(
                before_data.get("internal_state", {}),
                internal_state_after,
            ),
            "mental": mental_after or before_data.get("mental", ""),
            "work_goal": merged_work_goal,
        }
    )


def _update_reason(parsed: dict, key: str) -> str:
    value = parsed.get(key, {})
    if not isinstance(value, dict):
        return ""
    return str(value.get("reason", "") or "").strip()


def _merge_numeric_state(before: dict, after) -> dict:
    merged = dict(before or {})
    if isinstance(after, dict):
        merged.update(after)
    return merged


def _merge_work_goal_updates(*, before_goals: list, updated_goals) -> list[dict]:
    if not isinstance(updated_goals, list):
        updated_goals = []

    completed_by_text: dict[str, bool] = {}
    for item in updated_goals:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        completed_by_text[text] = bool(item.get("completed", False))

    merged = []
    for item in before_goals:
        if not isinstance(item, dict):
            text = str(item).strip()
            before_completed = False
        else:
            text = str(item.get("text", "")).strip()
            before_completed = bool(item.get("completed", False))
        if not text:
            continue
        merged.append(
            {
                "text": text,
                "completed": before_completed or completed_by_text.get(text, False),
            }
        )
    return merged


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
    raise ValueError("No JSON object found in desire update response.")

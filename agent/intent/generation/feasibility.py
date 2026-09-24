from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from agent.belief.short_time_memory import ShortTermMemory
from agent.belief.spatial_memory.spatial_belief import SpatialBelief
from agent.intent.generation.candidate_generation import IntentCandidate
from llm.api_manager import APIManager


FEASIBLE = "feasible"
INFEASIBLE = "infeasible"


@dataclass
class IntentFeasibilityCandidate:
    original_intent_text: str
    intent_text: str
    source_desire: str
    candidate_reason: str
    feasibility: str
    feasibility_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "original_intent_text": self.original_intent_text,
            "intent_text": self.intent_text,
            "source_desire": self.source_desire,
            "candidate_reason": self.candidate_reason,
            "feasibility": self.feasibility,
            "feasibility_reason": self.feasibility_reason,
        }


@dataclass
class IntentFeasibilityResult:
    candidates: list[IntentFeasibilityCandidate]
    provider_name: str
    model: str | None = None
    raw_response: str = ""
    error: str = ""

    @property
    def feasible_candidates(self) -> list[IntentFeasibilityCandidate]:
        return [item for item in self.candidates if item.feasibility == FEASIBLE]

    @property
    def infeasible_candidates(self) -> list[IntentFeasibilityCandidate]:
        return [item for item in self.candidates if item.feasibility == INFEASIBLE]

    def to_dict(self) -> dict:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "provider_name": self.provider_name,
            "model": self.model,
            "error": self.error,
        }


def build_intent_feasibility_prompt(
    *,
    agent_name: str,
    candidates: Iterable[IntentCandidate | dict],
    spatial_belief: SpatialBelief | dict | None,
    self_belief: str = "",
    short_time_memory: ShortTermMemory | str | None = None,
    personal_context: str = "",
) -> str:
    candidate_data = [_candidate_to_dict(candidate) for candidate in candidates]
    spatial_data = _spatial_belief_to_prompt_data(spatial_belief)
    memory_text = _short_time_memory_text(short_time_memory)
    self_belief_block = ""
    if self_belief.strip():
        self_belief_block = f"""
{agent_name}'s assessment of their current state:
{self_belief.strip()}
"""
    memory_block = ""
    if memory_text.strip():
        memory_block = f"""
{agent_name} just experienced:
{memory_text.strip()}
"""
    personal_context_block = ""
    if personal_context.strip():
        personal_context_block = f"""
Long-term personal background of {agent_name}:
{personal_context.strip()}
"""
    return f"""
You are judging whether intent candidates for a human agent are feasible.
You will receive a list of intent candidates. Please evaluate each candidate individually to determine if it is currently feasible. Each candidate must return a judgment.
If an intent is feasible, mark it as "feasible"; otherwise, mark it as "infeasible".
The primary basis for judging feasibility is the likelihood of the intent being realized in the current context. For example, the intent might be to play computer games. However, if there is no computer in the scene, this intent should be marked as infeasible.

agent_name:
{agent_name}

intent candidates:
{json.dumps(candidate_data, ensure_ascii=False, indent=2)}

Belief: spatial memory
{json.dumps(spatial_data, ensure_ascii=False, indent=2)}
{self_belief_block}{memory_block}{personal_context_block}
Judgment rules:
1. Do not mark a candidate as infeasible simply because it is vague. For example, "find something to eat" can be realized based on spatial memory of the kitchen, refrigerator, dining table, etc.
2. If a candidate explicitly requires a certain type of spatial element, but the spatial memory contains no corresponding or similar elements, mark it as infeasible.
3. Preserve source_desire and candidate_reason.
4. For infeasible intents, return an empty string for intent_text.
5. Return only JSON; do not explain the reasoning process.

Return JSON format:
{{
  "candidates": [
    {{
      "original_intent_text": "{agent_name} plans to tidy up the house.",
      "intent_text": "{agent_name} plans to tidy up the clutter in the living room.",
      "source_desire": "work_goal",
      "candidate_reason": "There is an unfinished work_goal: {agent_name} wants to tidy up the house.",
      "feasibility": "feasible",
      "feasibility_reason": "Spatial memory contains a living room and items that can be tidied."
    }},
    {{
      "original_intent_text": "{agent_name} plans to play computer games to relax.",
      "intent_text": "",
      "source_desire": "mental",
      "candidate_reason": "The current mental state needs relaxation.",
      "feasibility": "infeasible",
      "feasibility_reason": "Spatial memory contains no computers, game consoles, or similar devices."
    }}
  ]
}}
"""


def check_intent_feasibility(
    *,
    agent_name: str,
    candidates: list[IntentCandidate],
    spatial_belief: SpatialBelief | None,
    self_belief: str = "",
    short_time_memory: ShortTermMemory | str | None = None,
    provider_name: str = "ollama",
    model: str | None = None,
    personal_context: str = "",
) -> IntentFeasibilityResult:
    prompt = build_intent_feasibility_prompt(
        agent_name=agent_name,
        candidates=candidates,
        spatial_belief=spatial_belief,
        self_belief=self_belief,
        short_time_memory=short_time_memory,
        personal_context=personal_context,
    )
    raw = ""
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.intent_feasibility",
    )
    try:
        raw = api.generate(prompt, model=model)
        parsed = json.loads(_extract_json_text(raw))
        feasible_candidates = _parse_feasibility_candidates(parsed, candidates)
        return IntentFeasibilityResult(
            candidates=feasible_candidates,
            provider_name=api.provider_name,
            model=api.route.model,
            raw_response=raw,
        )
    except Exception as error:
        return IntentFeasibilityResult(
            candidates=_fallback_feasibility(candidates, spatial_belief=spatial_belief),
            provider_name=api.provider_name,
            model=api.route.model,
            raw_response=raw,
            error=f"Intent feasibility failed: {error}",
        )


def _parse_feasibility_candidates(
    parsed: dict,
    original_candidates: list[IntentCandidate],
) -> list[IntentFeasibilityCandidate]:
    items = parsed.get("candidates", [])
    if not isinstance(items, list):
        return []

    by_original_text = {candidate.intent_text: candidate for candidate in original_candidates}
    result: list[IntentFeasibilityCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        original_text = str(item.get("original_intent_text", "")).strip()
        original = by_original_text.get(original_text)
        source_desire = str(item.get("source_desire", "")).strip()
        candidate_reason = str(item.get("candidate_reason", "")).strip()
        if original is not None:
            source_desire = source_desire or original.source_desire
            candidate_reason = candidate_reason or original.reason

        feasibility = _normalize_feasibility(str(item.get("feasibility", "")).strip())
        intent_text = str(item.get("intent_text", "")).strip()
        if feasibility == INFEASIBLE:
            intent_text = ""
        if not original_text or not source_desire or not feasibility:
            continue
        result.append(
            IntentFeasibilityCandidate(
                original_intent_text=original_text,
                intent_text=intent_text,
                source_desire=source_desire,
                candidate_reason=candidate_reason,
                feasibility=feasibility,
                feasibility_reason=str(item.get("feasibility_reason", "")).strip()
                or str(item.get("reason", "")).strip(),
            )
        )
    return result


def _fallback_feasibility(
    candidates: list[IntentCandidate],
    *,
    spatial_belief: SpatialBelief | None,
) -> list[IntentFeasibilityCandidate]:
    element_names = set()
    area_names = set()
    if spatial_belief is not None:
        element_names = {element.name for element in spatial_belief.iter_elements()}
        area_names = {area.area_name for area in spatial_belief.iter_areas()}

    result: list[IntentFeasibilityCandidate] = []
    for candidate in candidates:
        intent_text = _ground_generic_intent(candidate.intent_text, element_names, area_names)
        result.append(
            IntentFeasibilityCandidate(
                original_intent_text=candidate.intent_text,
                intent_text=intent_text,
                source_desire=candidate.source_desire,
                candidate_reason=candidate.reason,
                feasibility=FEASIBLE,
                feasibility_reason="LLM feasibility is unavailable; conservatively retain this candidate and wait for subsequent stages or environmental feedback to correct it.",
            )
        )
    return result


def _ground_generic_intent(
    intent_text: str,
    element_names: set[str],
    area_names: set[str],
) -> str:
    if "find something to eat" in intent_text and "kitchen" in area_names:
        return intent_text.replace("find something to eat", "go to the kitchen to find something to eat")
    if "find something to drink" in intent_text and "kitchen" in area_names:
        return intent_text.replace("find something to drink", "Go to the kitchen to find something to drink.")
    if "Tidy up the house." in intent_text:
        if "Living room" in area_names:
            return intent_text.replace("Tidy up the house.", "Tidy up items in the living room.")
        if "Dining table" in element_names:
            return intent_text.replace("Tidy up the house.", "Tidy up items near the dining table.")
    return intent_text


def _candidate_to_dict(candidate: IntentCandidate | dict) -> dict:
    if isinstance(candidate, IntentCandidate):
        return candidate.to_dict()
    return dict(candidate)


def _spatial_belief_to_prompt_data(spatial_belief: SpatialBelief | dict | None) -> dict:
    if spatial_belief is None:
        return {"areas": []}
    if isinstance(spatial_belief, dict):
        return spatial_belief
    return {
        "home_id": spatial_belief.home_id,
        "areas": [
            {
                "area_id": area.area_id,
                "area_name": area.area_name,
                "elements": [
                    {
                        "element_id": element.element_id,
                        "element_name": element.name,
                        "physical_status": element.physical_status,
                        "evolution_status": element.evolution_status,
                        "interaction_status": element.interaction_status,
                        "state_details": dict(element.state_details),
                    }
                    for element in area.elements.values()
                ],
            }
            for area in spatial_belief.iter_areas()
        ],
    }


def _short_time_memory_text(short_time_memory: ShortTermMemory | str | None) -> str:
    if short_time_memory is None:
        return ""
    if isinstance(short_time_memory, str):
        return short_time_memory
    return short_time_memory.format_experience_for_prompt(count=8, empty_text="")


def _normalize_feasibility(value: str) -> str:
    lowered = value.lower()
    if lowered == FEASIBLE:
        return FEASIBLE
    if lowered == INFEASIBLE:
        return INFEASIBLE
    return ""


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
    raise ValueError("No JSON object found in intent feasibility response.")

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from agent.belief.short_time_memory import ShortTermMemory
from agent.desire.desire_state import DesireState
from agent.desire.desire_subjective_interpretation import (
    DesireSubjectiveSignal,
    build_desire_subjective_signals,
)
from agent.intent.generation.feasibility import FEASIBLE, IntentFeasibilityCandidate
from llm.api_manager import APIManager


@dataclass
class SelectedIntent:
    intent_text: str
    source_desire: str
    candidate_reason: str = ""
    feasibility_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "intent_text": self.intent_text,
            "source_desire": self.source_desire,
            "candidate_reason": self.candidate_reason,
            "feasibility_reason": self.feasibility_reason,
        }


@dataclass
class IntentSelectionResult:
    selected_intent: SelectedIntent | None
    selection_reason: str
    provider_name: str
    model: str | None = None
    raw_response: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "selected_intent": self.selected_intent.to_dict() if self.selected_intent else None,
            "selection_reason": self.selection_reason,
            "provider_name": self.provider_name,
            "model": self.model,
            "error": self.error,
        }


def build_intent_selection_prompt(
    *,
    agent_name: str,
    desire_signals: list[DesireSubjectiveSignal] | list[dict],
    feasible_candidates: Iterable[IntentFeasibilityCandidate | dict],
    self_belief: str = "",
    short_time_memory: ShortTermMemory | str | None = None,
    personal_context: str = "",
) -> str:
    desire_data = [
        signal.to_dict() if isinstance(signal, DesireSubjectiveSignal) else dict(signal)
        for signal in desire_signals
    ]
    candidate_data = [_candidate_to_dict(candidate) for candidate in feasible_candidates]
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

You are choosing the final intent for {agent_name} to commit.
According to Bratman's philosophy of action, human intent generation may be related to one's own desires; a person may have many desires simultaneously. For example, I want to stay healthy, but I also want to sleep in. Beliefs may also support different means. Which desire becomes the intent usually requires a selection process.
Therefore, you need to choose the most suitable intent to commit now from some feasible intent candidates.

Intent candidates:
{json.dumps(candidate_data, ensure_ascii=False, indent=2)}
{self_belief_block}{memory_block}{personal_context_block}

Some notes:
1. An attention candidate indicates something recently observed that is noteworthy; if it is obviously urgent or will change the current situation, it can be prioritized.
2. If short-term memory shows that a certain type of task has just been completed or repeatedly attempted, avoid choosing it again.
3. If self_belief shows that the current state is more continuous with a candidate, consider continuity and convenience.
4. The output is the committed intent, which remains a high-level intention, not an action sequence.
5. Return only JSON; do not explain the reasoning process.

Return JSON format:
{{
  "selected_intent": {{
    "intent_text": "{agent_name} intends to tidy up items in the living room.",
    "source_desire": "work_goal",
    "candidate_reason": "There is an unfinished work_goal: {agent_name} wants to tidy up the house.",
    "feasibility_reason": "Spatial memory contains the living room and tidiable items."
  }},
  "selection_reason": "{agent_name}'s current physiological and internal state needs are only moderate, and the mental state is stable; meanwhile, there is an unfinished work_goal, so it is more suitable to proceed with tidying up the house now."
}}
"""


def select_intent(
    *,
    agent_name: str,
    desire_state: DesireState,
    feasible_candidates: list[IntentFeasibilityCandidate],
    self_belief: str = "",
    short_time_memory: ShortTermMemory | str | None = None,
    provider_name: str = "ollama",
    model: str | None = None,
    personal_context: str = "",
) -> IntentSelectionResult:
    feasible_only = [candidate for candidate in feasible_candidates if candidate.feasibility == FEASIBLE]
    if not feasible_only:
        return IntentSelectionResult(
            selected_intent=None,
            selection_reason="No feasible intent candidate.",
            provider_name=provider_name,
            model=model,
        )

    desire_signals = build_desire_subjective_signals(desire_state, agent_name=agent_name)
    prompt = build_intent_selection_prompt(
        agent_name=agent_name,
        desire_signals=desire_signals,
        feasible_candidates=feasible_only,
        self_belief=self_belief,
        short_time_memory=short_time_memory,
        personal_context=personal_context,
    )
    raw = ""
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.intent_selection",
    )
    try:
        raw = api.generate(prompt, model=model)
        parsed = json.loads(_extract_json_text(raw))
        selected = _parse_selected_intent(parsed, feasible_only)
        return IntentSelectionResult(
            selected_intent=selected,
            selection_reason=str(parsed.get("selection_reason", "")).strip(),
            provider_name=api.provider_name,
            model=api.route.model,
            raw_response=raw,
        )
    except Exception as error:
        selected = _fallback_select_intent(desire_signals=desire_signals, feasible_candidates=feasible_only)
        return IntentSelectionResult(
            selected_intent=selected,
            selection_reason="LLM selection unavailable; using conservative fallback selection.",
            provider_name=api.provider_name,
            model=api.route.model,
            raw_response=raw,
            error=f"Intent selection failed: {error}",
        )


def _parse_selected_intent(
    parsed: dict,
    feasible_candidates: list[IntentFeasibilityCandidate],
) -> SelectedIntent | None:
    data = parsed.get("selected_intent", {})
    if not isinstance(data, dict):
        return None
    intent_text = str(data.get("intent_text", "")).strip()
    if not intent_text:
        return None

    matching = _find_matching_candidate(intent_text, feasible_candidates)
    if matching is not None:
        return SelectedIntent(
            intent_text=matching.intent_text,
            source_desire=matching.source_desire,
            candidate_reason=matching.candidate_reason,
            feasibility_reason=matching.feasibility_reason,
        )
    return SelectedIntent(
        intent_text=intent_text,
        source_desire=str(data.get("source_desire", "")).strip(),
        candidate_reason=str(data.get("candidate_reason", "")).strip(),
        feasibility_reason=str(data.get("feasibility_reason", "")).strip(),
    )


def _fallback_select_intent(
    *,
    desire_signals: list[DesireSubjectiveSignal],
    feasible_candidates: list[IntentFeasibilityCandidate],
) -> SelectedIntent:
    by_source = {candidate.source_desire: candidate for candidate in feasible_candidates}

    urgent_source = _subjectively_urgent_source(desire_signals)
    if urgent_source and urgent_source in by_source:
        return _selected_from_feasible_candidate(by_source[urgent_source])

    if any(signal.source_desire == "work_goal" for signal in desire_signals) and "work_goal" in by_source:
        return _selected_from_feasible_candidate(by_source["work_goal"])

    if urgent_source in by_source:
        return _selected_from_feasible_candidate(by_source[urgent_source])

    return _selected_from_feasible_candidate(feasible_candidates[0])


def _subjectively_urgent_source(desire_signals: list[DesireSubjectiveSignal]) -> str:
    urgent_markers = ("Very", "Really want to", "Very hungry", "Very thirsty", "Very tired")
    for signal in desire_signals:
        if any(marker in signal.subjective_interpretation for marker in urgent_markers):
            return signal.source_desire
    return ""


def _selected_from_feasible_candidate(candidate: IntentFeasibilityCandidate) -> SelectedIntent:
    return SelectedIntent(
        intent_text=candidate.intent_text,
        source_desire=candidate.source_desire,
        candidate_reason=candidate.candidate_reason,
        feasibility_reason=candidate.feasibility_reason,
    )


def _find_matching_candidate(
    intent_text: str,
    feasible_candidates: list[IntentFeasibilityCandidate],
) -> IntentFeasibilityCandidate | None:
    for candidate in feasible_candidates:
        if candidate.intent_text == intent_text:
            return candidate
    return None


def _candidate_to_dict(candidate: IntentFeasibilityCandidate | dict) -> dict:
    if isinstance(candidate, IntentFeasibilityCandidate):
        return candidate.to_dict()
    return dict(candidate)


def _short_time_memory_text(short_time_memory: ShortTermMemory | str | None) -> str:
    if short_time_memory is None:
        return ""
    if isinstance(short_time_memory, str):
        return short_time_memory
    return short_time_memory.format_experience_for_prompt(count=8, empty_text="")


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
    raise ValueError("No JSON object found in intent selection response.")

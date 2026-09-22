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
{agent_name} 对自己当前状态的判断：
{self_belief.strip()}
"""
    memory_block = ""
    if memory_text.strip():
        memory_block = f"""
{agent_name} 刚刚经历过：
{memory_text.strip()}
"""
    personal_context_block = ""
    if personal_context.strip():
        personal_context_block = f"""
{agent_name} 的长期个人背景：
{personal_context.strip()}
"""
    return f"""

你在为一个人类 {agent_name} 选择最终要 commit 的 intent。
根据 Bratman 的行动哲学理论，人类intent的产生可能和自身的Desire有关，人可能同时有很多 desire。比如我想保持健康，也想睡懒觉。belief 也可能支持不同手段。最后哪个 desire 变成 intent，通常需要一个选择过程。
因此，你需要从一些 feasible 的 intent candidates 中选择一个最适合现在 commit 的 intent。

intent candidates:
{json.dumps(candidate_data, ensure_ascii=False, indent=2)}
{self_belief_block}{memory_block}{personal_context_block}

一些主意事项：
1. attention candidate 表示刚刚观察到值得注意的事情；如果它明显紧急或会改变当前处境，可以优先。
2. 如果 short-term memory 显示刚刚已经完成或重复尝试过某类事情，避免重复选择。
3. 如果 self_belief 显示当前状态与某个 candidate 更连续，可以考虑顺路和连续性。
4. 输出的是 committed intent，仍然是高层意图，不是 action sequence。
5. 只返回 JSON，不要解释推理过程。

返回 JSON 格式：
{{
  "selected_intent": {{
    "intent_text": "{agent_name}打算整理客厅里的物品。",
    "source_desire": "work_goal",
    "candidate_reason": "存在未完成的 work_goal：{agent_name}想要整理家务。",
    "feasibility_reason": "空间记忆中有客厅和可整理物品。"
  }},
  "selection_reason": "{agent_name} 当前生理和内部状态需求只是中等，mental 状态平稳；同时存在未完成的 work_goal，因此现在更适合推进整理家务。"
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
            selection_reason="没有可行的 intent candidate。",
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
            selection_reason="LLM selection 不可用，使用保守 fallback 选择。",
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
    urgent_markers = ("非常", "很想", "很饿", "很渴", "很累")
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

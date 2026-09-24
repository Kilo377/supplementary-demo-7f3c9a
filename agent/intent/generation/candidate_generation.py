from __future__ import annotations

import json
from dataclasses import dataclass

from agent.desire.desire_state import DesireState
from agent.desire.desire_subjective_interpretation import (
    DesireSubjectiveSignal,
    build_desire_subjective_signals,
)
from agent.intent.generation.attention_affect import IntentAttentionAffect
from llm.api_manager import APIManager


@dataclass
class IntentCandidate:
    intent_text: str
    source_desire: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "intent_text": self.intent_text,
            "source_desire": self.source_desire,
            "reason": self.reason,
        }


@dataclass
class IntentCandidateGenerationResult:
    candidates: list[IntentCandidate]
    provider_name: str
    model: str | None = None
    raw_response: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "provider_name": self.provider_name,
            "model": self.model,
            "error": self.error,
        }


# TODO: Intent lifecycle should eventually support suspended intents. After each
# action, the agent may decide whether the active intent should continue, be
# abandoned, or be suspended; suspended intents should re-enter this candidate pool.
# TODO: Intent candidates should also be produced from sudden events surfaced by
# Perception + Attention, not only from Desire-derived subjective signals.
def build_intent_candidate_generation_prompt(
    *,
    agent_name: str,
    desire_signals: list[DesireSubjectiveSignal] | list[dict],
    attention_affect: IntentAttentionAffect | dict | None = None,
    max_candidates: int = 5,
    personal_context: str = "",
) -> str:
    signal_data = [
        signal.to_dict() if isinstance(signal, DesireSubjectiveSignal) else dict(signal)
        for signal in desire_signals
    ]
    attention_block = ""
    if attention_affect is not None:
        attention_data = (
            attention_affect.to_dict()
            if isinstance(attention_affect, IntentAttentionAffect)
            else dict(attention_affect)
        )
        attention_block = f"""
Current attention trigger:
{json.dumps(attention_data, ensure_ascii=False, indent=2)}
"""
    personal_context_block = ""
    if personal_context.strip():
        personal_context_block = f"""
Long-term personal background:
{personal_context.strip()}
"""
    return f"""You are generating intent candidates for a human {agent_name}.
You need to generate the thoughts of {agent_name} based on Desire and current attention. These may be intentions to do something or desire-driven thoughts.
According to Bratman's BDI theory, human intent generation may be related to one's own Desires; a person may have multiple desires simultaneously. For example, I want to stay healthy, but I also want to sleep in.
Additionally, a person might generate a temporary candidate intent due to suddenly noticing something.
Therefore, there may be multiple Intents as candidates.

Desire:
{json.dumps(signal_data, ensure_ascii=False, indent=2)}
{attention_block}{personal_context_block}

It is worth noting that,
1. Intent is a high-level thought, not an action. In fact, intent is a commitment-like mental state that will influence subsequent actions.
2. It is not necessary to generate an intent for every Desire.
3. If the mental state is merely stable, no candidate needs to be generated; if there is a clear tendency to maintain the state, a low-interference candidate can also be generated.
4. source_desire must use one of these formats:
   - physiological_state.hunger
   - physiological_state.thirst
   - physiological_state.hygiene
   - internal_state.stress
   - internal_state.tension
   - internal_state.fatigue
   - mental
   - work_goal
   - attention
5. reason should directly reuse the subjective_interpretation of the corresponding Desire.
   - Do not expand "I'm a little hungry." into "Having a slight sense of hunger prompts {agent_name} to have the thought of wanting to eat."
   - If source_desire is attention, reason can directly reuse notice_text, adding a very brief explanation if necessary.
6. Generate at most {max_candidates} candidates.
7. Do not generate candidates unsupported by the subjective interpretation of Desire or current attention.
8. Return only JSON; do not explain the reasoning process.

Return JSON format:
{{
  "candidates": [
    {{
      "intent_text": "{agent_name} plans to find something to eat.",
      "source_desire": "physiological_state.hunger",
      "reason": "I'm a little hungry."
    }}
  ]
}}
"""


def generate_intent_candidates(
    *,
    agent_name: str,
    desire_state: DesireState,
    attention_affect: IntentAttentionAffect | None = None,
    provider_name: str = "ollama",
    model: str | None = None,
    max_candidates: int = 5,
    personal_context: str = "",
) -> IntentCandidateGenerationResult:
    desire_signals = build_desire_subjective_signals(desire_state, agent_name=agent_name)
    prompt = build_intent_candidate_generation_prompt(
        agent_name=agent_name,
        desire_signals=desire_signals,
        attention_affect=attention_affect,
        max_candidates=max_candidates,
        personal_context=personal_context,
    )
    raw = ""
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.intent_candidate_generation",
    )
    try:
        raw = api.generate(prompt, model=model)
        parsed = json.loads(_extract_json_text(raw))
        candidates = _parse_candidates(parsed, agent_name=agent_name, max_candidates=max_candidates)
        return IntentCandidateGenerationResult(
            candidates=candidates,
            provider_name=api.provider_name,
            model=api.route.model,
            raw_response=raw,
        )
    except Exception as error:
        return IntentCandidateGenerationResult(
            candidates=_fallback_candidates(
                agent_name=agent_name,
                desire_signals=desire_signals,
                attention_affect=attention_affect,
                max_candidates=max_candidates,
            ),
            provider_name=api.provider_name,
            model=api.route.model,
            raw_response=raw,
            error=f"Intent candidate generation failed: {error}",
        )


def _parse_candidates(
    parsed: dict,
    *,
    agent_name: str,
    max_candidates: int,
) -> list[IntentCandidate]:
    items = parsed.get("candidates", [])
    if not isinstance(items, list):
        return []

    candidates: list[IntentCandidate] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        intent_text = _normalize_agent_name(str(item.get("intent_text", "")).strip(), agent_name)
        source_desire = str(item.get("source_desire", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not intent_text or not source_desire or intent_text in seen:
            continue
        candidates.append(
            IntentCandidate(
                intent_text=intent_text,
                source_desire=source_desire,
                reason=reason,
            )
        )
        seen.add(intent_text)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _fallback_candidates(
    *,
    agent_name: str,
    desire_signals: list[DesireSubjectiveSignal],
    attention_affect: IntentAttentionAffect | None,
    max_candidates: int,
) -> list[IntentCandidate]:
    candidates: list[IntentCandidate] = []
    for signal in desire_signals:
        candidate = _fallback_candidate_from_signal(agent_name=agent_name, signal=signal)
        if candidate is not None:
            candidates.append(candidate)

    if attention_affect is not None and len(candidates) < max_candidates:
        candidates.append(
            IntentCandidate(
                intent_text=f"{agent_name} plans to first focus on the situation just noticed.",
                source_desire="attention",
                reason=attention_affect.notice_text,
            )
        )

    return candidates[:max_candidates]


def _fallback_candidate_from_signal(
    *,
    agent_name: str,
    signal: DesireSubjectiveSignal,
) -> IntentCandidate | None:
    source = signal.source_desire
    interpretation = signal.subjective_interpretation
    if _subjective_need_is_satisfied_or_minor(interpretation):
        return None

    if source == "physiological_state.hunger":
        intent_text = f"{agent_name} plans to find something to eat."
    elif source == "physiological_state.thirst":
        intent_text = f"{agent_name} plans to find something to drink."
    elif source == "internal_state.fatigue":
        intent_text = f"{agent_name} plans to rest for a while."
    elif source == "physiological_state.hygiene":
        intent_text = f"{agent_name} plans to clean and tidy themselves up."
    elif source in {"internal_state.stress", "internal_state.tension"}:
        intent_text = f"{agent_name} plans to let themselves recover first."
    elif source == "mental":
        if _looks_mentally_stable(interpretation):
            return None
        intent_text = f"{agent_name} plans to do something to stabilize their state."
    elif source == "work_goal":
        intent_text = _goal_signal_to_intent(agent_name=agent_name, interpretation=interpretation)
    else:
        return None

    return IntentCandidate(
        intent_text=intent_text,
        source_desire=source,
        reason=interpretation,
    )


def _goal_signal_to_intent(*, agent_name: str, interpretation: str) -> str:
    normalized = interpretation.strip().rstrip("。")
    marker = "Unfinished serious goals:"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1].strip().rstrip("。")
    for prefix in (f"{agent_name} wants to", f"{agent_name} wants to", f"{agent_name} is going to"):
        if normalized.startswith(prefix):
            content = normalized[len(prefix) :].strip()
            if content:
                return f"{agent_name} plans to {content}."
    if normalized:
        return f"{agent_name} intends to {normalized}."
    return f"{agent_name} intends to advance this goal: {normalized}."


def _subjective_need_is_satisfied_or_minor(text: str) -> bool:
    minor_markers = (
        "Satisfied",
        "Can be ignored",
        "Not needed",
        "Not hungry",
        "Not thirsty",
        "Not tired",
        "No pressure",
        "Low stress",
        "Very relaxed",
        "Not very tense",
        "very clean",
        "fairly clean",
        "minimal impact",
        "can be handled later",
        "still able to move normally",
    )
    return any(marker in text for marker in minor_markers)


def _looks_mentally_stable(text: str) -> bool:
    stable_markers = ("smooth", "stable", "relaxed", "easygoing", "not bad", "fairly good")
    unstable_markers = ("Anxiety", "annoying", "nervous", "low", "sad", "Stress", "exhausted")
    return any(marker in text for marker in stable_markers) and not any(
        marker in text for marker in unstable_markers
    )


def _normalize_agent_name(text: str, agent_name: str) -> str:
    if text.startswith("You"):
        return f"{agent_name}{text[1:]}"
    if text.startswith(("He", "She")):
        return f"{agent_name}{text[1:]}"
    return text


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
    raise ValueError("No JSON object found in intent candidate generation response.")

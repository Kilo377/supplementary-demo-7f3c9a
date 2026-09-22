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
当下注意力触发：
{json.dumps(attention_data, ensure_ascii=False, indent=2)}
"""
    personal_context_block = ""
    if personal_context.strip():
        personal_context_block = f"""
长期个人背景：
{personal_context.strip()}
"""
    return f"""你在为一个 人类 {agent_name} 生成 intent candidates。
你需要根据 Desire 和当下注意力生成 {agent_name} 想的念头。可能是想做的事情，可能是一个欲望念头。
根据 Bratman 的BDI理论，人类intent的产生可能和自身的Desire有关，人可能同时有很多 desire。比如我想保持健康，也想睡懒觉。
此外，人也可能因为突然注意到某件事，而产生一个临时的候选 intent。
因此，同时可能会有多个 Intent 作为候选。

Desire：
{json.dumps(signal_data, ensure_ascii=False, indent=2)}
{attention_block}{personal_context_block}

值得注意的是，
1. intent 是高层念头，不是 action。 事实上，intent是一种承诺性的心理，将会影响后续的行动。
2. 不是一定要为每一个 Desire 都生成意图。
3. mental 如果只是平稳状态，可以不生成候选；如果有维持状态的明显倾向，也可以生成低干扰候选。
4. source_desire 必须使用这些格式之一：
   - physiological_state.hunger
   - physiological_state.thirst
   - physiological_state.hygiene
   - internal_state.stress
   - internal_state.tension
   - internal_state.fatigue
   - mental
   - work_goal
   - attention
5. reason 尽量直接复用对应 Desire 的 subjective_interpretation。
   - 不要把“稍微有点饿。”扩写成“有轻微饥饿感，促使{agent_name}产生想要吃东西的念头。”
   - 如果 source_desire 是 attention，reason 可以直接复用 notice_text，必要时加极短说明。
6. 最多生成 {max_candidates} 个 candidates。
7. 不要生成 Desire 主观解释或当下注意力中没有支持的候选。
8. 只返回 JSON，不要解释推理过程。

返回 JSON 格式：
{{
  "candidates": [
    {{
      "intent_text": "{agent_name}打算找点东西吃。",
      "source_desire": "physiological_state.hunger",
      "reason": "稍微有点饿。"
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
                intent_text=f"{agent_name}打算先关注一下刚刚注意到的情况。",
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
        intent_text = f"{agent_name}打算找点东西吃。"
    elif source == "physiological_state.thirst":
        intent_text = f"{agent_name}打算找点东西喝。"
    elif source == "internal_state.fatigue":
        intent_text = f"{agent_name}打算休息一会儿。"
    elif source == "physiological_state.hygiene":
        intent_text = f"{agent_name}打算清洁整理一下自己。"
    elif source in {"internal_state.stress", "internal_state.tension"}:
        intent_text = f"{agent_name}打算先让自己缓一缓。"
    elif source == "mental":
        if _looks_mentally_stable(interpretation):
            return None
        intent_text = f"{agent_name}打算做点让自己状态稳定下来的事情。"
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
    marker = "未完成的正事目标："
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1].strip().rstrip("。")
    for prefix in (f"{agent_name}想要", f"{agent_name}想", f"{agent_name}要"):
        if normalized.startswith(prefix):
            content = normalized[len(prefix) :].strip()
            if content:
                return f"{agent_name}打算{content}。"
    if normalized:
        return f"{agent_name}打算{normalized}。"
    return f"{agent_name}打算推进这个目标：{normalized}。"


def _subjective_need_is_satisfied_or_minor(text: str) -> bool:
    minor_markers = (
        "已满足",
        "可以忽略",
        "不需要",
        "不饿",
        "不渴",
        "不累",
        "没有压力",
        "压力不大",
        "很放松",
        "不太紧张",
        "很干净",
        "还算干净",
        "影响很小",
        "可以稍后处理",
        "还能正常活动",
    )
    return any(marker in text for marker in minor_markers)


def _looks_mentally_stable(text: str) -> bool:
    stable_markers = ("平稳", "稳定", "放松", "轻松", "不错", "还算好")
    unstable_markers = ("焦虑", "烦", "紧张", "低落", "难过", "压力", "疲惫")
    return any(marker in text for marker in stable_markers) and not any(
        marker in text for marker in unstable_markers
    )


def _normalize_agent_name(text: str, agent_name: str) -> str:
    if text.startswith("你"):
        return f"{agent_name}{text[1:]}"
    if text.startswith(("他", "她")):
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

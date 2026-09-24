from __future__ import annotations

import json
import random

from agent.desire import DesireState
from agent.habitual_controller.cue_extraction import CueExtractionResult
from agent.perceive import PerceiveResult
from llm.api_manager import APIManager

from .awareness_prompt import build_habitual_awareness_prompt
from .prompt_inputs import habitual_current_state_text, habitual_trigger_text
from .types import (
    AwarenessResult,
    HabitualActivationResult,
    HabitualAwarenessGateResult,
    PreparedHabitualResponse,
)


def route_habitual_by_awareness(
    activation: HabitualActivationResult,
    perception: PerceiveResult,
    context: CueExtractionResult,
    *,
    provider_name: str = "ollama",
    model: str | None = None,
    use_llm: bool = True,
    desire_state: DesireState | None = None,
    rng: random.Random | None = None,
) -> HabitualAwarenessGateResult:
    response = activation.strongest
    if response is None:
        return HabitualAwarenessGateResult(route="goal_directed")
    awareness = judge_habitual_awareness(
        response,
        perception,
        context,
        provider_name=provider_name,
        model=model,
        use_llm=use_llm,
    )
    return HabitualAwarenessGateResult(
        route="competition" if awareness.conscious else "habitual_direct",
        response=response,
        awareness=awareness,
        awareness_source="habitual_awareness" if awareness.conscious else "none",
    )


def judge_habitual_awareness(
    response: PreparedHabitualResponse,
    perception: PerceiveResult,
    context: CueExtractionResult,
    *,
    provider_name: str = "ollama",
    model: str | None = None,
    use_llm: bool = True,
) -> AwarenessResult:
    if use_llm:
        prompt = prepare_habitual_awareness_prompt(response, perception, context)
        try:
            raw = APIManager(
                provider_name=provider_name,
                task_name="agent.habitual_awareness",
            ).generate(prompt, model=model)
            parsed = json.loads(_json_text(raw))
            conscious = parsed.get("conscious")
            if not isinstance(conscious, bool):
                raise ValueError("Habitual awareness output must contain a boolean conscious field.")
            return AwarenessResult(
                conscious=conscious,
                reason=str(parsed.get("reason", "") or "").strip(),
                raw_response=raw,
            )
        except Exception:
            pass
    return _fallback_awareness(response, perception)


def prepare_habitual_awareness_prompt(
    response: PreparedHabitualResponse,
    perception: PerceiveResult,
    context: CueExtractionResult,
) -> str:
    return build_habitual_awareness_prompt(
        trigger_context_text=habitual_trigger_text(response, perception, context),
        current_state_text=habitual_current_state_text(perception, context),
        habitual_response=response,
    )


def _fallback_awareness(
    response: PreparedHabitualResponse,
    perception: PerceiveResult,
) -> AwarenessResult:
    if response.awareness_type == "conscious":
        return AwarenessResult(True, "This associative response usually enters consciousness.")
    if response.awareness_type == "unconscious":
        blocked = any(item.source == "world_feedback" for item in perception.attention.items)
        if blocked:
            return AwarenessResult(True, "Recent obstruction of actions has brought an originally unconscious response into consciousness.")
        return AwarenessResult(False, "This is an associative response that typically does not enter consciousness.")
    attentive_cues = {item.source for item in perception.attention.items if item.noteworthy}
    conscious = bool(attentive_cues)
    return AwarenessResult(
        conscious,
        "Current salient signals bring the response into consciousness." if conscious else "There are currently no salient signals bringing the response into consciousness.",
    )


def _json_text(text: str) -> str:
    stripped = str(text or "").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object in habitual awareness output.")
    return stripped[start : end + 1]

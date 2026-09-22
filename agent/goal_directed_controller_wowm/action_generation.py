from __future__ import annotations

import json
from dataclasses import dataclass

from agent.action.proposal import _clean_action_text
from agent.action.types import ActionProposalResult
from llm.api_manager import APIManager

from .prompt import build_without_world_model_action_prompt


DEBUG_WITHOUT_WORLD_MODEL_PROMPT = False


@dataclass
class WithoutWorldModelActionGenerationResult:
    candidates: tuple[ActionProposalResult, ...]
    selected_index: int
    prompt: str
    raw_response: str

    @property
    def selected_action(self) -> ActionProposalResult:
        return self.candidates[self.selected_index]

    def to_trace_dict(self) -> dict:
        return {
            "sample_window": len(self.candidates),
            "candidates": [candidate.action_text for candidate in self.candidates],
            "selected_index": self.selected_index,
            "selection_method": "llm_rank_first",
        }


def set_debug_without_world_model_prompt(enabled: bool) -> None:
    global DEBUG_WITHOUT_WORLD_MODEL_PROMPT
    DEBUG_WITHOUT_WORLD_MODEL_PROMPT = enabled


def generate_without_world_model_action(
    *,
    agent_name: str,
    intent,
    short_time_memory,
    spatial_belief,
    current_area_id: str,
    scene_name: str,
    action_sample_window: int = 1,
    provider_name: str = "ollama",
    model: str | None = None,
) -> WithoutWorldModelActionGenerationResult:
    sample_window = min(12, max(1, int(action_sample_window)))
    prompt = build_without_world_model_action_prompt(
        agent_name=agent_name,
        intent=intent,
        short_time_memory=short_time_memory,
        spatial_belief=spatial_belief,
        current_area_id=current_area_id,
        scene_name=scene_name,
        action_sample_window=sample_window,
    )
    if DEBUG_WITHOUT_WORLD_MODEL_PROMPT:
        print("=" * 72)
        print("Without World Model Action Prompt")
        print(prompt)
        print("=" * 72, flush=True)
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.goal_directed_without_world_model.action_generation",
    )
    raw = api.generate(prompt, model=model)
    candidates = _parse_action_candidates(
        raw,
        agent_name=agent_name,
        provider_name=api.provider_name,
        model=api.route.model,
        limit=sample_window,
    )
    if not candidates:
        raise ValueError("Without World Model action generation returned no valid action.")
    return WithoutWorldModelActionGenerationResult(
        candidates=tuple(candidates),
        selected_index=0,
        prompt=prompt,
        raw_response=raw,
    )


def _parse_action_candidates(
    raw: str,
    *,
    agent_name: str,
    provider_name: str,
    model: str | None,
    limit: int,
) -> list[ActionProposalResult]:
    texts: list[str] = []
    try:
        parsed = json.loads(_extract_json_text(raw))
        items = parsed.get("actions", []) if isinstance(parsed, dict) else []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                text = str(item.get("action", "") or "").strip()
            else:
                text = str(item or "").strip()
            if text:
                texts.append(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        texts = [raw]

    candidates: list[ActionProposalResult] = []
    seen: set[str] = set()
    for text in texts:
        action_text = _clean_action_text(text, agent_name=agent_name)
        comparison_key = "".join(action_text.split()).rstrip("。！？")
        if not comparison_key or comparison_key in seen:
            continue
        seen.add(comparison_key)
        candidates.append(
            ActionProposalResult(
                action_text=action_text,
                provider_name=provider_name,
                model=model,
            )
        )
        if len(candidates) >= limit:
            break
    return candidates


def _extract_json_text(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    if "```json" in stripped:
        return stripped.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in stripped:
        return stripped.split("```", 1)[1].split("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    raise ValueError("No JSON object found in action generation response.")

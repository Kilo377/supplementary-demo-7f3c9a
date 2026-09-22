from __future__ import annotations

import json

from agent.intent.state import IntentState
from agent.working_memory import WorkingMemoryFrame
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from llm.api_manager import APIManager

from .prompt import build_intuition_prompt
from .types import IntuitionError, IntuitionResult, normalize_wait_duration


DEBUG_INTUITION_PROMPT = False


def set_debug_intuition_prompt(enabled: bool) -> None:
    global DEBUG_INTUITION_PROMPT
    DEBUG_INTUITION_PROMPT = enabled


def generate_intuition(
    *,
    agent_name: str,
    intent: IntentState,
    working_memory: WorkingMemoryFrame,
    engine: PhysicsEngine,
    provider_name: str = "ollama",
    model: str | None = None,
    temperature: float | None = None,
) -> IntuitionResult:
    prompt = build_intuition_prompt(
        agent_name=agent_name,
        intent=intent,
        working_memory=working_memory,
        engine=engine,
    )
    if DEBUG_INTUITION_PROMPT:
        print("=" * 72)
        print("Intuition Prompt")
        print(prompt)
        print("=" * 72, flush=True)
    raw = APIManager(
        provider_name=provider_name,
        task_name="agent.intuition",
    ).generate(
        prompt,
        model=model,
        temperature=temperature,
    )
    try:
        parsed = json.loads(_extract_json_text(raw))
    except Exception as error:
        raise IntuitionError(
            "Failed to parse intuition LLM output.",
            raw_response=raw,
            prompt=prompt,
        ) from error
    result = parse_intuition_reply(parsed, raw_response=raw)
    if not result.thought:
        raise IntuitionError(
            "Intuition result has no thought.",
            raw_response=raw,
            prompt=prompt,
        )
    return result


def parse_intuition_reply(
    payload: dict,
    *,
    raw_response: str = "",
) -> IntuitionResult:
    route = _normalize_route(payload.get("route"))
    return IntuitionResult(
        route=route,
        thought=str(payload.get("thought", "") or "").strip(),
        target_area_id=str(payload.get("target_area_id", "") or "").strip(),
        target_area_name=str(payload.get("target_area_name", "") or "").strip(),
        chat_target=str(payload.get("chat_target", "") or "").strip(),
        wait_duration=normalize_wait_duration(
            payload.get("wait_duration"),
            default="5min" if route == "wait" else "",
        ),
        raw_response=raw_response,
    )


def _normalize_route(value) -> str:
    route = str(value or "").strip().lower()
    aliases = {
        "move": "walk",
        "move_to_area": "walk",
        "go": "walk",
        "talk": "chat",
        "speak": "chat",
    }
    route = aliases.get(route, route)
    if route in {"action", "wait", "walk", "chat", "think"}:
        return route
    return "action"


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
    raise ValueError("No JSON object found in intuition response.")

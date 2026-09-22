from __future__ import annotations

import json

from agent.intent.state import IntentState
from agent.intuition import IntuitionResult
from agent.intuition.types import normalize_wait_duration
from agent.working_memory import WorkingMemoryFrame
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from llm.api_manager import APIManager

from .prompt import build_post_think_route_prompt, build_think_prompt
from .types import ThinkError, ThinkResult


DEBUG_THINK_PROMPT = False


def set_debug_think_prompt(enabled: bool) -> None:
    global DEBUG_THINK_PROMPT
    DEBUG_THINK_PROMPT = enabled


def generate_think(
    *,
    agent_name: str,
    intent: IntentState,
    working_memory: WorkingMemoryFrame,
    intuition: IntuitionResult,
    provider_name: str = "ollama",
    model: str | None = None,
) -> ThinkResult:
    prompt = build_think_prompt(
        agent_name=agent_name,
        intent=intent,
        working_memory=working_memory,
        intuition=intuition,
    )
    if DEBUG_THINK_PROMPT:
        print("=" * 72)
        print("Think Prompt")
        print(prompt)
        print("=" * 72, flush=True)
    raw = APIManager(
        provider_name=provider_name,
        task_name="agent.think",
    ).generate(prompt, model=model)
    try:
        parsed = json.loads(_extract_json_text(raw))
    except Exception as error:
        raise ThinkError(
            "Failed to parse think LLM output.",
            raw_response=raw,
            prompt=prompt,
        ) from error

    result = ThinkResult(
        thought=str(parsed.get("thought", "") or "").strip(),
        conclusion=str(parsed.get("conclusion", "") or "").strip(),
        raw_response=raw,
    )
    if not result.thought:
        raise ThinkError(
            "Think result has no thought.",
            raw_response=raw,
            prompt=prompt,
        )
    return result


def route_after_thinking(
    *,
    agent_name: str,
    intent: IntentState,
    working_memory: WorkingMemoryFrame,
    think_result: ThinkResult,
    engine: PhysicsEngine,
    provider_name: str = "ollama",
    model: str | None = None,
) -> IntuitionResult:
    prompt = build_post_think_route_prompt(
        agent_name=agent_name,
        intent=intent,
        working_memory=working_memory,
        think_result=think_result,
        engine=engine,
    )
    if DEBUG_THINK_PROMPT:
        print("=" * 72)
        print("Post-Think Route Prompt")
        print(prompt)
        print("=" * 72, flush=True)
    raw = APIManager(
        provider_name=provider_name,
        task_name="agent.post_think_route",
    ).generate(prompt, model=model)
    try:
        parsed = json.loads(_extract_json_text(raw))
    except Exception as error:
        raise ThinkError(
            "Failed to parse post-think route LLM output.",
            raw_response=raw,
            prompt=prompt,
        ) from error

    route = _normalize_post_think_route(parsed.get("route"))
    result = IntuitionResult(
        route=route,
        thought=str(parsed.get("thought", "") or "").strip(),
        target_area_id=str(parsed.get("target_area_id", "") or "").strip(),
        target_area_name=str(parsed.get("target_area_name", "") or "").strip(),
        chat_target=str(parsed.get("chat_target", "") or "").strip(),
        wait_duration=normalize_wait_duration(
            parsed.get("wait_duration"),
            default="5min" if route == "wait" else "",
        ),
        raw_response=raw,
    )
    if not result.thought:
        raise ThinkError(
            "Post-think route result has no thought.",
            raw_response=raw,
            prompt=prompt,
        )
    return result


def _normalize_post_think_route(value) -> str:
    route = str(value or "").strip().lower()
    aliases = {
        "move": "walk",
        "move_to_area": "walk",
        "go": "walk",
        "talk": "chat",
        "speak": "chat",
        "think": "action",
    }
    route = aliases.get(route, route)
    if route in {"action", "wait", "walk", "chat"}:
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
    raise ValueError("No JSON object found in think response.")

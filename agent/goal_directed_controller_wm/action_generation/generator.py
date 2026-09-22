from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from agent.habitual_controller import PreparedHabitualResponse
from agent.intent.state import IntentState
from agent.intuition import IntuitionResult, parse_intuition_reply
from agent.working_memory import WorkingMemoryFrame
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from llm.api_manager import APIManager

from .prompt import build_action_generation_prompt
from .types import ActionCandidate, ActionGenerationResult, HabitualActionSupport


def generate_action_space(
    *,
    agent_name: str,
    intent: IntentState,
    working_memory: WorkingMemoryFrame,
    engine: PhysicsEngine,
    habitual_responses: Iterable[PreparedHabitualResponse] = (),
    action_sample_window: int = 3,
    provider_name: str = "ollama",
    model: str | None = None,
) -> ActionGenerationResult:
    prompt = build_action_generation_prompt(
        agent_name=agent_name,
        intent=intent,
        working_memory=working_memory,
        engine=engine,
        action_sample_window=action_sample_window,
    )
    raw = ""
    sampled_replies: list[IntuitionResult] = []
    error_text = ""
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.world_model.action_generation",
    )
    try:
        raw = api.generate(prompt, model=model)
        parsed = json.loads(_extract_json_text(raw))
        sampled_replies = _parse_sampled_replies(
            parsed,
            action_sample_window=action_sample_window,
            raw_response=raw,
        )
    except Exception as error:
        error_text = f"Action generation failed: {error}"

    responses = list(habitual_responses)
    return ActionGenerationResult(
        action_space=merge_action_space(
            sampled_replies,
            responses,
            agent_name=agent_name,
        ),
        sampled_intuition_replies=sampled_replies,
        provider_name=api.provider_name,
        model=api.route.model,
        prompt=prompt,
        raw_response=raw,
        error=error_text,
    )


def merge_action_space(
    sampled_intuition_replies: Iterable[IntuitionResult],
    habitual_responses: Iterable[PreparedHabitualResponse],
    *,
    agent_name: str,
) -> list[ActionCandidate]:
    entries: dict[str, dict] = {}
    order: list[str] = []

    for reply in sampled_intuition_replies:
        thought = _normalize_reply_thought(reply.thought)
        if not thought:
            continue
        normalized_reply = IntuitionResult(
            route=reply.route,
            thought=thought,
            target_area_id=reply.target_area_id,
            target_area_name=reply.target_area_name,
            chat_target=reply.chat_target,
            wait_duration=reply.wait_duration,
            raw_response=reply.raw_response,
        )
        key = _comparison_key(normalized_reply, agent_name=agent_name)
        if key not in entries:
            entries[key] = {
                "intuition_reply": normalized_reply,
                "generated_by_goal": True,
                "habitual_support": [],
            }
            order.append(key)
        else:
            entries[key]["generated_by_goal"] = True

    for response in habitual_responses:
        action_text = _normalize_action_text(response.response_text, agent_name=agent_name)
        if not action_text:
            continue
        habitual_reply = IntuitionResult(
            route="action",
            thought=action_text,
        )
        key = _comparison_key(habitual_reply, agent_name=agent_name)
        if key not in entries:
            entries[key] = {
                "intuition_reply": habitual_reply,
                "generated_by_goal": False,
                "habitual_support": [],
            }
            order.append(key)
        support = HabitualActionSupport(
            association_id=response.association_id,
            response_key=response.response_key,
            habit_strength=min(1.0, max(0.0, float(response.habit_strength))),
            activation=min(1.0, max(0.0, float(response.activation))),
            awareness_type=response.awareness_type,
        )
        existing_ids = {
            item.association_id
            for item in entries[key]["habitual_support"]
        }
        if support.association_id not in existing_ids:
            entries[key]["habitual_support"].append(support)

    return [
        ActionCandidate(
            candidate_id=_candidate_id(key),
            intuition_reply=entries[key]["intuition_reply"],
            generated_by_goal=entries[key]["generated_by_goal"],
            habitual_support=tuple(entries[key]["habitual_support"]),
        )
        for key in order
    ]


def _parse_sampled_replies(
    parsed: dict,
    *,
    action_sample_window: int,
    raw_response: str,
) -> list[IntuitionResult]:
    raw_items = parsed.get("replies")
    if raw_items is None:
        items = [parsed]
    elif isinstance(raw_items, list):
        items = raw_items
    else:
        return []
    limit = min(12, max(1, int(action_sample_window)))
    replies: list[IntuitionResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        reply = parse_intuition_reply(item, raw_response=raw_response)
        if not reply.thought or reply.route == "think":
            continue
        replies.append(reply)
        if len(replies) >= limit:
            break
    return replies


def _normalize_action_text(text: str, *, agent_name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    cleaned = cleaned.strip("-*0123456789.、 ：:")
    if not cleaned:
        return ""
    if cleaned.startswith("你"):
        cleaned = f"{agent_name}{cleaned[1:]}"
    elif cleaned.startswith(("他", "她")):
        cleaned = f"{agent_name}{cleaned[1:]}"
    elif not cleaned.startswith(agent_name):
        cleaned = f"{agent_name}{cleaned}"
    if not cleaned.endswith(("。", "！", "？")):
        cleaned = f"{cleaned}。"
    return cleaned


def _normalize_reply_thought(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _comparison_key(reply: IntuitionResult, *, agent_name: str) -> str:
    thought = str(reply.thought or "").replace(agent_name, "", 1).lower()
    normalized_thought = re.sub(r"[\s，。！？、；：,.!?;:'\"()（）\[\]【】]", "", thought)
    return "|".join(
        [
            reply.route,
            normalized_thought,
            reply.target_area_id.strip().lower(),
            reply.target_area_name.strip().lower(),
            reply.chat_target.strip().lower(),
            reply.wait_duration.strip().lower(),
        ]
    )


def _candidate_id(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"action_{digest}"


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
    raise ValueError("No JSON object found in action generation response.")

from __future__ import annotations

from agent.goal_directed_controller_wm.action_generation import (
    ActionGenerationResult,
    merge_action_space,
)
from agent.intuition import parse_intuition_reply
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from llm.api_manager import APIManager

from .iteration_prompt import build_transition_iteration_intuition_prompt
from .json_output import parse_json_object


def generate_transition_iteration_action(
    *,
    agent_name: str,
    intent_text: str,
    transition_state_text: str,
    failure_context_text: str = "",
    action_sample_window: int = 1,
    engine: PhysicsEngine,
    provider_name: str = "ollama",
    model: str | None = None,
) -> ActionGenerationResult:
    prompt = build_transition_iteration_intuition_prompt(
        agent_name=agent_name,
        intent_text=intent_text,
        transition_state_text=transition_state_text,
        failure_context_text=failure_context_text,
        action_sample_window=action_sample_window,
        engine=engine,
    )
    raw_response = ""
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.world_model.iteration_action",
    )
    try:
        raw_response = api.generate(
            prompt,
            model=model,
        )
        parsed = parse_json_object(raw_response)
        raw_items = parsed.get("replies")
        items = raw_items if isinstance(raw_items, list) else [parsed]
        replies = []
        for item in items[: max(1, min(12, int(action_sample_window)))]:
            if not isinstance(item, dict):
                continue
            reply = parse_intuition_reply(item, raw_response=raw_response)
            if reply.thought and reply.route != "think":
                replies.append(reply)
        return ActionGenerationResult(
            action_space=merge_action_space(replies, (), agent_name=agent_name),
            sampled_intuition_replies=replies,
            provider_name=api.provider_name,
            model=api.route.model,
            prompt=prompt,
            raw_response=raw_response,
        )
    except Exception as error:
        return ActionGenerationResult(
            provider_name=api.provider_name,
            model=api.route.model,
            prompt=prompt,
            raw_response=raw_response,
            error=f"State transition iteration action generation failed: {error}",
        )

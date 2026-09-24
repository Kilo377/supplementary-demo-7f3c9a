from __future__ import annotations

from dataclasses import dataclass

from agent.action import ActionProposalResult
from agent.perceive import PerceiveResult
from agent.target_resolver import (
    TargetResolution,
    TargetResolutionError,
    decision_from_target_resolution,
    resolve_target_fallback,
    resolve_target_with_llm,
)
from contextual_world.world_old.action_execution import WorldActionExecutor
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from core.action_types import AgentDecision

from .action_generation import generate_without_world_model_action


@dataclass
class WithoutWorldModelResult:
    decision: AgentDecision
    action_proposal: ActionProposalResult | None = None
    target_resolution: TargetResolution | None = None
    target_resolver_error: str = ""
    action_candidates: tuple[ActionProposalResult, ...] = ()


def run_without_world_model_controller(
    agent,
    engine: PhysicsEngine,
    perception: PerceiveResult,
    *,
    action_executor: WorldActionExecutor,
    action_sample_window: int = 1,
    provider_name: str = "ollama",
    model: str | None = None,
) -> WithoutWorldModelResult:
    pending_action_text = action_executor.pending_graph_action_text()
    if pending_action_text:
        return WithoutWorldModelResult(
            decision=AgentDecision(
                action_type="action",
                reason=pending_action_text,
                action_proposal_text=pending_action_text,
            )
        )

    intent = getattr(agent, "active_intent", None)
    if intent is None or intent.status != "active":
        return WithoutWorldModelResult(
            decision=AgentDecision(
                action_type="wait",
                reason=f"{agent.name} has not yet formed a sufficiently clear next action.",
            )
        )

    generation = generate_without_world_model_action(
        agent_name=perception.agent_name,
        intent=intent,
        short_time_memory=getattr(agent, "short_time_memory", None),
        spatial_belief=getattr(agent, "belief", None),
        current_area_id=perception.area_id,
        scene_name=_scene_name(engine),
        action_sample_window=action_sample_window,
        provider_name=provider_name,
        model=model,
    )
    proposal = generation.selected_action
    target = None
    target_error = ""
    try:
        target = resolve_target_with_llm(
            engine,
            agent_name=perception.agent_name,
            action_text=proposal.action_text,
            current_area_id=perception.area_id,
            current_area_name=perception.area_name,
            provider_name=provider_name,
            model=model,
            spatial_belief=getattr(agent, "belief", None),
            world_graph_context=action_executor.target_resolver_context(agent, engine),
        )
    except TargetResolutionError as error:
        target_error = str(error)
        target = resolve_target_fallback(
            engine,
            action_text=proposal.action_text,
            current_area_id=perception.area_id,
            spatial_belief=getattr(agent, "belief", None),
        )
        target.error = target_error
        target.raw_response = error.raw_response
        target.prompt = error.prompt
        target.duration_seconds = error.duration_seconds

    decision = decision_from_target_resolution(proposal, target)
    decision.action_generation_trace = generation.to_trace_dict()
    return WithoutWorldModelResult(
        decision=decision,
        action_proposal=proposal,
        target_resolution=target,
        target_resolver_error=target_error,
        action_candidates=generation.candidates,
    )


def _scene_name(engine: PhysicsEngine) -> str:
    return str(
        getattr(engine.home, "name", "")
        or getattr(engine.home, "node_id", "")
        or "Current scene"
    )

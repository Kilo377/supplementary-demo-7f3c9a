from __future__ import annotations

"""Central LLM routing policy for the complete cognition-to-world pipeline.

The routing table is intentionally data-only. Modules identify the kind of LLM
work they perform, while this file decides which provider/model pair performs
that work. This keeps provider experiments separate from prompt and runtime
logic.
"""

from dataclasses import dataclass
import os


OPENAI_PROFILE = "openai"
DEEPSEEK_PROFILE = "deepseek"
HYBRID_PROFILE = "hybrid"
HYBRID_DEEPSEEK_PROFILE = "hybrid_deepseek"
OLLAMA_PROFILE = "ollama"
AVAILABLE_LLM_PROFILES = (
    OPENAI_PROFILE,
    DEEPSEEK_PROFILE,
    HYBRID_PROFILE,
    HYBRID_DEEPSEEK_PROFILE,
    OLLAMA_PROFILE,
)


@dataclass(frozen=True)
class LLMRoute:
    provider_name: str
    model: str | None


@dataclass(frozen=True)
class ResolvedLLMRoute:
    task_name: str
    profile_name: str
    provider_name: str
    model: str | None
    overridden: bool


OPENAI_ROUTE = LLMRoute(
    provider_name="openai",
    model=os.getenv("LLM_ROUTING_OPENAI_MODEL", "gpt-4.1"),
)
DEEPSEEK_ROUTE = LLMRoute(
    provider_name="deepseek",
    model=os.getenv("LLM_ROUTING_DEEPSEEK_MODEL", "deepseek-v4-flash"),
)
OLLAMA_EMBEDDING_ROUTE = LLMRoute(
    provider_name="ollama",
    model=os.getenv("LLM_ROUTING_EMBEDDING_MODEL", "bge-m3:latest"),
)
OLLAMA_GENERATION_ROUTE = LLMRoute(
    provider_name="ollama",
    model=os.getenv("LLM_ROUTING_OLLAMA_MODEL", "qwen2.5:14b"),
)


# Every active LLM stage in the main agent/world pipeline is named here, even
# when several stages currently share the same provider. That makes later
# prompt-level experiments a routing-table edit rather than a code migration.
HYBRID_ROUTES: dict[str, LLMRoute] = {
    # Perception-adjacent cognition and habitual behavior.
    "agent.habitual_embedding": OLLAMA_EMBEDDING_ROUTE,
    "agent.intuition": DEEPSEEK_ROUTE,
    "agent.think": DEEPSEEK_ROUTE,
    "agent.post_think_route": DEEPSEEK_ROUTE,
    "agent.habitual_awareness": DEEPSEEK_ROUTE,
    "agent.habitual_conflict": DEEPSEEK_ROUTE,
    "agent.demonstrated_habit_learning": OPENAI_ROUTE,
    "agent.arbiter": OPENAI_ROUTE,

    # Intent generation and lifecycle.
    "agent.intent_proposal": DEEPSEEK_ROUTE,
    "agent.intent_candidate_generation": DEEPSEEK_ROUTE,
    "agent.intent_feasibility": OPENAI_ROUTE,
    "agent.intent_selection": OPENAI_ROUTE,
    "agent.intent_progress": DEEPSEEK_ROUTE,
    "agent.intent_lifecycle": OPENAI_ROUTE,

    # Action generation, grounding, and agent-side state interpretation.
    "agent.action_proposal": DEEPSEEK_ROUTE,
    "agent.target_resolver": OPENAI_ROUTE,
    "agent.self_belief_update": DEEPSEEK_ROUTE,
    "agent.desire_update": DEEPSEEK_ROUTE,
    "agent.narration": DEEPSEEK_ROUTE,

    # Goal-directed controller without a world model.
    "agent.goal_directed_without_world_model.action_generation": DEEPSEEK_ROUTE,

    # Goal-directed controller with a world model.
    "agent.world_model.action_generation": DEEPSEEK_ROUTE,
    "agent.world_model.state_transition": OPENAI_ROUTE,
    "agent.world_model.iteration_action": DEEPSEEK_ROUTE,
    "agent.world_model.intent_satisfaction": OPENAI_ROUTE,
    "agent.world_model.reward_evaluation": OPENAI_ROUTE,

    # Contextual World. Grounding and state transition stay on GPT; the final
    # natural-language projection is high-volume and can use DeepSeek.
    "world.node_support": OPENAI_ROUTE,
    "world.interaction_focus": OPENAI_ROUTE,
    "world.state_transition": OPENAI_ROUTE,
    "world.feedback_summary": DEEPSEEK_ROUTE,

    # Run-level analysis does not alter the simulated world state.
    "evaluation.cumulative_reward": DEEPSEEK_ROUTE,
}


# Cost-oriented hybrid profile. Keep the original hybrid table stable so old
# experiment records remain reproducible, while moving high-frequency
# structured prediction and grounding calls to DeepSeek. OpenAI still handles
# the actual World transition, Intent lifecycle, rollout reward comparison,
# and demonstrated-habit admission.
HYBRID_DEEPSEEK_ROUTES: dict[str, LLMRoute] = {
    **HYBRID_ROUTES,
    "agent.arbiter": DEEPSEEK_ROUTE,
    "agent.intent_feasibility": DEEPSEEK_ROUTE,
    "agent.intent_selection": DEEPSEEK_ROUTE,
    "agent.target_resolver": DEEPSEEK_ROUTE,
    "agent.world_model.state_transition": DEEPSEEK_ROUTE,
    "agent.world_model.intent_satisfaction": DEEPSEEK_ROUTE,
    "world.node_support": DEEPSEEK_ROUTE,
    "world.interaction_focus": DEEPSEEK_ROUTE,
}


HYBRID_ROUTE_TABLES: dict[str, dict[str, LLMRoute]] = {
    HYBRID_PROFILE: HYBRID_ROUTES,
    HYBRID_DEEPSEEK_PROFILE: HYBRID_DEEPSEEK_ROUTES,
}


_active_profile = os.getenv("LLM_ROUTING_PROFILE", OPENAI_PROFILE).strip().lower()
if _active_profile not in AVAILABLE_LLM_PROFILES:
    _active_profile = OPENAI_PROFILE


def configure_llm_routing(profile_name: str) -> None:
    """Select one routing profile for subsequent LLM calls in this process."""
    normalized = str(profile_name or OPENAI_PROFILE).strip().lower()
    if normalized not in AVAILABLE_LLM_PROFILES:
        raise ValueError(
            f"Unsupported LLM routing profile: {profile_name}. "
            f"Expected one of {AVAILABLE_LLM_PROFILES}."
        )
    global _active_profile
    _active_profile = normalized


def current_llm_profile() -> str:
    return _active_profile


def resolve_llm_route(
    *,
    task_name: str,
    requested_provider: str,
    requested_model: str | None,
) -> ResolvedLLMRoute:
    """Resolve a task without silently rerouting unnamed/unknown call sites."""
    task = str(task_name or "").strip()
    profile = current_llm_profile()
    route: LLMRoute | None = None

    # DeepSeek has no embedding endpoint compatible with this project. Keep the
    # established local embedding backend under every generation profile.
    if task == "agent.habitual_embedding":
        route = OLLAMA_EMBEDDING_ROUTE
    elif task and profile in HYBRID_ROUTE_TABLES:
        route = HYBRID_ROUTE_TABLES[profile].get(task)
    elif task and profile == OPENAI_PROFILE:
        route = OPENAI_ROUTE
    elif task and profile == DEEPSEEK_PROFILE:
        route = DEEPSEEK_ROUTE
    elif task and profile == OLLAMA_PROFILE:
        route = OLLAMA_GENERATION_ROUTE

    if route is None:
        return ResolvedLLMRoute(
            task_name=task,
            profile_name=profile,
            provider_name=requested_provider,
            model=requested_model,
            overridden=False,
        )
    return ResolvedLLMRoute(
        task_name=task,
        profile_name=profile,
        provider_name=route.provider_name,
        model=route.model,
        overridden=True,
    )


def rate_limit_fallback_route(
    route: ResolvedLLMRoute,
) -> ResolvedLLMRoute | None:
    """Return the other cloud provider only for hybrid generation calls."""
    if route.profile_name not in HYBRID_ROUTE_TABLES:
        return None
    if route.provider_name == "openai":
        fallback = DEEPSEEK_ROUTE
    elif route.provider_name == "deepseek":
        fallback = OPENAI_ROUTE
    else:
        return None
    return ResolvedLLMRoute(
        task_name=route.task_name,
        profile_name=route.profile_name,
        provider_name=fallback.provider_name,
        model=fallback.model,
        overridden=True,
    )


def hybrid_route_table(
    profile_name: str = HYBRID_PROFILE,
) -> dict[str, dict[str, str | None]]:
    """Return a serialization-friendly routing table for CLI/debug output."""
    routes = HYBRID_ROUTE_TABLES.get(profile_name)
    if routes is None:
        raise ValueError(f"Unsupported hybrid routing profile: {profile_name}")
    return {
        task_name: {
            "provider_name": route.provider_name,
            "model": route.model,
        }
        for task_name, route in routes.items()
    }

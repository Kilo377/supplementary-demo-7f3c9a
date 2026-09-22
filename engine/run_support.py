"""Shared construction and lifecycle helpers for the GEMS command-line runner."""

from __future__ import annotations

import json
from pathlib import Path

from agent.agent import Agent
from agent.avatar import Avatar, load_avatar
from agent.desire.desire_update import (
    DESIRE_UPDATE_EVERY_STEP,
    DESIRE_UPDATE_ON_INTENT_END,
    apply_desire_update,
)
from engine.interaction_engine import EnvironmentInteractionEngine


AVATAR_DIR = Path(__file__).resolve().parents[1] / "agent" / "avatars"
DEFAULT_AGENT_AVATAR = "michael_anderson"
DEFAULT_WORLD_FAILURE_EXPORT = str(
    Path(__file__).resolve().parents[1]
    / "contextual_world"
    / "failures"
    / "world_failure_cases.json"
)


def load_agent_avatar(args) -> Avatar:
    if args.agent_avatar and args.agent_avatar_file:
        raise ValueError("Use either --agent-avatar or --agent-avatar-file, not both.")
    avatar_path = (
        Path(args.agent_avatar_file)
        if args.agent_avatar_file
        else AVATAR_DIR / f"{args.agent_avatar or DEFAULT_AGENT_AVATAR}.json"
    )
    return load_avatar(
        avatar_path,
        fallback_name="Agent",
        override_name=args.agent_name.strip(),
    )


def build_agent_from_args(args, *, agent_start: tuple[float, float]) -> Agent:
    avatar = load_agent_avatar(args)
    degree_override = getattr(args, "degree_of_model_based_control", None)
    if degree_override is not None and not 0.0 <= degree_override <= 1.0:
        raise ValueError("--degree-of-model-based-control must be between 0 and 1.")
    agent = Agent(
        node_id="agent_01",
        name=avatar.name,
        center=agent_start,
        personality=avatar.personality,
        degree_of_model_based_control=(
            avatar.degree_of_model_based_control
            if degree_override is None
            else degree_override
        ),
        desire_state=avatar.desire_state,
        base_desire_state=avatar.desire_state,
    )
    avatar.world_agent_state.apply_to_agent(agent, update_base=True)
    agent.long_term_memory.agent_name = agent.name
    if avatar.biography:
        agent.long_term_memory.remember_biography(avatar.biography)
    agent.long_term_memory.cue_memory.habit_strength_threshold = min(
        1.0, max(0.0, avatar.habit_strength_threshold)
    )
    for record in avatar.cue_memory_records:
        agent.long_term_memory.cue_memory.remember(record)
    return agent


def update_desire_after_intent(
    runtime: EnvironmentInteractionEngine, cycle_results
) -> dict | None:
    if not cycle_results or cycle_results[-1].intent_status == "active":
        return None
    final_result = cycle_results[-1]
    _, record = apply_desire_update(
        agent=runtime.agent,
        intent_text=final_result.intent_text,
        intent_status=final_result.intent_status,
        update_mode=DESIRE_UPDATE_ON_INTENT_END,
        provider_name=runtime.loop.provider_name,
        model=runtime.loop.model,
        context_count=20,
    )
    return record


def update_desire_after_step(runtime: EnvironmentInteractionEngine, result) -> dict:
    _, record = apply_desire_update(
        agent=runtime.agent,
        intent_text=result.intent_text,
        intent_status=result.intent_status or "active",
        update_mode=DESIRE_UPDATE_EVERY_STEP,
        provider_name=runtime.loop.provider_name,
        model=runtime.loop.model,
        context_count=1,
    )
    return record


def review_intent_after_turn(runtime: EnvironmentInteractionEngine, result) -> None:
    reflection = runtime.review_active_intent()
    if reflection is None:
        return
    result.intent_status = reflection.intent_status
    runtime.agent.short_time_memory.status = reflection.intent_status
    if reflection.intent_status != "active":
        result.intent_lifecycle_reason = reflection.reason or "No reason returned."


def behavior_control_label(result) -> str:
    if str(getattr(result, "execution_kind", "") or "") == "move_precondition":
        return "World Positioning"
    mode = str(getattr(result, "arbiter_mode", "") or "").strip()
    has_habit = bool(str(getattr(result, "habitual_response_key", "") or "").strip())
    if has_habit and mode in {"fast_path", "habitual"}:
        return "Habitual"
    if has_habit and mode in {"combine", "sequence"}:
        return "Goal-Directed + Habitual"
    return "Goal-Directed"


def print_habitual_trace(result) -> None:
    gate = getattr(result, "emprical_gate_trace", {}) or {}
    if gate:
        print(
            "Psychological-state gate: "
            f"F={gate.get('suppression_probability', 0):.3f}, "
            f"route={gate.get('route', '')}"
        )
    decision = str(getattr(result, "habitual_gate_decision", "") or "").strip()
    if decision:
        print(
            "Habitual retrieval: "
            f"{decision}, p={getattr(result, 'habitual_gate_probability', 0):.2f}"
        )
    response = str(getattr(result, "habitual_response_text", "") or "").strip()
    if response:
        print(f"Habitual response: {response}")


def print_action_generation_trace(result) -> None:
    trace = dict(getattr(result, "action_generation_trace", {}) or {})
    candidates = list(trace.get("candidates", []) or [])
    if len(candidates) <= 1:
        return
    selected_index = int(trace.get("selected_index", 0) or 0)
    print("Candidate actions:")
    for index, action in enumerate(candidates):
        marker = " [selected]" if index == selected_index else ""
        print(f"  {index + 1}. {action}{marker}")


def print_desire_update(record: dict) -> None:
    print("Desire update:")
    print(json.dumps(record, ensure_ascii=False, indent=2))


def print_step(result, *, desire_update: dict | None = None, **_) -> None:
    print(f"Engine step {result.step_id}: {behavior_control_label(result)}")
    print_habitual_trace(result)
    print_action_generation_trace(result)
    print(result.execution_narration or result.action_text)
    if desire_update is not None:
        print_desire_update(desire_update)


def collect_world_failure_report(runtime: EnvironmentInteractionEngine) -> dict:
    output = runtime.loop.action_executor.close_world_failure_session()
    if output is None:
        return {
            "schema_version": "world_failure_cases.v3",
            "failure_count": 0,
            "failure_counts_by_module": {},
            "failures": [],
            "output_path": "",
        }
    report = json.loads(output.read_text(encoding="utf-8"))
    report["output_path"] = str(output)
    return report

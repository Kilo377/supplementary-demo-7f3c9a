from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

from engine.interaction_engine import DEFAULT_AGENT_START
from engine.run_support import (
    DEFAULT_WORLD_FAILURE_EXPORT,
    build_agent_from_args,
    behavior_control_label,
    collect_world_failure_report,
    print_desire_update,
    print_action_generation_trace,
    print_habitual_trace,
    print_step,
    review_intent_after_turn,
    update_desire_after_intent,
    update_desire_after_step,
)
from agent.belief.self_state import (
    ElementReference,
    MovementPoint,
    MovementTrace,
    SelfState,
    SelfStateUpdateInput,
)
from agent.belief.short_time_memory import decide_short_time_memory_encoding
from agent.action import set_debug_action_proposal_prompt
from agent.goal_directed_controller_wowm import set_debug_without_world_model_prompt
from agent.interaction_loop import (
    COMPUTATION_ARCHITECTURES,
    HABITUAL_AWARENESS_MODES,
    set_debug_target_resolver,
)
from agent.goal_directed_controller_wm.arbiter import ARBITER_MODES, SELECTION_METHODS
from agent.goal_directed_controller_wm.planning import PLANNING_STRATEGIES
from agent.goal_directed_controller_wm.state_transition import WORLD_MODEL_MODES
from agent.habitual_controller.demonstrated_learning import (
    capture_demonstrated_action_observation,
    learn_demonstrated_habits,
    load_demonstrated_habit_memory,
    save_demonstrated_habit_memory,
    successful_semantic_action_text,
)
from agent.intuition import set_debug_intuition_prompt
from agent.think import set_debug_think_prompt
from agent.intent.state import IntentState
from agent.belief.short_time_memory import ShortTermMemory
from agent.desire.desire_update import (
    DEFAULT_DESIRE_UPDATE_MODE,
    DESIRE_UPDATE_EVERY_STEP,
    DESIRE_UPDATE_MODES,
    DESIRE_UPDATE_ON_INTENT_END,
)
from engine.interaction_engine import EnvironmentInteractionEngine
from evaluation.cumulative_reward import (
    CumulativeRewardResult,
    evaluate_cumulative_reward,
)
from evaluation.behavior_log import DEFAULT_BEHAVIOR_LOG, write_behavior_log
from visualization.intent_run_visualizer import OUTPUT_HTML, write_intent_run_html
from contextual_world.structure import available_scene_names, build_scene
from llm.api_manager import SUPPORTED_PROVIDER_NAMES
from llm.routing import AVAILABLE_LLM_PROFILES, configure_llm_routing


DEFAULT_INTENT_TEMPLATE = "{agent_name} plans to go to the kitchen to fill their stomach."
DEFAULT_DEMONSTRATED_HABIT_DIR = (
    Path(__file__).resolve().parent / "demonstrated_habits"
)
DEMONSTRATED_HABIT_MODES = ("off", "learn", "use", "learn_and_use")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inject one fixed intent and observe only the action sequence it drives.",
    )
    parser.add_argument(
        "--intent",
        default="",
        help="High-level intent to inject into the agent. Defaults to a simple kitchen/food intent.",
    )
    parser.add_argument(
        "--scene",
        default="unity_home",
        choices=available_scene_names(),
        help="Scene name to load.",
    )
    parser.add_argument(
        "--agent-name",
        default="",
        help="Optional override for the human agent name.",
    )
    parser.add_argument(
        "--agent-avatar",
        default="",
        help="Named avatar under agent/avatars, without the .json suffix.",
    )
    parser.add_argument(
        "--agent-avatar-file",
        default="",
        help="Optional JSON avatar file.",
    )
    parser.add_argument(
        "--degree-of-model-based-control",
        type=float,
        default=None,
        help="Optional runtime override for the Avatar's rho parameter in [0, 1].",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Force an intent continuation check every N action turns.",
    )
    parser.add_argument(
        "--safety-turns",
        "--turns",
        dest="safety_turns",
        type=int,
        default=30,
        help="Hard safety cap for this single-intent run, used only to prevent infinite runs.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=SUPPORTED_PROVIDER_NAMES,
        help="LLM provider name.",
    )
    from agent.emprical_gate.cli import add_arguments
    add_arguments(parser)
    parser.add_argument(
        "--habitual-retrieval-probability",
        type=float,
        default=0.9,
        help="Probability of checking Cue Memory before Goal-Directed.",
    )
    parser.add_argument(
        "--habitual-activation-threshold",
        type=float,
        default=0.35,
        help="Minimum dimension-embedding cue similarity admitted to the Habitual response race.",
    )
    parser.add_argument(
        "--habitual-awareness-mode",
        default="llm",
        choices=HABITUAL_AWARENESS_MODES,
        help="Use the LLM awareness judgment or trust each Habit Memory awareness_type label.",
    )
    parser.add_argument(
        "--unconscious-habit-memory-probability",
        type=float,
        default=0.2,
        help="Probability that an unconscious Habitual action becomes recallable short-term memory.",
    )
    parser.add_argument(
        "--demonstrated-habit-mode",
        default="off",
        choices=DEMONSTRATED_HABIT_MODES,
        help="Learn conscious habits from this Goal-Directed run, use a saved demonstrated pool, or both.",
    )
    parser.add_argument(
        "--demonstrated-habit-file",
        default="",
        help="Separate JSON pool used for demonstrated habits. Defaults to one file per avatar.",
    )
    parser.add_argument(
        "--demonstrated-habit-strength",
        type=float,
        default=0.90,
        help="Initial established strength assigned to LLM-approved demonstrated habits.",
    )
    parser.add_argument(
        "--debug-demonstrated-habit-prompt",
        action="store_true",
        default=False,
        help="Print the batch LLM prompt used to select demonstrated habits.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override. When omitted, the selected provider uses its default model.",
    )
    parser.add_argument(
        "--llm-profile",
        "--llm-setting",
        dest="llm_profile",
        default="openai",
        choices=AVAILABLE_LLM_PROFILES,
        help="LLM routing profile, including standard and DeepSeek-heavy hybrid modes.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Write a single-file HTML replay after the run.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Do not write the HTML replay.",
    )
    parser.add_argument(
        "--visualization-out",
        default=OUTPUT_HTML,
        help="Output path for --visualize.",
    )
    parser.add_argument(
        "--world-failure-export-out",
        default=DEFAULT_WORLD_FAILURE_EXPORT,
        help="Base JSON path for this run's timestamped Contextual World failure report.",
    )
    parser.add_argument(
        "--interaction-reach",
        type=float,
        default=0.0,
        help=(
            "Experimental interaction distance in scene units. 0 keeps normal spatial checks; "
            "a large value such as 100 effectively disables precise within-area reach checks."
        ),
    )
    parser.add_argument(
        "--disable-spatial-gating",
        action="store_true",
        default=True,
        help=(
            "Use the default easy spatial mode: skip precise within-area positioning and relocate "
            "to a valid target-area anchor when normal cross-area pathfinding fails."
        ),
    )
    parser.add_argument(
        "--strict-spatial-gating",
        action="store_false",
        dest="disable_spatial_gating",
        help="Restore strict reach, interaction-position, and pathfinding checks.",
    )
    parser.add_argument(
        "--behavior-log-out",
        default=DEFAULT_BEHAVIOR_LOG,
        help="JSON path for the simulated time and executed behavior log.",
    )
    parser.add_argument(
        "--debug-action-prompt",
        action="store_true",
        default=False,
        help="Print action proposal prompt inputs while running the sequence.",
    )
    parser.add_argument(
        "--no-debug-action-prompt",
        action="store_false",
        dest="debug_action_prompt",
        help="Print the normal action execution logs instead of prompt debug output.",
    )
    parser.add_argument(
        "--debug-intuition-prompt",
        action="store_true",
        default=False,
        help="Print the intuition routing prompt while running the sequence.",
    )
    parser.add_argument(
        "--debug-think-prompt",
        action="store_true",
        default=False,
        help="Print the think and post-think route prompts while running the sequence.",
    )
    parser.add_argument(
        "--intuition-mode",
        default="active",
        choices=["active", "shadow", "off"],
        help="active lets Intuition route before Action Proposal; shadow logs it without routing; off disables it.",
    )
    parser.add_argument(
        "--computation-architecture",
        default="world_model",
        choices=COMPUTATION_ARCHITECTURES,
        help="Goal-Directed computation architecture.",
    )
    parser.add_argument(
        "--action-sample-window",
        type=int,
        default=None,
        help="Candidate actions generated in one call; defaults to 1 for without_world_model and 3 for world_model.",
    )
    parser.add_argument(
        "--world-model-mode",
        default="complex",
        choices=WORLD_MODEL_MODES,
        help="State transition backend: complex executes a forked Contextual World; simple uses direct LLM prediction.",
    )
    parser.add_argument("--iteration-number", type=int, default=3)
    parser.add_argument(
        "--planning-strategy",
        default="greedy",
        choices=PLANNING_STRATEGIES,
        help="World Model planning search: greedy rollout or beam search.",
    )
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--planning-branching-factor", type=int, default=3)
    parser.add_argument("--mcts-simulations", type=int, default=12)
    parser.add_argument("--mcts-exploration-constant", type=float, default=2 ** 0.5)
    parser.add_argument("--recent-experience-count", type=int, default=8)
    parser.add_argument(
        "--selection-method",
        default="greedy",
        choices=SELECTION_METHODS,
    )
    parser.add_argument("--selection-temperature", type=float, default=0.2)
    parser.add_argument("--selection-epsilon", type=float, default=0.1)
    parser.add_argument(
        "--arbiter-mode",
        default="weighted",
        choices=ARBITER_MODES,
        help="World Model cross-controller arbitration: weighted or 50/50 random-controller ablation.",
    )
    parser.add_argument(
        "--arbiter-random-seed",
        type=int,
        default=None,
        help="Optional reproducible seed for random-controller arbitration.",
    )
    parser.add_argument(
        "--desire-update-mode",
        default=DEFAULT_DESIRE_UPDATE_MODE,
        choices=DESIRE_UPDATE_MODES,
        help="When Desire changes: every_step (default) or intent_end.",
    )
    parser.add_argument(
        "--verbose-output",
        action="store_true",
        default=False,
        help="Print the older full per-turn diagnostics, self state, and final memory summary.",
    )
    return parser


def print_sequence_summary(results) -> None:
    print("=" * 72)
    print("Action Sequence Summary")
    if not results:
        print("No action turns were produced.")
        return

    for result in results:
        decision = getattr(result, "decision", None)
        intuition_thought = (
            str(getattr(result, "intuition_thought", "") or "").strip()
            or str(getattr(decision, "intuition_thought", "") or "").strip()
        )
        intuition_route = (
            str(getattr(result, "intuition_route", "") or "").strip()
            or str(getattr(decision, "intuition_route", "") or "").strip()
        )
        direct_intuition_route = (
            intuition_thought
            and intuition_route in {"wait", "walk", "chat"}
            and result.action_proposal_text == intuition_thought
        )
        if intuition_thought:
            route_label = f"[{intuition_route}] " if intuition_route else ""
            print(f"{result.intent_action_index}. intuition: {route_label}{intuition_thought}")
        think_thought = str(getattr(result, "think_thought", "") or "").strip()
        think_conclusion = str(getattr(result, "think_conclusion", "") or "").strip()
        if think_thought:
            print(f"{result.intent_action_index}. think: {think_thought}")
            if think_conclusion:
                print(f"   conclusion: {think_conclusion}")
        shadow_intuition_thought = (
            str(getattr(result, "shadow_intuition_thought", "") or "").strip()
            or str(getattr(decision, "shadow_intuition_thought", "") or "").strip()
        )
        if shadow_intuition_thought:
            shadow_intuition_route = (
                str(getattr(result, "shadow_intuition_route", "") or "").strip()
                or str(getattr(decision, "shadow_intuition_route", "") or "").strip()
            )
            route_label = f"[{shadow_intuition_route}] " if shadow_intuition_route else ""
            print(f"{result.intent_action_index}. shadow intuition: {route_label}{shadow_intuition_thought}")
        if not direct_intuition_route:
            print(
                f"{result.intent_action_index}. [{behavior_control_label(result)}] "
                f"{result.action_proposal_text or '(no action proposal)'}"
            )
        if result.execution_narration:
            print(f"   -> {result.execution_narration}")
        else:
            print(f"   -> {result.action_text}")
    print(f"Final intent status: {results[-1].intent_status or 'active'}")


def print_concise_step(result, *, desire_update: dict | None = None) -> None:
    print("=" * 72)
    if getattr(result, "counts_as_intent_action", True):
        print(f"Turn {result.intent_action_index}")
    else:
        print(f"Positioning for Turn {result.intent_action_index}")
    print(f"Behavior Control: {behavior_control_label(result)}")
    print_habitual_trace(result)
    intuition_route = str(getattr(result, "intuition_route", "") or "").strip()
    intuition_thought = str(getattr(result, "intuition_thought", "") or "").strip()
    direct_intuition_route = (
        intuition_thought
        and intuition_route in {"wait", "walk", "chat"}
        and result.action_proposal_text == intuition_thought
    )
    if result.action_proposal_text and not direct_intuition_route:
        print_action_generation_trace(result)
        print("Action Proposal:")
        print(result.action_proposal_text)
    if intuition_thought:
        print("Intuition:")
        wait_duration = str(getattr(result, "wait_duration", "") or "").strip()
        route_value = f"{intuition_route}, {wait_duration}" if intuition_route == "wait" and wait_duration else intuition_route
        route_label = f"[{route_value}] " if route_value else ""
        print(f"{route_label}{intuition_thought}")
    think_thought = str(getattr(result, "think_thought", "") or "").strip()
    think_conclusion = str(getattr(result, "think_conclusion", "") or "").strip()
    if think_thought:
        print("Think:")
        print(think_thought)
        if think_conclusion:
            print(f"Conclusion: {think_conclusion}")
    shadow_route = str(getattr(result, "shadow_intuition_route", "") or "").strip()
    shadow_thought = str(getattr(result, "shadow_intuition_thought", "") or "").strip()
    if shadow_thought:
        print("Shadow Intuition:")
        route_label = f"[{shadow_route}] " if shadow_route else ""
        print(f"{route_label}{shadow_thought}")
    world_feedback = _world_feedback_text(result)
    if world_feedback:
        print("World Feedback:")
        print(world_feedback)
    lifecycle_reason = str(getattr(result, "intent_lifecycle_reason", "") or "").strip()
    if lifecycle_reason:
        print("Intent Lifecycle:")
        print(f"active -> {result.intent_status}")
        print(f"Reason: {lifecycle_reason}")
    if desire_update is not None:
        print_desire_update(desire_update)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _find_area_for_element(runtime: EnvironmentInteractionEngine, element_id: str):
    for area in runtime.home.areas:
        if area.find_element(element_id) is not None:
            return area
    return None


def _element_reference(runtime: EnvironmentInteractionEngine, element_id: str) -> ElementReference | None:
    element = runtime.home.find_element(element_id)
    if element is None:
        return None
    area = _find_area_for_element(runtime, element_id)
    return ElementReference(
        element_id=element.node_id,
        element_name=element.name,
        area_id=area.node_id if area is not None else "",
        area_name=area.name if area is not None else "",
    )


def _nearest_anchor(runtime: EnvironmentInteractionEngine, position: tuple[float, float]) -> ElementReference | None:
    nearby = runtime.physics.get_nearby_elements(center=position, radius=1.2)
    if not nearby:
        return None
    nearby.sort(
        key=lambda item: (
            (item.element.center[0] - position[0]) ** 2
            + (item.element.center[1] - position[1]) ** 2
        )
    )
    return _element_reference(runtime, nearby[0].element.node_id)


def _anchor_from_summary(runtime: EnvironmentInteractionEngine, summary: str) -> ElementReference | None:
    if "in front of" not in summary:
        return None
    after = summary.rsplit("in front of", 1)[1].strip()
    if not after:
        return None
    for area in runtime.home.areas:
        for element in area.elements:
            if after.startswith(element.name):
                return _element_reference(runtime, element.node_id)
    return None


def _movement_point(
    runtime: EnvironmentInteractionEngine,
    position: tuple[float, float],
    *,
    summary: str = "",
) -> MovementPoint:
    area = runtime.physics.get_area_for_point(*position)
    anchor = _anchor_from_summary(runtime, summary) or _nearest_anchor(runtime, position)
    return MovementPoint(
        position=position,
        area_id=area.node_id if area is not None else "",
        area_name=area.name if area is not None else "",
        anchor_element_id=anchor.element_id if anchor is not None else "",
        anchor_element_name=anchor.element_name if anchor is not None else "",
        anchor_area_id=anchor.area_id if anchor is not None else "",
        anchor_area_name=anchor.area_name if anchor is not None else "",
    )


def _interacted_elements(runtime: EnvironmentInteractionEngine, result) -> list[ElementReference]:
    target_ids = _unique(
        [
            result.target_element_id or "",
            result.navigation_target_element_id or "",
            result.secondary_target_element_id or "",
            *[change.element_id for change in result.world_changes],
        ]
    )
    references = []
    for element_id in target_ids:
        reference = _element_reference(runtime, element_id)
        if reference is not None:
            references.append(reference)
    return references


def update_self_state(
    runtime: EnvironmentInteractionEngine,
    result,
    *,
    from_position: tuple[float, float],
) -> None:
    movement = MovementTrace(
        from_point=_movement_point(runtime, from_position),
        to_point=_movement_point(runtime, result.final_position, summary=result.action_text),
        summary=result.action_text,
    )
    runtime.agent.self_state.update(
        SelfStateUpdateInput(
            step_id=result.step_id,
            intent_text=result.intent_text,
            action_proposal_text=result.action_proposal_text,
            interacted_elements=_interacted_elements(runtime, result),
            environment_feedback=result.execution_narration or result.action_text,
            movement=movement,
            world_changes=[
                {
                    "area_id": change.area_id,
                    "element_id": change.element_id,
                    "element_name": change.element_name,
                    "old_physical_status": change.old_physical_status,
                    "new_physical_status": change.new_physical_status,
                    "old_evolution_status": change.old_evolution_status,
                    "new_evolution_status": change.new_evolution_status,
                    "old_interaction_status": change.old_interaction_status,
                    "new_interaction_status": change.new_interaction_status,
                    "old_state_details": change.old_state_details,
                    "new_state_details": change.new_state_details,
                }
                for change in result.world_changes
            ],
        )
    )
    runtime.agent.self_state.update_self_belief(
        agent_name=runtime.agent.name,
        provider_name=runtime.loop.provider_name,
        model=runtime.loop.model,
    )
    result.self_belief = runtime.agent.self_state.self_belief


def print_self_state(self_state: SelfState) -> None:
    print("Self State:")
    print(json.dumps(self_state.to_dict(), ensure_ascii=False, indent=2))
    print(flush=True)


def remember_turn(runtime: EnvironmentInteractionEngine, result) -> None:
    if not getattr(result, "counts_as_intent_action", True):
        result.short_time_memory_encoded = False
        result.short_time_memory_encoding_probability = 0.0
        result.short_time_memory_encoding_sample = None
        return
    area = runtime.physics.get_area_for_point(*result.final_position)
    target_ids = _unique(
        [
            result.target_element_id or "",
            result.navigation_target_element_id or "",
            result.secondary_target_element_id or "",
            *[change.element_id for change in result.world_changes],
        ]
    )
    target_names = []
    for element_id in target_ids:
        element = runtime.home.find_element(element_id)
        if element is not None:
            target_names.append(element.name)
    intuition_route = str(getattr(result, "intuition_route", "") or "").strip()
    intuition_thought = str(getattr(result, "intuition_thought", "") or "").strip()
    intuition_mode = "active" if intuition_thought else ""
    if not intuition_thought:
        intuition_route = str(getattr(result, "shadow_intuition_route", "") or "").strip()
        intuition_thought = str(getattr(result, "shadow_intuition_thought", "") or "").strip()
        intuition_mode = "shadow" if intuition_thought else ""
    encoding = decide_short_time_memory_encoding(
        unconscious_habit=(
            bool(str(getattr(result, "habitual_response_key", "") or "").strip())
            and getattr(result, "habitual_awareness", "") == "unconscious"
        ),
        probability=runtime.loop.unconscious_habit_memory_probability,
    )
    result.short_time_memory_encoded = encoding.recallable
    result.short_time_memory_encoding_probability = encoding.probability
    result.short_time_memory_encoding_sample = encoding.sample
    runtime.agent.short_time_memory.remember(
        step_id=result.step_id,
        area_id=area.node_id if area is not None else "",
        area_name=area.name if area is not None else "Unknown area",
        perceived_summary=result.perception_notice_text,
        intended_action=result.action_proposal_text,
        experienced_result=_memory_experienced_result(result),
        intuition_route=intuition_route,
        intuition_thought=intuition_thought,
        intuition_mode=intuition_mode,
        time_text=_time_memory_label(runtime.agent),
        interacted_element_ids=target_ids,
        interacted_element_names=_unique(target_names),
        learned_facts=[],
        recallable=encoding.recallable,
        encoding_probability=encoding.probability,
        encoding_sample=encoding.sample,
    )


def _memory_experienced_result(result) -> str:
    feedback_payload = getattr(result, "environment_feedback", {}) or {}
    if isinstance(feedback_payload, dict):
        summary = str(feedback_payload.get("perception_summary", "") or "").strip()
        if summary:
            return summary
    return result.execution_narration or result.action_text


def _time_memory_label(agent) -> str:
    time_belief = getattr(agent, "time_belief", None)
    formatter = getattr(time_belief, "format_short_label", None)
    if callable(formatter):
        return str(formatter() or "").strip()
    return ""


def _world_feedback_text(result) -> str:
    feedback_payload = getattr(result, "environment_feedback", {}) or {}
    if isinstance(feedback_payload, dict):
        summary = str(feedback_payload.get("perception_summary", "") or "").strip()
        if summary:
            return summary
    return str(getattr(result, "execution_narration", "") or getattr(result, "action_text", "") or "").strip()


def _world_graph_warnings_from_result(result) -> list[str]:
    feedback = getattr(result, "environment_feedback", {}) or {}
    if not isinstance(feedback, dict):
        return []
    report = feedback.get("graph_transition_report") or {}
    if not isinstance(report, dict):
        return []
    return [str(item) for item in (report.get("warnings") or []) if str(item)]


def print_short_time_memory(memory: ShortTermMemory) -> None:
    print("=" * 72)
    print("Short Time Memory")
    print(f"Intent: {memory.intent_text} [{memory.status}]")
    episodes = [
        episode
        for episode in memory.episodes
        if getattr(episode, "recallable", True)
    ]
    if not episodes:
        print("None yet.")
        return

    for episode in episodes:
        print(episode.header_text())
        if episode.perceived_summary:
            print(f"   notice: {episode.perceived_summary}")
        if episode.intended_action:
            print(f"   intended: {episode.intended_action}")
        if episode.experienced_result:
            print(f"   experienced: {episode.experienced_result}")
        if getattr(episode, "intuition_thought", ""):
            mode = f"{episode.intuition_mode}/" if getattr(episode, "intuition_mode", "") else ""
            route = f"[{mode}{episode.intuition_route}] " if getattr(episode, "intuition_route", "") or mode else ""
            print(f"   intuition: {route}{episode.intuition_thought}")
        if episode.interacted_element_names:
            print(f"   interacted: {', '.join(episode.interacted_element_names)}")
        elif episode.interacted_element_ids:
            print(f"   interacted: {', '.join(episode.interacted_element_ids)}")
        if episode.learned_facts:
            print("   learned:")
            for fact in episode.learned_facts:
                print(f"   - {fact}")


def print_short_time_memory_prompt(memory: ShortTermMemory) -> None:
    print("=" * 72)
    print("Short Time Memory Prompt")
    print(memory.format_for_prompt(count=50))


def finish_run(
    runtime: EnvironmentInteractionEngine,
    *,
    args,
    initial_position: tuple[float, float],
    results: list,
    initial_desire_state: dict,
    desire_updates: list[dict],
    cumulative_reward: CumulativeRewardResult,
    world_failure_report: dict,
) -> None:
    runtime.agent.short_time_memory.status = results[-1].intent_status if results else "active"
    if args.verbose_output:
        print_sequence_summary(results)
        print_short_time_memory(runtime.agent.short_time_memory)
    print_short_time_memory_prompt(runtime.agent.short_time_memory)
    print_cumulative_reward(cumulative_reward)
    if args.visualize:
        output = write_intent_run_html(
            runtime.home,
            initial_position=initial_position,
            results=results,
            output_path=args.visualization_out,
            initial_desire_state=initial_desire_state,
            desire_state=runtime.agent.desire_state.to_dict(),
            desire_updates=desire_updates,
            cumulative_reward=cumulative_reward.to_dict(),
            world_failure_report=world_failure_report,
        )
        if args.verbose_output:
            print(f"Intent replay visualization: {output}")


def print_cumulative_reward(result: CumulativeRewardResult) -> None:
    print("=" * 72)
    print("Cumulative Reward")
    if result.error:
        print(result.error)
        return
    print(f"Intent Satisfaction: {result.intent_satisfaction:.2f}/10")
    print(f"Execution Steps: {result.execution_steps}")
    print(f"Cumulative Reward: {result.cumulative_reward:.4f}")
    if result.reason:
        print(f"Reason: {result.reason}")


def print_demonstrated_habit_learning(result, *, output_path: Path | None) -> None:
    print("=" * 72)
    print("Demonstrated Habit Learning")
    if result.error:
        print(result.error)
        return
    print(f"Accepted: {len(result.records)}")
    print(f"Created: {result.created_count}")
    print(f"Reinforced: {result.reinforced_count}")
    print(f"Rejected: {len(result.rejections)}")
    for record in result.records:
        print(
            f"- h={record.habit_strength:.2f} "
            f"[{', '.join(record.required_cue_ids)}] -> {record.response_text}"
        )
    if output_path is not None:
        print(f"Demonstrated Habit Pool: {output_path}")


def demonstrated_habit_path(*, configured_path: str, avatar_key: str, agent_name: str) -> Path:
    if str(configured_path or "").strip():
        return Path(configured_path).expanduser()
    identity = str(avatar_key or agent_name or "agent").strip().lower()
    filename = re.sub(r"[^a-z0-9_-]+", "_", identity).strip("_") or "agent"
    return DEFAULT_DEMONSTRATED_HABIT_DIR / f"{filename}.json"


def main() -> None:
    args = build_parser().parse_args()
    configure_llm_routing(args.llm_profile)
    set_debug_action_proposal_prompt(args.debug_action_prompt)
    set_debug_without_world_model_prompt(args.debug_action_prompt)
    set_debug_intuition_prompt(args.debug_intuition_prompt)
    set_debug_think_prompt(args.debug_think_prompt)
    set_debug_target_resolver(args.verbose_output)
    home = build_scene(args.scene)
    agent_start = home.default_agent_start or DEFAULT_AGENT_START
    agent = build_agent_from_args(args, agent_start=agent_start)
    intent_text = args.intent.strip() or DEFAULT_INTENT_TEMPLATE.format(agent_name=agent.name)
    demonstrated_habit_file = demonstrated_habit_path(
        configured_path=args.demonstrated_habit_file,
        avatar_key=args.agent_avatar,
        agent_name=agent.name,
    )
    runtime = EnvironmentInteractionEngine(
        home=home,
        scene_name=args.scene,
        agent=agent,
        provider_name=args.provider,
        model=args.model,
        intent_reflection_interval=args.max_turns,
        intuition_mode=args.intuition_mode,
        habitual_retrieval_probability=args.habitual_retrieval_probability,
        habitual_activation_threshold=args.habitual_activation_threshold,
        habitual_awareness_mode=args.habitual_awareness_mode,
        unconscious_habit_memory_probability=args.unconscious_habit_memory_probability,
        computation_architecture=args.computation_architecture,
        world_model_mode=args.world_model_mode,
        action_sample_window=args.action_sample_window,
        iteration_number=args.iteration_number,
        planning_strategy=args.planning_strategy,
        beam_width=args.beam_width,
        planning_branching_factor=args.planning_branching_factor,
        mcts_simulations=args.mcts_simulations,
        mcts_exploration_constant=args.mcts_exploration_constant,
        recent_experience_count=args.recent_experience_count,
        selection_method=args.selection_method,
        selection_temperature=args.selection_temperature,
        selection_epsilon=args.selection_epsilon,
        arbiter_mode=args.arbiter_mode,
        arbiter_random_seed=args.arbiter_random_seed,
        world_failure_output_path=args.world_failure_export_out,
        interaction_reach=args.interaction_reach,
        disable_spatial_gating=args.disable_spatial_gating,
    )
    from agent.emprical_gate.cli import configure_runtime
    configure_runtime(runtime, args)
    demonstrated_memory = load_demonstrated_habit_memory(
        demonstrated_habit_file,
        agent_name=runtime.agent.name,
        habit_strength_threshold=(
            runtime.agent.long_term_memory.cue_memory.habit_strength_threshold
        ),
    ) if args.demonstrated_habit_mode != "off" else None
    if args.demonstrated_habit_mode in {"use", "learn_and_use"}:
        runtime.agent.long_term_memory.demonstrated_habit_memory = demonstrated_memory
    runtime.agent.active_intent = IntentState(
        intent_text=intent_text,
        decided_at=runtime.agent.time_belief.current_datetime,
    )
    runtime.agent.short_time_memory = ShortTermMemory(intent_text=intent_text)
    initial_position = runtime.agent.center
    initial_desire_state = runtime.agent.desire_state.to_dict()
    desire_updates = []
    demonstrated_observations = []
    previous_successful_action = ""

    print(f"Computation architecture: {runtime.loop.computation_architecture}")
    print(f"LLM routing profile: {args.llm_profile}")
    print(
        "Degree of model-based control: "
        f"{runtime.agent.degree_of_model_based_control:.2f}"
    )
    print(f"Habitual awareness mode: {runtime.loop.habitual_awareness_mode}")
    print(f"Interaction reach: {runtime.loop.action_executor.interaction_reach:g}")
    spatial_mode = "easy" if runtime.loop.action_executor.disable_spatial_gating else "strict"
    print(f"Spatial mode: {spatial_mode}")
    if args.demonstrated_habit_mode != "off":
        loaded_count = len(demonstrated_memory.records) if demonstrated_memory is not None else 0
        print(
            f"Demonstrated habits: mode={args.demonstrated_habit_mode}, "
            f"loaded={loaded_count}, strength={args.demonstrated_habit_strength:.2f}"
        )
    if runtime.loop.computation_architecture == "world_model":
        print(f"World Model mode: {runtime.loop.world_model_mode}")
        print(
            "Planning: "
            f"strategy={runtime.loop.planning_strategy}, "
            f"depth={runtime.loop.iteration_number}, "
            f"beam_width={runtime.loop.beam_width}, "
            f"branching_factor={runtime.loop.planning_branching_factor}"
        )
        if runtime.loop.planning_strategy == "mcts":
            print(
                "MCTS: "
                f"simulations={runtime.loop.mcts_simulations}, "
                f"exploration_constant={runtime.loop.mcts_exploration_constant:.4f}"
            )
        print(f"Arbiter mode: {runtime.loop.arbiter_mode}")

    if args.verbose_output:
        print("Single Intent Main Entry")
        print("DB -> Intent generation is bypassed.")
        print(f"Scene: {args.scene} ({runtime.home.name})")
        print(f"Agent: {runtime.agent.name}")
        print(f"Injected intent: {intent_text}")
        print(f"Reflection interval: every {args.max_turns} action turns")
        print(f"Intuition mode: {args.intuition_mode}")
        print(f"Computation architecture: {runtime.loop.computation_architecture}")
        if runtime.loop.computation_architecture == "world_model":
            print(
                "World Model: "
                f"mode={runtime.loop.world_model_mode}, "
                f"samples={runtime.loop.action_sample_window}, "
                f"iterations={runtime.loop.iteration_number}, "
                f"selection={runtime.loop.selection_method}"
            )
            print(f"Arbiter mode: {runtime.loop.arbiter_mode}")
        print(f"Habitual retrieval probability: {runtime.loop.habitual_retrieval_probability:.0%}")
        print(f"Habitual activation threshold: {runtime.loop.habitual_activation_threshold:.2f}")
        print("Habitual retrieval: Algorithm B (dimension embedding + S-R strength, Ollama bge-m3)")
        print(f"Unconscious habit memory probability: {runtime.loop.unconscious_habit_memory_probability:.0%}")
        print(f"Desire update mode: {args.desire_update_mode}")
        print("Cognitive route: Perceive -> Context Cue -> Habitual Gate -> Habitual/Goal-Directed -> Arbiter -> Target Resolver -> World")
        print(f"Safety cap: {args.safety_turns} action turns")
        print()

    results = []
    run_error = ""
    try:
        for _ in range(args.safety_turns):
            from_position = runtime.agent.center
            result = runtime.action_turn()
            results.append(result)
            if args.demonstrated_habit_mode in {"learn", "learn_and_use"}:
                area = runtime.physics.get_area_for_point(*result.final_position)
                observation = capture_demonstrated_action_observation(
                    result,
                    runtime.agent.latest_cue_extraction,
                    previous_successful_action=previous_successful_action,
                    area_name=area.name if area is not None else "",
                )
                if observation is not None:
                    demonstrated_observations.append(observation)
            successful_action = successful_semantic_action_text(result)
            if successful_action:
                previous_successful_action = successful_action
            remember_turn(runtime, result)
            update_self_state(runtime, result, from_position=from_position)
            review_intent_after_turn(runtime, result)
            desire_update = None
            if (
                args.desire_update_mode == DESIRE_UPDATE_EVERY_STEP
                and result.counts_as_intent_action
            ):
                desire_update = update_desire_after_step(runtime, result)
                desire_updates.append(desire_update)
            runtime.loop._apply_emprical_gate_fixed_state(runtime.agent)
            result.desire_update = desire_update or {}
            result.desire_state = runtime.agent.desire_state.to_dict()
            if args.verbose_output:
                print_step(result, desire_update=desire_update)
                print_self_state(runtime.agent.self_state)
            else:
                print_concise_step(result, desire_update=desire_update)
            if runtime.agent.active_intent is None or result.intent_status != "active":
                break
    except Exception:
        run_error = traceback.format_exc()
        print("=" * 72)
        print("Checkpoint: run interrupted by error")
        print(run_error, flush=True)
    finally:
        if args.desire_update_mode == DESIRE_UPDATE_ON_INTENT_END:
            desire_update = update_desire_after_intent(runtime, results)
            if desire_update is not None and results:
                desire_updates.append(desire_update)
                results[-1].desire_update = desire_update
                results[-1].desire_state = runtime.agent.desire_state.to_dict()
                print_desire_update(desire_update)
        if (
            not run_error
            and args.demonstrated_habit_mode in {"learn", "learn_and_use"}
            and demonstrated_memory is not None
        ):
            learning_result = learn_demonstrated_habits(
                agent_name=runtime.agent.name,
                intent_text=intent_text,
                observations=demonstrated_observations,
                memory=demonstrated_memory,
                persona_memory=runtime.agent.long_term_memory.cue_memory,
                initial_habit_strength=args.demonstrated_habit_strength,
                provider_name=args.provider,
                model=args.model,
            )
            if args.debug_demonstrated_habit_prompt and learning_result.prompt:
                print("=" * 72)
                print("Demonstrated Habit Learning Prompt")
                print(learning_result.prompt)
            demonstrated_output = None
            if not learning_result.error:
                runtime.agent.long_term_memory.demonstrated_habit_memory = demonstrated_memory
                demonstrated_output = save_demonstrated_habit_memory(
                    demonstrated_habit_file,
                    agent_name=runtime.agent.name,
                    intent_text=intent_text,
                    memory=demonstrated_memory,
                )
            print_demonstrated_habit_learning(
                learning_result,
                output_path=demonstrated_output,
            )
        cumulative_reward = evaluate_cumulative_reward(
            agent_name=runtime.agent.name,
            intent_text=intent_text,
            results=results,
            home=runtime.home,
            final_desire_state=runtime.agent.desire_state.to_dict(),
            provider_name=args.provider,
            model=args.model,
        )
        world_failure_report = collect_world_failure_report(runtime)
        behavior_log_output = write_behavior_log(
            results,
            output_path=args.behavior_log_out,
        )
        print(f"Behavior evaluation log: {behavior_log_output} ({len(results)} records)")
        finish_run(
            runtime,
            args=args,
            initial_position=initial_position,
            results=results,
            initial_desire_state=initial_desire_state,
            desire_updates=desire_updates,
            cumulative_reward=cumulative_reward,
            world_failure_report=world_failure_report,
        )
        if run_error:
            print("Checkpoint rendered up to the last completed action turn.", flush=True)


if __name__ == "__main__":
    main()

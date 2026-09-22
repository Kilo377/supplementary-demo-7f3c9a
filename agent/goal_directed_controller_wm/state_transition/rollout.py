from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agent.goal_directed_controller_wm.action_generation import (
    ActionCandidate,
)
from agent.intent.state import IntentState
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from contextual_world.world_old.action_execution import WorldActionExecutor

from .backend import DEFAULT_WORLD_MODEL_MODE, normalize_world_model_mode
from .complex_transition import (
    ComplexWorldSnapshot,
    fork_complex_world,
    run_complex_state_transition,
)
from .context import build_state_transition_context
from .iteration_action import generate_transition_iteration_action
from .transition import run_state_transition
from .types import (
    StateTransition,
    StateTransitionPromptContext,
    StateTransitionResult,
)


@dataclass(frozen=True)
class StateTransitionRolloutState:
    agent_name: str
    intent_text: str
    spatial_belief_text: str
    physical_state_text: str
    physiological_state: dict[str, int]
    internal_state: dict[str, int]
    mental_text: str = ""
    motivation_texts: tuple[str, ...] = ()
    current_datetime: datetime | None = None
    fallback_time_text: str = ""
    actual_recent_experience_text: str = ""
    simulated_experiences: tuple[str, ...] = ()
    simulated_spatial_updates: tuple[str, ...] = ()

    @classmethod
    def from_agent(
        cls,
        agent,
        *,
        intent_text: str,
        recent_experience_count: int,
    ) -> "StateTransitionRolloutState":
        context = build_state_transition_context(
            agent,
            action_text="",
            intent_text=intent_text,
            recent_experience_count=recent_experience_count,
        )
        desire = getattr(agent, "desire_state", None)
        physiological = (
            desire.physiological_state.to_dict()
            if desire is not None
            else {"hunger": 0, "thirst": 0, "hygiene": 0}
        )
        internal = (
            desire.internal_state.to_dict()
            if desire is not None
            else {"stress": 0, "tension": 0, "fatigue": 0}
        )
        motivation_texts = tuple(
            str(goal.text).strip()
            for goal in list(getattr(desire, "work_goal", []) or [])
            if not bool(getattr(goal, "completed", False))
            and str(getattr(goal, "text", "") or "").strip()
        )
        time_belief = getattr(agent, "time_belief", None)
        current_datetime = getattr(time_belief, "current_datetime", None)
        return cls(
            agent_name=context.agent_name,
            intent_text=context.intent_text,
            spatial_belief_text=context.spatial_belief_text,
            physical_state_text=context.physical_state_text,
            physiological_state=dict(physiological),
            internal_state=dict(internal),
            mental_text=str(getattr(desire, "mental", "") or "").strip(),
            motivation_texts=motivation_texts,
            current_datetime=(
                current_datetime if isinstance(current_datetime, datetime) else None
            ),
            fallback_time_text=context.time_text,
            actual_recent_experience_text=context.recent_experience_text,
        )

    def to_prompt_context(self, *, action_text: str) -> StateTransitionPromptContext:
        return StateTransitionPromptContext(
            agent_name=self.agent_name,
            action_text=action_text,
            intent_text=self.intent_text,
            spatial_belief_text=self._current_spatial_belief_text(),
            time_text=self._current_time_text(),
            physical_state_text=self.physical_state_text,
            internal_state_text=self._current_internal_state_text(),
            recent_experience_text=self._current_experience_text(),
        )

    def format_for_action_prompt(self) -> str:
        sections = [
            (
                "这是 State Transition 推测出的下一刻状态，尚未在真实 World 中发生。"
                "在当前推演分支中，把它当作现在相信的状态。"
            )
        ]
        _append_section(sections, "当前时间：", self._current_time_text())
        _append_section(sections, "当前身体状态：", self.physical_state_text)
        _append_section(sections, "当前内部状态：", self._current_internal_state_text())
        _append_section(sections, "当前空间记忆：", self._current_spatial_belief_text())
        _append_section(sections, "已经经历的事情：", self._current_experience_text())
        return "\n\n".join(sections)

    def after_transition(
        self,
        *,
        action_text: str,
        transition: StateTransition,
    ) -> "StateTransitionRolloutState":
        deltas = transition.internal_state_changes
        physiological = {
            "hunger": _apply_delta(self.physiological_state.get("hunger", 0), deltas.hunger),
            "thirst": _apply_delta(self.physiological_state.get("thirst", 0), deltas.thirst),
            "hygiene": _apply_delta(self.physiological_state.get("hygiene", 0), deltas.hygiene),
        }
        internal = {
            "stress": _apply_delta(self.internal_state.get("stress", 0), deltas.stress),
            "tension": _apply_delta(self.internal_state.get("tension", 0), deltas.tension),
            "fatigue": _apply_delta(self.internal_state.get("fatigue", 0), deltas.fatigue),
        }
        elapsed_seconds = max(0, transition.elapsed_seconds)
        current_datetime = (
            self.current_datetime + timedelta(seconds=elapsed_seconds)
            if self.current_datetime is not None
            else None
        )
        experience = _simulated_experience_text(action_text, transition)
        updates = tuple(
            f"{update.element}：{update.state_change}"
            for update in transition.spatial_belief_updates
            if update.element or update.state_change
        )
        return StateTransitionRolloutState(
            agent_name=self.agent_name,
            intent_text=self.intent_text,
            spatial_belief_text=self.spatial_belief_text,
            physical_state_text=_transition_physical_state_text(
                self.agent_name,
                transition,
            ),
            physiological_state=physiological,
            internal_state=internal,
            mental_text=(deltas.mental_change or self.mental_text),
            motivation_texts=self.motivation_texts,
            current_datetime=current_datetime,
            fallback_time_text=self.fallback_time_text,
            actual_recent_experience_text=self.actual_recent_experience_text,
            simulated_experiences=(*self.simulated_experiences, experience),
            simulated_spatial_updates=(*self.simulated_spatial_updates, *updates),
        )

    def _current_spatial_belief_text(self) -> str:
        sections = [self.spatial_belief_text.strip()]
        if self.simulated_spatial_updates:
            updates = "\n".join(
                f"- {text}" for text in self.simulated_spatial_updates if text
            )
            if updates:
                sections.append(f"在当前推演分支中，你进一步认为这些变化已经发生：\n{updates}")
        return "\n\n".join(section for section in sections if section)

    def _current_time_text(self) -> str:
        if self.current_datetime is None:
            return self.fallback_time_text
        return _format_datetime_label(self.current_datetime)

    def _current_internal_state_text(self) -> str:
        lines = [
            "生理需求采用0到10的程度："
            f"饥饿{self.physiological_state.get('hunger', 0)}，"
            f"口渴{self.physiological_state.get('thirst', 0)}，"
            f"清洁需求{self.physiological_state.get('hygiene', 0)}。",
            "内部状态采用0到10的程度："
            f"压力{self.internal_state.get('stress', 0)}，"
            f"紧张{self.internal_state.get('tension', 0)}，"
            f"疲劳{self.internal_state.get('fatigue', 0)}。",
        ]
        if self.mental_text:
            lines.append(self.mental_text)
        if self.motivation_texts:
            lines.append(f"{self.agent_name}仍然在意：{'；'.join(self.motivation_texts)}")
        return "\n".join(lines)

    def _current_experience_text(self) -> str:
        sections = []
        if self.actual_recent_experience_text.strip():
            sections.append(self.actual_recent_experience_text.strip())
        if self.simulated_experiences:
            lines = "\n".join(
                f"- {text}" for text in self.simulated_experiences if text
            )
            if lines:
                sections.append(f"在当前推演分支中，假设已经发生：\n{lines}")
        return "\n\n".join(sections)


@dataclass
class StateTransitionRolloutNode:
    path: tuple[int, ...]
    depth: int
    candidate: ActionCandidate
    transition_result: StateTransitionResult
    stop_reason: str = ""
    expansion_error: str = ""
    iteration_intuition_prompt: str = ""
    children: list["StateTransitionRolloutNode"] = field(default_factory=list)

    @property
    def transition(self) -> StateTransition | None:
        return self.transition_result.transition

    @property
    def path_text(self) -> str:
        return ".".join(str(index) for index in self.path)


@dataclass
class StateTransitionRolloutResult:
    roots: list[StateTransitionRolloutNode]
    iteration_number: int
    world_model_mode: str = DEFAULT_WORLD_MODEL_MODE

    def iter_nodes(self):
        for root in self.roots:
            yield from _iter_node(root)

    @property
    def total_transitions(self) -> int:
        return sum(1 for _ in self.iter_nodes())


def rollout_state_transitions(
    agent,
    *,
    intent: IntentState,
    engine: PhysicsEngine,
    initial_action_space: list[ActionCandidate],
    iteration_number: int = 3,
    recent_experience_count: int = 8,
    world_model_mode: str = DEFAULT_WORLD_MODEL_MODE,
    action_executor: WorldActionExecutor | None = None,
    previous_execution_result: str = "",
    provider_name: str = "ollama",
    model: str | None = None,
) -> StateTransitionRolloutResult:
    mode = normalize_world_model_mode(world_model_mode)
    depth_limit = max(1, int(iteration_number))
    initial_state = StateTransitionRolloutState.from_agent(
        agent,
        intent_text=intent.intent_text,
        recent_experience_count=recent_experience_count,
    )
    roots = []
    for index, candidate in enumerate(initial_action_space, start=1):
        complex_snapshot = (
            fork_complex_world(
                agent,
                engine,
                action_executor,
                provider_name=provider_name,
                model=model,
                previous_execution_result=previous_execution_result,
            )
            if mode == "complex"
            else None
        )
        roots.append(_rollout_candidate(
            state=initial_state,
            candidate=candidate,
            path=(index,),
            depth=1,
            depth_limit=depth_limit,
            intent=intent,
            engine=engine,
            world_model_mode=mode,
            complex_snapshot=complex_snapshot,
            provider_name=provider_name,
            model=model,
        ))
    return StateTransitionRolloutResult(
        roots=roots,
        iteration_number=depth_limit,
        world_model_mode=mode,
    )


def _rollout_candidate(
    *,
    state: StateTransitionRolloutState,
    candidate: ActionCandidate,
    path: tuple[int, ...],
    depth: int,
    depth_limit: int,
    intent: IntentState,
    engine: PhysicsEngine,
    world_model_mode: str,
    complex_snapshot: ComplexWorldSnapshot | None,
    provider_name: str,
    model: str | None,
) -> StateTransitionRolloutNode:
    if world_model_mode == "complex":
        if complex_snapshot is None:
            raise ValueError("Complex World Model requires a forked World snapshot.")
        transition_result = run_complex_state_transition(
            complex_snapshot,
            candidate,
            intent_text=intent.intent_text,
            provider_name=provider_name,
            model=model,
        )
    else:
        transition_result = run_state_transition(
            state.to_prompt_context(action_text=candidate.action_text),
            provider_name=provider_name,
            model=model,
        )
    node = StateTransitionRolloutNode(
        path=path,
        depth=depth,
        candidate=candidate,
        transition_result=transition_result,
    )
    transition = transition_result.transition
    if transition is None:
        node.stop_reason = "transition_error"
        return node
    if transition.intent_satisfaction.is_satisfied:
        node.stop_reason = "intent_satisfied"
        return node
    if depth >= depth_limit:
        node.stop_reason = "iteration_limit"
        return node

    next_state = state.after_transition(
        action_text=candidate.action_text,
        transition=transition,
    )
    generation = generate_transition_iteration_action(
        agent_name=state.agent_name,
        intent_text=intent.intent_text,
        transition_state_text=next_state.format_for_action_prompt(),
        failure_context_text=(
            transition_result.failure_context.format_for_recovery_prompt(
                state.agent_name
            )
            if transition_result.failure_context is not None
            else ""
        ),
        engine=(complex_snapshot.engine if complex_snapshot is not None else engine),
        provider_name=provider_name,
        model=model,
    )
    node.iteration_intuition_prompt = generation.prompt
    if generation.error:
        node.stop_reason = "action_generation_error"
        node.expansion_error = generation.error
        return node
    if not generation.action_space:
        node.stop_reason = "no_next_action"
        return node

    node.children = [
        _rollout_candidate(
            state=next_state,
            candidate=generation.action_space[0],
            path=(*path, 1),
            depth=depth + 1,
            depth_limit=depth_limit,
            intent=intent,
            engine=engine,
            world_model_mode=world_model_mode,
            complex_snapshot=complex_snapshot,
            provider_name=provider_name,
            model=model,
        )
    ]
    return node


def _transition_physical_state_text(
    agent_name: str,
    transition: StateTransition,
) -> str:
    state = transition.next_self_state
    lines = []
    if state.area:
        line = f"{agent_name}现在在{state.area}"
        if state.near_element:
            line += f"，靠近{state.near_element}"
        lines.append(f"{line}。")
    if state.posture:
        lines.append(f"{agent_name}当前姿态是{state.posture}。")
    if state.facing_or_gaze:
        lines.append(f"{agent_name}当前朝向或注视{state.facing_or_gaze}。")
    if state.holding:
        lines.append(f"{agent_name}手里拿着{'、'.join(state.holding)}。")
    if state.interacting_with:
        lines.append(f"{agent_name}正在与{'、'.join(state.interacting_with)}交互。")
    if state.worn_items_change:
        lines.append(f"穿戴变化：{state.worn_items_change}")
    if state.body_surface_change:
        lines.append(f"身体表面变化：{state.body_surface_change}")
    return "\n".join(lines)


def _simulated_experience_text(
    action_text: str,
    transition: StateTransition,
) -> str:
    result = transition.outcome.description
    feedback = transition.expected_feedback
    text = f"做了“{action_text}”。"
    if result:
        text += f"结果可能是：{result}"
    if feedback and feedback != result:
        text += f" 预计会感受到：{feedback}"
    return text


def _apply_delta(value: int, delta: int) -> int:
    return max(0, min(10, int(value) + int(delta)))


def _format_datetime_label(value: datetime) -> str:
    if 5 <= value.hour < 12:
        period = "上午"
    elif 12 <= value.hour < 14:
        period = "中午"
    elif 14 <= value.hour < 18:
        period = "下午"
    elif 18 <= value.hour < 24:
        period = "晚上"
    else:
        period = "凌晨"
    return f"{period} {value.hour}:{value.minute:02d}"


def _append_section(sections: list[str], heading: str, content: str) -> None:
    cleaned = str(content or "").strip()
    if cleaned:
        sections.append(f"{heading}\n{cleaned}")


def _iter_node(node: StateTransitionRolloutNode):
    yield node
    for child in node.children:
        yield from _iter_node(child)

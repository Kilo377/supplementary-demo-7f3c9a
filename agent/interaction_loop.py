from __future__ import annotations

from dataclasses import dataclass, field
import random
import traceback

from agent.agent import Agent
from agent.action import ActionProposalResult, propose_next_action
from agent.intent.lifecycle import IntentLifecycleResult, evaluate_intent_lifecycle
from agent.intent.progress import append_progress, fallback_progress_item, summarize_intent_progress
from agent.intent.proposal import propose_new_intent
from agent.intent.state import IntentState
from agent.perceive import PerceiveResult
from agent.habitual_controller.cue_extraction import extract_cues, previous_execution_from_memory_episode
from agent.habitual_controller import (
    AwarenessResult,
    PreparedHabitualResponse,
    activate_habitual,
    decide_habitual_retrieval,
    judge_goal_conflict,
    judge_habitual_awareness,
    route_habitual_by_awareness,
)
from agent.goal_directed_controller_wm import run_world_model_controller
from agent.goal_directed_controller_wm.arbiter import (
    DEFAULT_ARBITER_MODE,
    normalize_arbiter_mode,
)
from agent.goal_directed_controller_wm.state_transition import (
    DEFAULT_WORLD_MODEL_MODE,
    normalize_world_model_mode,
)
from agent.goal_directed_controller_wm.planning import normalize_planning_strategy
from agent.goal_directed_controller_wowm import run_without_world_model_controller
from agent.habitual_controller.prompt_inputs import habitual_current_state_text, habitual_trigger_text
from agent.arbiter import run_arbiter
from agent.working_memory import build_working_memory
from agent.intuition import IntuitionResult, generate_intuition
from agent.think import ThinkResult, generate_think, route_after_thinking
from agent.target_resolver import (
    TargetResolution,
    decision_from_target_resolution,
    resolve_target_fallback,
    resolve_target_with_llm,
    target_resolution_trace,
)
from core.action_types import ActionResult, AgentDecision
from agent.emprical_gate import evaluate_gate
from contextual_world.world_old.action_execution import WorldActionExecutor
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine


DEBUG_TARGET_RESOLVER = True
COMPUTATION_ARCHITECTURES = (
    "intuition",
    "without_world_model",
    "direct_action",
    "world_model",
)
HABITUAL_AWARENESS_MODES = ("llm", "memory")


def set_debug_target_resolver(enabled: bool) -> None:
    global DEBUG_TARGET_RESOLVER
    DEBUG_TARGET_RESOLVER = enabled


def _normalize_intuition_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"active", "shadow", "off"}:
        return mode
    return "active"


def _normalize_computation_architecture(value: str) -> str:
    architecture = str(value or "").strip().lower()
    if architecture == "direct_action":
        return "without_world_model"
    if architecture in COMPUTATION_ARCHITECTURES:
        return architecture
    return "intuition"


def _scene_name(engine: PhysicsEngine) -> str:
    return str(getattr(engine.home, "name", "") or getattr(engine.home, "node_id", "") or "Current scene")


@dataclass
class AgentStepRecord:
    step_id: int
    area_id: str
    area_name: str
    narration_text: str
    notice_text: str
    intent_text: str
    intent_status: str
    intent_action_index: int
    intent_progress: list[str]
    decision: AgentDecision
    action_summary: str
    execution_kind: str
    execution_narration: str
    estimated_duration: str
    temporary_element_changes: list[dict]
    actor_state_update: dict
    environment_feedback: dict
    execution_debug: dict
    final_position: tuple[float, float]
    counts_as_intent_action: bool = True


@dataclass
class AgentInteractionLoop:
    provider_name: str = "ollama"
    model: str | None = None
    intent_reflection_interval: int = 10
    intuition_mode: str = "active"
    habitual_retrieval_probability: float = 0.9
    habitual_activation_threshold: float = 0.35
    habitual_awareness_mode: str = "llm"
    emprical_gate_enabled: bool = True
    emprical_gate_overrides: dict = field(default_factory=dict)
    emprical_gate_freeze_state: bool = False
    emprical_gate_rng: random.Random = field(default_factory=random.Random)
    emprical_gate_history: list[dict] = field(default_factory=list)
    emprical_gate_action_history: list[dict] = field(default_factory=list)
    unconscious_habit_memory_probability: float = 0.2
    computation_architecture: str = "intuition"
    world_model_mode: str = DEFAULT_WORLD_MODEL_MODE
    action_sample_window: int | None = None
    iteration_number: int = 3
    recent_experience_count: int = 8
    selection_method: str = "greedy"
    selection_temperature: float = 0.2
    selection_epsilon: float = 0.1
    planning_strategy: str = "greedy"
    beam_width: int = 3
    planning_branching_factor: int = 3
    mcts_simulations: int = 12
    mcts_exploration_constant: float = 2 ** 0.5
    arbiter_mode: str = DEFAULT_ARBITER_MODE
    arbiter_random_seed: int | None = None
    world_failure_output_path: str | None = None
    interaction_reach: float = 0.0
    disable_spatial_gating: bool = True
    previous_decision_text: str = ""
    previous_execution_result: str = ""
    last_target_resolver_error: str = ""
    last_think_error: str = ""
    last_lifecycle_reviewed_step_count: int = 0
    action_executor: WorldActionExecutor = field(init=False)
    arbiter_rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.computation_architecture = _normalize_computation_architecture(
            self.computation_architecture
        )
        self.world_model_mode = normalize_world_model_mode(self.world_model_mode)
        self.planning_strategy = normalize_planning_strategy(self.planning_strategy)
        self.beam_width = max(1, int(self.beam_width))
        self.planning_branching_factor = max(
            1, min(12, int(self.planning_branching_factor))
        )
        self.mcts_simulations = max(1, int(self.mcts_simulations))
        self.mcts_exploration_constant = max(
            0.0, float(self.mcts_exploration_constant)
        )
        self.arbiter_mode = normalize_arbiter_mode(self.arbiter_mode)
        self.arbiter_rng = random.Random(self.arbiter_random_seed)
        default_sample_window = 3 if self.computation_architecture == "world_model" else 1
        self.action_sample_window = max(
            1,
            min(
                12,
                int(
                    self.action_sample_window
                    if self.action_sample_window is not None
                    else default_sample_window
                ),
            ),
        )
        self.iteration_number = max(1, int(self.iteration_number))
        self.recent_experience_count = max(0, int(self.recent_experience_count))
        self.habitual_retrieval_probability = min(
            1.0,
            max(0.0, float(self.habitual_retrieval_probability)),
        )
        self.habitual_activation_threshold = min(
            1.0,
            max(0.0, float(self.habitual_activation_threshold)),
        )
        self.habitual_awareness_mode = str(self.habitual_awareness_mode or "llm").strip().lower()
        if self.habitual_awareness_mode not in HABITUAL_AWARENESS_MODES:
            raise ValueError(
                "Unknown habitual awareness mode: "
                f"{self.habitual_awareness_mode}. Expected one of "
                f"{', '.join(HABITUAL_AWARENESS_MODES)}."
            )
        self.unconscious_habit_memory_probability = min(
            1.0,
            max(0.0, float(self.unconscious_habit_memory_probability)),
        )
        self.action_executor = WorldActionExecutor(
            provider_name=self.provider_name,
            model=self.model,
            failure_output_path=self.world_failure_output_path,
            interaction_reach=max(0.0, float(self.interaction_reach)),
            disable_spatial_gating=bool(self.disable_spatial_gating),
        )

    def run_step(self, agent: Agent, engine: PhysicsEngine) -> AgentStepRecord:
        return self.run_action_turn(agent, engine)

    def run_action_turn(self, agent: Agent, engine: PhysicsEngine) -> AgentStepRecord:
        self._apply_emprical_gate_fixed_state(agent)
        perception = agent.perceive(engine)
        previous_episode = agent.short_time_memory.episodes[-1] if agent.short_time_memory.episodes else None
        agent.latest_cue_extraction = extract_cues(
            agent,
            perception,
            previous_execution=previous_execution_from_memory_episode(previous_episode),
            cue_buffer=agent.cue_extraction_buffer,
        )
        self._ensure_active_intent(agent, perception)
        current_intent_text = agent.active_intent.intent_text if agent.active_intent is not None else ""
        intent_action_index = (agent.active_intent.step_count + 1) if agent.active_intent is not None else 0
        decision = self.decide(agent, engine, perception)
        decision.computation_architecture = self.computation_architecture
        decision.cognitive_step_id = perception.step_id
        decision.intent_action_index = intent_action_index
        decision.previous_execution_result = self.previous_execution_result
        action_result = self.action_executor.execute(
            agent,
            engine,
            decision,
            previous_execution_result=self.previous_execution_result,
        )
        self._record_habitual_execution(agent, decision, action_result)
        self._record_emprical_gate_action(decision, action_result)
        self._advance_time_belief(agent, action_result)
        self._record_intent_action(agent, decision, action_result)
        current_intent_status = agent.active_intent.status if agent.active_intent is not None else ""
        self._store_world_feedback_summary(agent, action_result)
        self.previous_decision_text = decision.action_proposal_text or decision.reason
        self.previous_execution_result = action_result.feedback_text()
        return AgentStepRecord(
            step_id=perception.step_id,
            area_id=perception.area_id,
            area_name=perception.area_name,
            narration_text=perception.narration_text,
            notice_text=perception.notice_text,
            intent_text=current_intent_text,
            intent_status=current_intent_status,
            intent_action_index=intent_action_index,
            intent_progress=list(agent.active_intent.progress) if agent.active_intent is not None else [],
            decision=decision,
            action_summary=action_result.summary,
            execution_kind=action_result.execution_kind,
            execution_narration=action_result.feedback_text(),
            estimated_duration=action_result.estimated_duration,
            temporary_element_changes=action_result.temporary_element_changes,
            actor_state_update=action_result.actor_state_update,
            environment_feedback=(
                action_result.environment_feedback.to_dict()
                if action_result.environment_feedback is not None
                else {}
            ),
            execution_debug=dict(action_result.execution_debug or {}),
            final_position=agent.center,
            counts_as_intent_action=action_result.counts_as_intent_action,
        )

    def run_intent_cycle(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        *,
        max_action_turns: int = 10,
    ) -> list[AgentStepRecord]:
        records: list[AgentStepRecord] = []
        for _ in range(max_action_turns):
            record = self.run_action_turn(agent, engine)
            records.append(record)
            if agent.active_intent is None or record.intent_status != "active":
                break
        return records

    def decide(self, agent: Agent, engine: PhysicsEngine, perception: PerceiveResult) -> AgentDecision:
        if self.action_executor.has_pending_graph_action():
            return self._decide_configured_goal_directed(agent, engine, perception)
        if self.computation_architecture == "world_model":
            return self._decide_world_model(agent, engine, perception)
        context = getattr(agent, "latest_cue_extraction", None)
        if context is None:
            return self._decide_configured_goal_directed(agent, engine, perception)
        gate = decide_habitual_retrieval(self.habitual_retrieval_probability)
        if not gate.should_retrieve:
            decision = self._decide_configured_goal_directed(agent, engine, perception)
            self._attach_habitual_gate(decision, gate, retrieval_status="skipped")
            return decision
        habitual = activate_habitual(
            agent,
            context,
            activation_threshold=self.habitual_activation_threshold,
        )
        response = habitual.strongest
        if response is None:
            decision = self._decide_configured_goal_directed(agent, engine, perception)
            self._attach_habitual_gate(decision, gate, retrieval_status=habitual.retrieval_status)
            return decision

        intent_text = agent.active_intent.intent_text if agent.active_intent is not None else ""
        conflict = judge_goal_conflict(
            intent_text=intent_text,
            habitual_response=response,
            provider_name=self.provider_name,
            model=self.model,
        )
        awareness = (
            AwarenessResult(True, "The habitual reaction conflicted with the current goal, so it entered consciousness.")
            if conflict.conflict
            else judge_habitual_awareness(
                response,
                perception,
                context,
                provider_name=self.provider_name,
                model=self.model,
                use_llm=self.habitual_awareness_mode == "llm",
            )
        )
        empirical = self._evaluate_emprical_gate(agent, conscious=awareness.conscious)
        if empirical.route == "habitual_direct":
            decision = self._decision_from_habitual_response(agent, engine, perception, response)
            self._attach_habitual_trace(
                decision,
                response,
                awareness="conscious" if awareness.conscious else "unconscious",
                conflict=conflict.conflict,
                arbiter_mode="habitual" if awareness.conscious else "fast_path",
                arbiter_reason=awareness.reason,
                conflict_reason=conflict.reason,
                awareness_reason=awareness.reason,
            )
            self._attach_habitual_gate(decision, gate, retrieval_status=habitual.retrieval_status)
            self._attach_emprical_gate(decision, empirical)
            return decision

        goal_decision = self._decide_configured_goal_directed(agent, engine, perception)
        goal_action = goal_decision.action_proposal_text or goal_decision.reason
        arbiter = run_arbiter(
            agent_name=agent.name,
            personality=agent.personality,
            intent_text=intent_text,
            trigger_context_text=habitual_trigger_text(response, perception, context),
            current_state_text=habitual_current_state_text(perception, context),
            intuition_text=goal_decision.intuition_thought,
            think_text=_decision_think_text(goal_decision),
            goal_directed_action=goal_action,
            habitual_response=response,
            entry_reason="goal_conflict" if conflict.conflict else "conscious_habit",
            provider_name=self.provider_name,
            model=self.model,
        )
        current_time = getattr(getattr(agent, "time_belief", None), "current_datetime", None)
        if current_time is not None:
            agent.habitual_last_considered_at[response.association_id] = current_time
        decision = self._decision_from_arbiter_action(
            agent,
            engine,
            perception,
            arbiter.action_text,
        )
        self._copy_goal_directed_trace(goal_decision, decision)
        self._attach_emprical_gate(decision, empirical)
        self._attach_habitual_trace(
            decision,
            response,
            awareness="conscious",
            conflict=conflict.conflict,
            arbiter_mode=arbiter.decision_mode,
            arbiter_reason=arbiter.reason or conflict.reason,
            conflict_reason=conflict.reason,
            awareness_reason=awareness.reason,
        )
        self._attach_habitual_gate(decision, gate, retrieval_status=habitual.retrieval_status)
        return decision

    def _decide_world_model(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        perception: PerceiveResult,
    ) -> AgentDecision:
        context = getattr(agent, "latest_cue_extraction", None)
        gate = decide_habitual_retrieval(self.habitual_retrieval_probability)
        response = None
        awareness_gate = None
        empirical = None
        retrieval_status = "no_context"
        if context is not None and gate.should_retrieve:
            activation = activate_habitual(
                agent,
                context,
                activation_threshold=self.habitual_activation_threshold,
            )
            retrieval_status = activation.retrieval_status
            response = activation.strongest
            if response is not None:
                awareness_gate = route_habitual_by_awareness(
                    activation,
                    perception,
                    context,
                    provider_name=self.provider_name,
                    model=self.model,
                    use_llm=self.habitual_awareness_mode == "llm",
                    desire_state=agent.desire_state,
                )
                empirical = self._evaluate_emprical_gate(
                    agent, conscious=awareness_gate.awareness.conscious,
                )
                if empirical.route == "habitual_direct":
                    decision = self._decision_from_habitual_response(
                        agent,
                        engine,
                        perception,
                        response,
                    )
                    self._attach_habitual_trace(
                        decision,
                        response,
                        awareness="conscious" if awareness_gate.awareness.conscious else "unconscious",
                        conflict=False,
                        arbiter_mode="habitual" if awareness_gate.awareness.conscious else "fast_path",
                        arbiter_reason=(
                            awareness_gate.awareness.reason
                            if awareness_gate.awareness is not None
                            else "The habitual reaction did not enter consciousness."
                        ),
                        conflict_reason="",
                        awareness_reason=(
                            awareness_gate.awareness.reason
                            if awareness_gate.awareness is not None
                            else ""
                        ),
                    )
                    self._attach_habitual_gate(
                        decision,
                        gate,
                        retrieval_status=retrieval_status,
                    )
                    self._attach_emprical_gate(decision, empirical)
                    return decision
        elif not gate.should_retrieve:
            retrieval_status = "skipped"

        competing_habits = (
            [response]
            if response is not None
            and awareness_gate is not None
            and awareness_gate.route == "competition"
            else []
        )
        controller = run_world_model_controller(
            agent,
            engine,
            perception,
            habitual_responses=competing_habits,
            action_sample_window=self.action_sample_window,
            iteration_number=self.iteration_number,
            recent_experience_count=self.recent_experience_count,
            world_model_mode=self.world_model_mode,
            action_executor=self.action_executor,
            previous_execution_result=self.previous_execution_result,
            selection_method=self.selection_method,
            selection_temperature=self.selection_temperature,
            selection_epsilon=self.selection_epsilon,
            planning_strategy=self.planning_strategy,
            beam_width=self.beam_width,
            planning_branching_factor=self.planning_branching_factor,
            mcts_simulations=self.mcts_simulations,
            mcts_exploration_constant=self.mcts_exploration_constant,
            arbiter_mode=self.arbiter_mode,
            arbiter_rng=self.arbiter_rng,
            provider_name=self.provider_name,
            model=self.model,
        )
        if controller.error or controller.selected_candidate is None:
            decision = self._decide_goal_directed(agent, engine, perception)
            decision.world_model_trace = controller.to_trace_dict()
            self._attach_emprical_gate(decision, empirical)
            decision.arbiter_reason = (
                f"World Model failed and used Intuition fallback: {controller.error}"
            )
            self._attach_habitual_gate(decision, gate, retrieval_status=retrieval_status)
            return decision

        candidate = controller.selected_candidate
        decision = self._decision_from_intuition(agent, engine, candidate.intuition_reply)
        if decision is None:
            decision = self._decision_from_arbiter_action(
                agent,
                engine,
                perception,
                candidate.action_text,
            )
        decision.intuition_route = candidate.route
        decision.intuition_thought = candidate.action_text
        decision.world_model_trace = controller.to_trace_dict()
        self._attach_emprical_gate(decision, empirical)
        selected_entry = (
            controller.arbitration.entry_by_id(controller.arbitration.selected_action_id)
            if controller.arbitration is not None
            else None
        )
        if selected_entry is not None:
            if controller.arbitration.mode.startswith("random_controller_"):
                selected_controller = controller.arbitration.mode.removeprefix(
                    "random_controller_"
                )
                decision.arbiter_reason = (
                    "Random Controller selected "
                    f"{selected_controller} with p(habitual)=0.50."
                )
            else:
                decision.arbiter_reason = (
                    f"World Model selected D={selected_entry.decision_value:.3f}, "
                    f"Q={selected_entry.normalized_reward:.3f}, "
                    f"H={selected_entry.habit_strength:.3f}."
                )

        selected_habit_ids = {
            support.association_id for support in candidate.habitual_support
        }
        random_competition = bool(
            response is not None
            and controller.arbitration is not None
            and controller.arbitration.mode.startswith("random_controller_")
        )
        if random_competition or (
            response is not None and response.association_id in selected_habit_ids
        ):
            selected_by_random_habit = bool(
                controller.arbitration is not None
                and controller.arbitration.mode == "random_controller_habitual"
            )
            self._attach_habitual_trace(
                decision,
                response,
                awareness="conscious",
                conflict=False,
                arbiter_mode=(
                    "habitual"
                    if selected_by_random_habit
                    else (
                        "goal_directed"
                        if random_competition
                        else ("combine" if candidate.generated_by_goal else "habitual")
                    )
                ),
                arbiter_reason=decision.arbiter_reason,
                conflict_reason="",
                awareness_reason=(
                    awareness_gate.awareness.reason
                    if awareness_gate is not None and awareness_gate.awareness is not None
                    else ""
                ),
            )
        self._attach_habitual_gate(decision, gate, retrieval_status=retrieval_status)
        return decision

    def _decide_configured_goal_directed(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        perception: PerceiveResult,
    ) -> AgentDecision:
        if self.computation_architecture == "without_world_model":
            result = run_without_world_model_controller(
                agent,
                engine,
                perception,
                action_executor=self.action_executor,
                action_sample_window=self.action_sample_window,
                provider_name=self.provider_name,
                model=self.model,
            )
            self.last_target_resolver_error = result.target_resolver_error
            return result.decision
        return self._decide_goal_directed(agent, engine, perception)

    def _decide_goal_directed(self, agent: Agent, engine: PhysicsEngine, perception: PerceiveResult) -> AgentDecision:
        mode = _normalize_intuition_mode(self.intuition_mode)
        if self.action_executor.has_pending_graph_action():
            pending_action_text = self.action_executor.pending_graph_action_text()
            if pending_action_text:
                decision = AgentDecision(
                    action_type="action",
                    reason=pending_action_text,
                    action_proposal_text=pending_action_text,
                )
                if mode == "shadow":
                    self._attach_shadow_intuition(agent, engine, perception, decision)
                return decision

        working_memory = None
        intuition = None
        action_intuition = None
        pre_think_intuition = None
        think_result = None
        if mode != "off" and agent.active_intent is not None and agent.active_intent.status == "active":
            working_memory = build_working_memory(
                agent,
                engine,
                perception,
                intent=agent.active_intent,
            )
            intuition = self._generate_intuition(
                agent,
                engine,
                working_memory,
            )
            if mode == "active" and intuition is not None and intuition.route == "think":
                pre_think_intuition = intuition
                think_result, post_think_intuition = self._think_then_route(
                    agent,
                    engine,
                    perception,
                    working_memory,
                    intuition,
                )
                if post_think_intuition is not None:
                    routed_decision = self._decision_from_intuition(
                        agent,
                        engine,
                        post_think_intuition,
                    )
                    if routed_decision is not None:
                        self._attach_think_result(routed_decision, think_result)
                        return routed_decision
                    action_intuition = post_think_intuition
            else:
                routed_decision = self._decision_from_intuition(
                    agent,
                    engine,
                    intuition,
                )
                if mode == "active" and routed_decision is not None:
                    return routed_decision
                if mode == "active":
                    action_intuition = intuition

        if mode != "active":
            action_intuition = None
        action_proposal = self._propose_next_action(
            agent,
            engine,
            perception,
            action_intuition,
            pre_think_intuition=pre_think_intuition,
            think_result=think_result,
        )
        resolved = self._resolve_action_proposal(agent, engine, perception, action_proposal)
        if resolved is not None:
            if mode == "active" and action_intuition is not None:
                resolved.intuition_route = action_intuition.route
                resolved.intuition_thought = action_intuition.thought
                self._attach_think_result(resolved, think_result)
            elif mode == "shadow" and intuition is not None:
                resolved.shadow_intuition_route = intuition.route
                resolved.shadow_intuition_thought = intuition.thought
            return resolved

        if action_proposal is not None and action_proposal.action_text:
            if self.last_target_resolver_error:
                raise RuntimeError(
                    f"Target resolver failed for proposal: {action_proposal.action_text}. "
                    f"{self.last_target_resolver_error}"
                )
            raise RuntimeError(
                f"Unresolved action proposal: {action_proposal.action_text}"
            )
        return AgentDecision(
            action_type="wait",
            reason=f"{agent.name} has not yet formed a sufficiently clear next action.",
        )

    def _decision_from_habitual_response(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        perception: PerceiveResult,
        response: PreparedHabitualResponse,
    ) -> AgentDecision:
        return self._decision_from_arbiter_action(
            agent,
            engine,
            perception,
            response.response_text,
        )

    def _decision_from_arbiter_action(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        perception: PerceiveResult,
        action_text: str,
    ) -> AgentDecision:
        proposal = ActionProposalResult(
            action_text=action_text,
            provider_name=self.provider_name,
            model=self.model,
        )
        resolved = self._resolve_action_proposal(agent, engine, perception, proposal)
        if resolved is not None:
            return resolved
        return AgentDecision(
            action_type="action",
            reason=action_text,
            action_proposal_text=action_text,
        )

    def _attach_habitual_trace(
        self,
        decision: AgentDecision,
        response: PreparedHabitualResponse,
        *,
        awareness: str,
        conflict: bool,
        arbiter_mode: str,
        arbiter_reason: str,
        conflict_reason: str,
        awareness_reason: str,
    ) -> None:
        decision.habitual_response_key = response.response_key
        decision.habitual_association_id = response.association_id
        decision.habitual_response_text = response.response_text
        decision.habitual_dimension = response.source_dimension
        decision.habitual_activation = response.activation
        decision.habitual_awareness = awareness
        decision.habitual_conflict = conflict
        decision.habitual_conflict_reason = conflict_reason
        decision.habitual_awareness_reason = awareness_reason
        decision.arbiter_mode = arbiter_mode
        decision.arbiter_reason = arbiter_reason

    def _evaluate_emprical_gate(self, agent, *, conscious):
        result = evaluate_gate(
            agent.desire_state.internal_state, has_habit=True, conscious=conscious,
            enabled=self.emprical_gate_enabled, overrides=self.emprical_gate_overrides,
            rng=self.emprical_gate_rng,
        )
        return result

    def _apply_emprical_gate_fixed_state(self, agent):
        if not self.emprical_gate_freeze_state:
            return
        internal = agent.desire_state.internal_state
        for key in ("stress", "depletion", "cognitive_load"):
            setattr(internal, key, int(self.emprical_gate_overrides[key]))

    def _record_emprical_gate_action(self, decision, action_result):
        if action_result.execution_kind == "move_precondition":
            control = "positioning"
        elif decision.habitual_response_key and decision.arbiter_mode in {"fast_path", "habitual"}:
            control = "habitual"
        elif decision.habitual_response_key and decision.arbiter_mode in {"combine", "sequence"}:
            control = "combined"
        else:
            control = "goal_directed"
        self.emprical_gate_action_history.append({
            "step_id": decision.cognitive_step_id,
            "control": control,
            "action": decision.action_proposal_text or decision.reason,
            "feedback": action_result.feedback_text(),
            "counts_as_intent_action": action_result.counts_as_intent_action,
            "gate": dict(decision.emprical_gate_trace),
        })

    def _attach_emprical_gate(self, decision, result):
        if result is not None:
            decision.emprical_gate_trace = result.to_dict()
            self.emprical_gate_history.append(decision.emprical_gate_trace)
            if result.reason == "suppressed":
                decision.arbiter_reason = (
                    f"Emprical gate suppressed Goal-Directed: F={result.suppression_probability:.3f}, "
                    f"sample={result.sample:.3f}; conscious habit retained."
                )

    def _attach_habitual_gate(
        self,
        decision: AgentDecision,
        gate,
        *,
        retrieval_status: str,
    ) -> None:
        decision.habitual_gate_decision = "check" if gate.should_retrieve else "skip"
        decision.habitual_gate_probability = gate.probability
        decision.habitual_gate_sample = gate.sample
        decision.habitual_retrieval_status = retrieval_status

    def _copy_goal_directed_trace(
        self,
        source: AgentDecision,
        target: AgentDecision,
    ) -> None:
        target.intuition_route = source.intuition_route
        target.intuition_thought = source.intuition_thought
        target.think_thought = source.think_thought
        target.think_conclusion = source.think_conclusion

    def _record_habitual_execution(
        self,
        agent: Agent,
        decision: AgentDecision,
        action_result: ActionResult,
    ) -> None:
        if not decision.habitual_association_id:
            return
        if decision.arbiter_mode not in {"fast_path", "habitual", "combine"}:
            return
        if action_result.error:
            return
        current_time = getattr(getattr(agent, "time_belief", None), "current_datetime", None)
        if current_time is not None:
            agent.habitual_last_executed_at[decision.habitual_association_id] = current_time

    def _attach_shadow_intuition(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        perception: PerceiveResult,
        decision: AgentDecision,
    ) -> None:
        if agent.active_intent is None or agent.active_intent.status != "active":
            return
        working_memory = build_working_memory(
            agent,
            engine,
            perception,
            intent=agent.active_intent,
        )
        intuition = self._generate_intuition(
            agent,
            engine,
            working_memory,
        )
        if intuition is None:
            return
        decision.shadow_intuition_route = intuition.route
        decision.shadow_intuition_thought = intuition.thought

    def _propose_next_action(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        perception: PerceiveResult,
        intuition: IntuitionResult | None,
        *,
        pre_think_intuition: IntuitionResult | None = None,
        think_result: ThinkResult | None = None,
    ) -> ActionProposalResult | None:
        if agent.active_intent is None or agent.active_intent.status != "active":
            return None
        try:
            return propose_next_action(
                agent_name=perception.agent_name,
                intent=agent.active_intent,
                short_time_memory=getattr(agent, "short_time_memory", None),
                intuition=intuition,
                pre_think_intuition=pre_think_intuition,
                think_result=think_result,
                spatial_belief=getattr(agent, "belief", None),
                current_area_id=perception.area_id,
                scene_name=_scene_name(engine),
                provider_name=self.provider_name,
                model=self.model,
            )
        except Exception:
            return None

    def _generate_intuition(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        working_memory,
    ) -> IntuitionResult | None:
        if agent.active_intent is None:
            return None
        try:
            return generate_intuition(
                agent_name=agent.name,
                intent=agent.active_intent,
                working_memory=working_memory,
                engine=engine,
                provider_name=self.provider_name,
                model=self.model,
            )
        except Exception:
            return None

    def _think_then_route(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        perception: PerceiveResult,
        working_memory,
        intuition: IntuitionResult,
    ) -> tuple[ThinkResult | None, IntuitionResult | None]:
        if agent.active_intent is None:
            return None, None
        try:
            think_result = generate_think(
                agent_name=agent.name,
                intent=agent.active_intent,
                working_memory=working_memory,
                intuition=intuition,
                provider_name=self.provider_name,
                model=self.model,
            )
            self._remember_think_result(agent, perception, intuition, think_result)
            post_think_working_memory = build_working_memory(
                agent,
                engine,
                perception,
                intent=agent.active_intent,
            )
            post_think_intuition = route_after_thinking(
                agent_name=agent.name,
                intent=agent.active_intent,
                working_memory=post_think_working_memory,
                think_result=think_result,
                engine=engine,
                provider_name=self.provider_name,
                model=self.model,
            )
            self.last_think_error = ""
            return think_result, post_think_intuition
        except Exception as error:
            self.last_think_error = str(error)
            return None, None

    def _remember_think_result(
        self,
        agent: Agent,
        perception: PerceiveResult,
        intuition: IntuitionResult,
        think_result: ThinkResult,
    ) -> None:
        memory = getattr(agent, "short_time_memory", None)
        if memory is None:
            return
        if agent.active_intent is not None and not memory.intent_text:
            memory.intent_text = agent.active_intent.intent_text
        memory.remember(
            step_id=perception.step_id,
            area_id=perception.area_id,
            area_name=perception.area_name,
            perceived_summary="",
            intended_action=f"{agent.name} thought carefully about it.",
            experienced_result=think_result.format_for_memory(),
            intuition_route=intuition.route,
            intuition_thought=intuition.thought,
            intuition_mode="active",
            time_text=_time_memory_label(agent),
        )

    def _attach_think_result(
        self,
        decision: AgentDecision,
        think_result: ThinkResult | None,
    ) -> None:
        if think_result is None:
            return
        decision.think_thought = think_result.thought
        decision.think_conclusion = think_result.conclusion

    def _decision_from_intuition(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        intuition: IntuitionResult | None,
    ) -> AgentDecision | None:
        if intuition is None or intuition.route == "action":
            return None

        if intuition.route == "wait":
            return AgentDecision(
                action_type="wait",
                reason=intuition.thought,
                action_proposal_text=intuition.thought,
                intuition_route=intuition.route,
                intuition_thought=intuition.thought,
                wait_duration=intuition.wait_duration,
            )

        if intuition.route == "chat":
            return AgentDecision(
                action_type="chat",
                reason=intuition.thought,
                action_proposal_text=intuition.thought,
                intuition_route=intuition.route,
                intuition_thought=intuition.thought,
            )

        if intuition.route == "think":
            return None

        if intuition.route == "walk":
            target_area_id = self._resolve_intuition_target_area(engine, intuition)
            if target_area_id:
                return AgentDecision(
                    action_type="move_to_area",
                    target_area_id=target_area_id,
                    reason=intuition.thought,
                    action_proposal_text=intuition.thought,
                    intuition_route=intuition.route,
                    intuition_thought=intuition.thought,
                )
            return AgentDecision(
                action_type="wait",
                reason=f"{agent.name} wants to move around, but hasn't decided where to go yet.",
                action_proposal_text=intuition.thought,
                intuition_route=intuition.route,
                intuition_thought=intuition.thought,
            )

        return None

    def _resolve_intuition_target_area(
        self,
        engine: PhysicsEngine,
        intuition: IntuitionResult,
    ) -> str:
        target_area_id = intuition.target_area_id.strip()
        if target_area_id and engine.get_area(target_area_id) is not None:
            return target_area_id

        target_area_name = intuition.target_area_name.strip()
        thought = intuition.thought.strip()
        for area in engine.home.areas:
            if target_area_name and (
                target_area_name == area.name
                or target_area_name == area.node_id
                or target_area_name in area.name
                or area.name in target_area_name
            ):
                return area.node_id
        for area in engine.home.areas:
            if area.name and area.name in thought:
                return area.node_id
        return ""

    def _resolve_action_proposal(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        perception: PerceiveResult,
        action_proposal: ActionProposalResult | None,
    ) -> AgentDecision | None:
        if action_proposal is None or not action_proposal.action_text:
            return None

        self.last_target_resolver_error = ""
        target = self._resolve_target_with_llm(agent, engine, perception, action_proposal)
        return self._decision_from_target_resolution(
            agent,
            engine,
            action_proposal,
            target,
        )

    def _decision_from_target_resolution(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        action_proposal: ActionProposalResult,
        target: TargetResolution | None,
    ) -> AgentDecision:
        return decision_from_target_resolution(action_proposal, target)

    def _target_resolution_trace(self, target: TargetResolution | None) -> dict:
        return target_resolution_trace(target)

    def _resolve_target_with_llm(
        self,
        agent: Agent,
        engine: PhysicsEngine,
        perception: PerceiveResult,
        action_proposal: ActionProposalResult,
    ) -> TargetResolution:
        try:
            return resolve_target_with_llm(
                engine,
                agent_name=perception.agent_name,
                action_text=action_proposal.action_text,
                current_area_id=perception.area_id,
                current_area_name=perception.area_name,
                provider_name=self.provider_name,
                model=self.model,
                spatial_belief=agent.belief,
                world_graph_context=self.action_executor.target_resolver_context(
                    agent,
                    engine,
                ),
            )
        except Exception as error:
            self.last_target_resolver_error = str(error)
            fallback = resolve_target_fallback(
                engine,
                action_text=action_proposal.action_text,
                current_area_id=perception.area_id,
                spatial_belief=agent.belief,
            )
            fallback.prompt = str(getattr(error, "prompt", "") or "")
            fallback.raw_response = str(getattr(error, "raw_response", "") or "")
            fallback.provider_name = self.provider_name
            fallback.model = self.model or "provider default"
            fallback.duration_seconds = getattr(error, "duration_seconds", None)
            fallback.error = str(error)
            if DEBUG_TARGET_RESOLVER:
                print("\n============================================================")
                print("Target Resolver Error")
                print(f"Action Proposal: {action_proposal.action_text}")
                print(f"Error: {error}")
                raw_response = getattr(error, "raw_response", "")
                if raw_response:
                    print("Raw LLM Output:")
                    print(raw_response)
                prompt = getattr(error, "prompt", "")
                if prompt and not raw_response:
                    print("Prompt Sent To LLM:")
                    print(prompt)
                print("Fallback Target:")
                print(
                    {
                        "target_element_id": fallback.target_element_id,
                        "target_element_name": fallback.target_element_name,
                        "secondary_target_element_id": fallback.secondary_target_element_id,
                        "arrival_completes_action": fallback.arrival_completes_action,
                        "reason": fallback.reason,
                    }
                )
                print(traceback.format_exc().rstrip())
            return fallback

    def _ensure_active_intent(self, agent: Agent, perception: PerceiveResult) -> None:
        if agent.active_intent is not None and agent.active_intent.status == "active":
            if agent.active_intent.decided_at is None:
                agent.active_intent.decided_at = agent.time_belief.current_datetime
            return
        try:
            intent = propose_new_intent(
                perception,
                agent_name=agent.name,
                provider_name=self.provider_name,
                model=self.model,
                previous_feedback=self.previous_execution_result,
            )
            agent.active_intent = IntentState(
                intent_text=intent.intent_text,
                decided_at=agent.time_belief.current_datetime,
            )
            self.last_lifecycle_reviewed_step_count = 0
        except Exception:
            if agent.active_intent is None:
                agent.active_intent = IntentState(
                    intent_text=f"{agent.name} wants to look around first.",
                    decided_at=agent.time_belief.current_datetime,
                )
                self.last_lifecycle_reviewed_step_count = 0
            else:
                agent.active_intent.status = "active"
                agent.active_intent.step_count = 0

    def _record_intent_action(
        self,
        agent: Agent,
        decision: AgentDecision,
        action_result: ActionResult,
    ) -> None:
        if agent.active_intent is None:
            return
        if not action_result.counts_as_intent_action:
            return
        agent.active_intent.step_count += 1
        if self._contributes_to_intent_progress(decision):
            self._update_intent_progress(agent, decision, action_result)
        agent.active_intent.status = "active"

    def _contributes_to_intent_progress(self, decision: AgentDecision) -> bool:
        if not decision.habitual_response_key:
            return True
        return decision.arbiter_mode in {"goal_directed", "inhibit_habit", "sequence", "combine"}

    def review_active_intent(self, agent: Agent) -> IntentLifecycleResult | None:
        intent = agent.active_intent
        if intent is None or intent.status != "active" or intent.step_count <= 0:
            return None
        if intent.step_count == self.last_lifecycle_reviewed_step_count:
            return None
        self.last_lifecycle_reviewed_step_count = intent.step_count
        try:
            force_reflect = (
                self.intent_reflection_interval > 0
                and intent.step_count % self.intent_reflection_interval == 0
            )
            reflection = evaluate_intent_lifecycle(
                agent_name=agent.name,
                personality=agent.personality,
                current_state_belief=self._lifecycle_current_state_belief(agent),
                intent=intent,
                intent_decided_time=self._intent_decided_time_text(agent),
                current_time=self._current_time_text(agent),
                short_time_memory_text=self._lifecycle_memory_text(agent),
                force_reflect=force_reflect,
                provider_name=self.provider_name,
                model=self.model,
            )
            intent.status = reflection.intent_status
            if reflection.updated_intent:
                intent.intent_text = reflection.updated_intent
            if reflection.intent_status != "active":
                agent.active_intent = None
            return reflection
        except Exception:
            return None

    def _intent_decided_time_text(self, agent: Agent) -> str:
        if agent.active_intent is None:
            return "A short while ago"
        time_belief = getattr(agent, "time_belief", None)
        formatter = getattr(time_belief, "format_elapsed_since", None)
        if callable(formatter):
            return str(formatter(agent.active_intent.decided_at) or "A short while ago")
        return "A short while ago"

    def _lifecycle_current_state_belief(
        self,
        agent: Agent,
    ) -> str:
        parts = []
        self_state = getattr(agent, "self_state", None)
        area_name = str(getattr(self_state, "current_area_name", "") or "").strip()
        posture = {
            "standing": "standing",
            "sitting": "sitting",
            "lying": "lying down",
            "crouching": "crouching",
        }.get(str(getattr(agent, "posture", "") or "").strip(), "")
        if area_name and posture:
            parts.append(f"You are currently in {area_name}, {posture}.")
        elif area_name:
            parts.append(f"You are currently in {area_name}.")
        elif posture:
            parts.append(f"You are currently {posture}.")

        current_belief = self._second_person_text(
            agent,
            str(getattr(self_state, "self_belief", "") or ""),
        )
        if current_belief:
            parts.append(f"Your understanding of your current state is: {current_belief}")
        blocked_markers = list(getattr(self_state, "blocked_or_failed_markers", []) or [])
        if blocked_markers:
            parts.append("You have encountered a clear obstacle in this step.")
        return " ".join(parts) or "You have just completed this immediate step."

    def _second_person_text(self, agent: Agent, text: str) -> str:
        cleaned = str(text or "").strip()
        if agent.name:
            cleaned = cleaned.replace(agent.name, "You")
        return cleaned

    def _current_time_text(self, agent: Agent) -> str:
        time_belief = getattr(agent, "time_belief", None)
        formatter = getattr(time_belief, "format_short_label", None)
        if callable(formatter):
            return str(formatter() or "Now")
        return "Now"

    def _lifecycle_memory_text(
        self,
        agent: Agent,
    ) -> str:
        lines = []
        memory = getattr(agent, "short_time_memory", None)
        if memory is not None:
            for episode in memory.episodes:
                if not getattr(episode, "recallable", True):
                    continue
                summary = episode.concise_summary().strip()
                if not summary:
                    continue
                prefix = episode.bracket_text()
                lines.append(f"- {prefix} {summary}" if prefix else f"- {summary}")

        if not lines and agent.active_intent is not None:
            lines.extend(f"- {item}" for item in agent.active_intent.progress if item)
        return "\n".join(lines) if lines else "I haven't done anything yet."

    def _store_world_feedback_summary(self, agent: Agent, action_result: ActionResult) -> None:
        feedback = action_result.environment_feedback
        if feedback is None:
            agent.last_world_feedback_summary = ""
            return
        agent.last_world_feedback_summary = feedback.perception_summary.strip()

    def _advance_time_belief(self, agent: Agent, action_result: ActionResult) -> None:
        time_belief = getattr(agent, "time_belief", None)
        advance = getattr(time_belief, "advance", None)
        if callable(advance):
            advance(action_result.estimated_duration)

    def _update_intent_progress(self, agent: Agent, decision: AgentDecision, action_result: ActionResult) -> None:
        if agent.active_intent is None:
            return
        action_text = decision.action_proposal_text or decision.reason
        feedback = action_result.feedback_text()
        item = ""
        try:
            result = summarize_intent_progress(
                agent_name=agent.name,
                intent=agent.active_intent,
                action_proposal=action_text,
                execution_feedback=feedback,
                provider_name=self.provider_name,
                model=self.model,
            )
            item = result.progress_item
        except Exception:
            item = fallback_progress_item(
                agent_name=agent.name,
                intent=agent.active_intent,
                action_proposal=action_text,
                execution_feedback=feedback,
            )
        append_progress(agent.active_intent, item)

    def _personal_context(self, agent: Agent, query: str = "") -> str:
        memory = getattr(agent, "long_term_memory", None)
        if memory is None:
            return ""
        retrieve = getattr(memory, "retrieve", None)
        if not callable(retrieve):
            return ""
        return retrieve(query=query)

def _time_memory_label(agent: Agent) -> str:
    time_belief = getattr(agent, "time_belief", None)
    formatter = getattr(time_belief, "format_short_label", None)
    if callable(formatter):
        return str(formatter() or "").strip()
    return ""


def _decision_think_text(decision: AgentDecision) -> str:
    parts = [part.strip() for part in [decision.think_thought, decision.think_conclusion] if part.strip()]
    return " ".join(dict.fromkeys(parts))

from __future__ import annotations

from dataclasses import dataclass, field

from agent.agent import Agent
from agent.intent.lifecycle import IntentLifecycleResult
from agent.interaction_loop import AgentInteractionLoop, AgentStepRecord
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from contextual_world.structure.scene_schema import Home
from contextual_world.structure import build_scene
from contextual_world.world_old.world_runtime import WorldRuntime

DEFAULT_AGENT_START = (3.10, 5.10)  # living_room


@dataclass
class WorldStateChange:
    area_id: str
    element_id: str
    element_name: str
    old_physical_status: str
    new_physical_status: str
    old_evolution_status: str
    new_evolution_status: str
    old_interaction_status: str
    new_interaction_status: str
    old_state_details: dict[str, str] = field(default_factory=dict)
    new_state_details: dict[str, str] = field(default_factory=dict)


@dataclass
class EngineStepResult:
    step_id: int
    agent_name: str
    perception_text: str
    perception_notice_text: str
    intent_text: str
    intent_status: str
    intent_action_index: int
    intent_progress: list[str]
    action_proposal_text: str
    decision_text: str
    action_type: str
    execution_kind: str
    target_area_id: str | None
    target_element_id: str | None
    navigation_target_element_id: str | None
    secondary_target_element_id: str | None
    action_text: str
    execution_narration: str
    estimated_duration: str
    temporary_element_changes: list[dict]
    actor_state: dict
    actor_state_update: dict
    environment_feedback: dict
    final_position: tuple[float, float]
    counts_as_intent_action: bool = True
    time_text: str = ""
    clock_time: str = ""
    desire_state: dict = field(default_factory=dict)
    desire_update: dict = field(default_factory=dict)
    self_belief: str = ""
    intent_lifecycle_reason: str = ""
    wait_duration: str = ""
    intuition_route: str = ""
    intuition_thought: str = ""
    think_thought: str = ""
    think_conclusion: str = ""
    shadow_intuition_route: str = ""
    shadow_intuition_thought: str = ""
    execution_debug: dict = field(default_factory=dict)
    world_changes: list[WorldStateChange] = field(default_factory=list)
    habitual_response_key: str = ""
    habitual_gate_decision: str = ""
    habitual_gate_probability: float = 0.0
    habitual_gate_sample: float = 0.0
    habitual_retrieval_status: str = ""
    habitual_association_id: str = ""
    habitual_response_text: str = ""
    habitual_dimension: str = ""
    habitual_activation: float = 0.0
    habitual_awareness: str = ""
    habitual_conflict: bool = False
    habitual_conflict_reason: str = ""
    habitual_awareness_reason: str = ""
    arbiter_mode: str = ""
    arbiter_reason: str = ""
    computation_architecture: str = "intuition"
    world_model_trace: dict = field(default_factory=dict)
    emprical_gate_trace: dict = field(default_factory=dict)
    action_generation_trace: dict = field(default_factory=dict)
    short_time_memory_encoded: bool = True
    short_time_memory_encoding_probability: float = 1.0
    short_time_memory_encoding_sample: float | None = None


class EnvironmentInteractionEngine:
    def __init__(
        self,
        *,
        home: Home | None = None,
        scene_name: str = "unity_home",
        agent: Agent | None = None,
        agent_name: str = "Agent",
        provider_name: str = "ollama",
        model: str | None = None,
        intent_reflection_interval: int = 10,
        intuition_mode: str = "active",
        habitual_retrieval_probability: float = 0.9,
        habitual_activation_threshold: float = 0.35,
        habitual_awareness_mode: str = "llm",
        unconscious_habit_memory_probability: float = 0.2,
        computation_architecture: str = "intuition",
        world_model_mode: str = "complex",
        action_sample_window: int | None = None,
        iteration_number: int = 3,
        recent_experience_count: int = 8,
        selection_method: str = "greedy",
        selection_temperature: float = 0.2,
        selection_epsilon: float = 0.1,
        planning_strategy: str = "greedy",
        beam_width: int = 3,
        planning_branching_factor: int = 3,
        mcts_simulations: int = 12,
        mcts_exploration_constant: float = 2 ** 0.5,
        arbiter_mode: str = "weighted",
        arbiter_random_seed: int | None = None,
        world_failure_output_path: str | None = None,
        interaction_reach: float = 0.0,
        disable_spatial_gating: bool = True,
    ) -> None:
        self.scene_name = scene_name if home is None else home.node_id
        self.home = home or build_scene(scene_name)
        self.physics = PhysicsEngine(self.home)
        self.world_runtime = WorldRuntime(
            self.home,
            scene_name=scene_name,
            provider_name=provider_name,
            model=model,
        )
        agent_start = self.home.default_agent_start or DEFAULT_AGENT_START
        self.agent = agent or Agent(node_id="agent_01", name=agent_name, center=agent_start)
        self.agent.initialize_belief(self.physics)
        self.world_runtime.register_actor(
            actor_id=self.agent.node_id,
            name=self.agent.name,
            center=self.agent.center,
            size=self.agent.size,
            facing=self.agent.facing,
            current_area_id=self.agent.current_area_id,
            posture=self.agent.posture,
            interaction_elements=self.agent.interaction_elements,
            interaction_method=self.agent.interaction_method,
            gaze_target=self.agent.gaze_target,
            worn_items=self.agent.worn_items,
            body_surface=self.agent.body_surface,
            text_to_motion_description=self.agent.text_to_motion_description,
        )
        self.loop = AgentInteractionLoop(
            provider_name=provider_name,
            model=model,
            intent_reflection_interval=intent_reflection_interval,
            intuition_mode=intuition_mode,
            habitual_retrieval_probability=habitual_retrieval_probability,
            habitual_activation_threshold=habitual_activation_threshold,
            habitual_awareness_mode=habitual_awareness_mode,
            unconscious_habit_memory_probability=unconscious_habit_memory_probability,
            computation_architecture=computation_architecture,
            world_model_mode=world_model_mode,
            action_sample_window=action_sample_window,
            iteration_number=iteration_number,
            recent_experience_count=recent_experience_count,
            selection_method=selection_method,
            selection_temperature=selection_temperature,
            selection_epsilon=selection_epsilon,
            planning_strategy=planning_strategy,
            beam_width=beam_width,
            planning_branching_factor=planning_branching_factor,
            mcts_simulations=mcts_simulations,
            mcts_exploration_constant=mcts_exploration_constant,
            arbiter_mode=arbiter_mode,
            arbiter_random_seed=arbiter_random_seed,
            world_failure_output_path=world_failure_output_path,
            interaction_reach=interaction_reach,
            disable_spatial_gating=disable_spatial_gating,
        )
    def step(self) -> EngineStepResult:
        return self.action_turn()

    def action_turn(self) -> EngineStepResult:
        before = self._snapshot_statuses()
        record = self.loop.run_action_turn(self.agent, self.physics)
        self._sync_registered_actor()
        after = self._snapshot_statuses()
        return self._build_step_result(record, before, after)

    def run_intent_cycle(self, *, max_action_turns: int = 30) -> list[EngineStepResult]:
        results: list[EngineStepResult] = []
        for _ in range(max_action_turns):
            before = self._snapshot_statuses()
            record = self.loop.run_action_turn(self.agent, self.physics)
            self._sync_registered_actor()
            after = self._snapshot_statuses()
            result = self._build_step_result(record, before, after)
            results.append(result)
            reflection = self.review_active_intent()
            if reflection is not None:
                result.intent_status = reflection.intent_status
                if reflection.intent_status != "active":
                    result.intent_lifecycle_reason = reflection.reason or "(The model did not provide a reason)"
            if self.agent.active_intent is None or result.intent_status != "active":
                break
        return results

    def run_single_intent_cycle(self, *, max_action_turns: int = 30) -> list[EngineStepResult]:
        return self.run_intent_cycle(max_action_turns=max_action_turns)

    def _build_step_result(
        self,
        record: AgentStepRecord,
        before: dict[str, tuple[str, str, str, str, str]],
        after: dict[str, tuple[str, str, str, str, str]],
    ) -> EngineStepResult:
        return EngineStepResult(
            step_id=record.step_id,
            agent_name=self.agent.name,
            perception_text=record.narration_text,
            perception_notice_text=record.notice_text,
            intent_text=record.intent_text,
            intent_status=record.intent_status,
            intent_action_index=record.intent_action_index,
            intent_progress=record.intent_progress,
            intent_lifecycle_reason="",
            action_proposal_text=record.decision.action_proposal_text,
            intuition_route=record.decision.intuition_route,
            intuition_thought=record.decision.intuition_thought,
            think_thought=record.decision.think_thought,
            think_conclusion=record.decision.think_conclusion,
            shadow_intuition_route=record.decision.shadow_intuition_route,
            shadow_intuition_thought=record.decision.shadow_intuition_thought,
            habitual_response_key=record.decision.habitual_response_key,
            habitual_gate_decision=record.decision.habitual_gate_decision,
            habitual_gate_probability=record.decision.habitual_gate_probability,
            habitual_gate_sample=record.decision.habitual_gate_sample,
            habitual_retrieval_status=record.decision.habitual_retrieval_status,
            habitual_association_id=record.decision.habitual_association_id,
            habitual_response_text=record.decision.habitual_response_text,
            habitual_dimension=record.decision.habitual_dimension,
            habitual_activation=record.decision.habitual_activation,
            habitual_awareness=record.decision.habitual_awareness,
            habitual_conflict=record.decision.habitual_conflict,
            habitual_conflict_reason=record.decision.habitual_conflict_reason,
            habitual_awareness_reason=record.decision.habitual_awareness_reason,
            arbiter_mode=record.decision.arbiter_mode,
            arbiter_reason=record.decision.arbiter_reason,
            computation_architecture=record.decision.computation_architecture,
            world_model_trace=dict(record.decision.world_model_trace or {}),
            emprical_gate_trace=dict(record.decision.emprical_gate_trace or {}),
            action_generation_trace=dict(record.decision.action_generation_trace or {}),
            decision_text=record.decision.reason,
            action_type=record.decision.action_type,
            execution_kind=record.execution_kind,
            target_area_id=record.decision.target_area_id,
            target_element_id=record.decision.target_element_id,
            navigation_target_element_id=record.decision.navigation_target_element_id,
            secondary_target_element_id=record.decision.secondary_target_element_id,
            action_text=record.action_summary,
            execution_narration=record.execution_narration,
            estimated_duration=record.estimated_duration,
            wait_duration=record.decision.wait_duration,
            temporary_element_changes=record.temporary_element_changes,
            actor_state=self._actor_state_payload(),
            actor_state_update=record.actor_state_update,
            environment_feedback=record.environment_feedback,
            execution_debug=dict(record.execution_debug or {}),
            final_position=record.final_position,
            counts_as_intent_action=record.counts_as_intent_action,
            time_text=self.agent.time_belief.format_short_label(),
            clock_time=self.agent.time_belief.current_datetime.strftime("%H:%M:%S"),
            desire_state=self.agent.desire_state.to_dict(),
            self_belief=str(getattr(self.agent.self_state, "self_belief", "") or ""),
            world_changes=self._diff_world_states(before, after),
        )

    def run_steps(self, count: int) -> list[EngineStepResult]:
        results: list[EngineStepResult] = []
        for _ in range(count):
            results.append(self.step())
        return results

    def review_active_intent(self) -> IntentLifecycleResult | None:
        return self.loop.review_active_intent(self.agent)

    def inject_status(self, element_id: str, status: str) -> None:
        self.home.update_element_status(status, node_id=element_id)

    def reset_new_game(self) -> None:
        self.home.reset_all_statuses()
        self.agent.reset_for_new_game(self.physics)
        self._sync_registered_actor()
        self.loop.previous_decision_text = ""
        self.loop.previous_execution_result = ""
        self.loop.last_lifecycle_reviewed_step_count = 0
        self.loop.action_executor.clear_pending_graph_action()
        if self.loop.action_executor.graph_pipe is not None:
            self.loop.action_executor.graph_pipe.reset()

    def _sync_registered_actor(self) -> None:
        area = self.physics.get_area_for_point(*self.agent.center)
        self.world_runtime.sync_actor(
            self.agent.node_id,
            center=self.agent.center,
            size=self.agent.size,
            facing=self.agent.facing,
            current_area_id=area.node_id if area is not None else None,
            posture=self.agent.posture,
            interaction_elements=self.agent.interaction_elements,
            interaction_method=self.agent.interaction_method,
            gaze_target=self.agent.gaze_target,
            worn_items=self.agent.worn_items,
            body_surface=self.agent.body_surface,
            text_to_motion_description=self.agent.text_to_motion_description,
        )

    def _actor_state_payload(self) -> dict:
        return {
            "facing": self.agent.facing,
            "field_of_view_degrees": self.agent.field_of_view_degrees,
            "posture": self.agent.posture,
            "interaction_elements": list(self.agent.interaction_elements or []),
            "interaction_method": self.agent.interaction_method,
            "gaze_target": self.agent.gaze_target,
            "worn_items": list(self.agent.worn_items or []),
            "body_surface": self.agent.body_surface,
            "text_to_motion_description": self.agent.text_to_motion_description,
        }

    def _snapshot_statuses(self) -> dict[str, tuple[str, str, str, str, str, dict[str, str]]]:
        snapshot: dict[str, tuple[str, str, str, str, str, dict[str, str]]] = {}
        for area in self.home.areas:
            for element in area.elements:
                snapshot[element.node_id] = (
                    area.node_id,
                    element.name,
                    element.physical_status,
                    element.evolution_status,
                    element.interaction_status,
                    dict(element.state_details),
                )
        return snapshot

    def _diff_world_states(
        self,
        before: dict[str, tuple[str, str, str, str, str, dict[str, str]]],
        after: dict[str, tuple[str, str, str, str, str, dict[str, str]]],
    ) -> list[WorldStateChange]:
        changes: list[WorldStateChange] = []
        for element_id, (
            area_id,
            element_name,
            new_physical_status,
            new_evolution_status,
            new_interaction_status,
            new_state_details,
        ) in after.items():
            old = before.get(element_id)
            if old is None:
                continue
            _, _, old_physical_status, old_evolution_status, old_interaction_status, old_state_details = old
            if (
                old_physical_status == new_physical_status
                and old_evolution_status == new_evolution_status
                and old_interaction_status == new_interaction_status
                and old_state_details == new_state_details
            ):
                continue
            changes.append(
                WorldStateChange(
                    area_id=area_id,
                    element_id=element_id,
                    element_name=element_name,
                    old_physical_status=old_physical_status,
                    new_physical_status=new_physical_status,
                    old_evolution_status=old_evolution_status,
                    new_evolution_status=new_evolution_status,
                    old_interaction_status=old_interaction_status,
                    new_interaction_status=new_interaction_status,
                    old_state_details=old_state_details,
                    new_state_details=new_state_details,
                )
            )
        return changes

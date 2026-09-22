from __future__ import annotations

from dataclasses import dataclass, field

from core.action_types import ActionResult, AgentDecision
from contextual_world.world_old.action_execution import WorldActionExecutor
from contextual_world.world_old.commonsense_rules import CommonsenseEngine, CommonsenseTickResult, RecentIntervention
from contextual_world.world_old.environment_mutation_api import (
    AppliedElementMutation,
    EnvironmentMutationAPI,
    EnvironmentMutationRequest,
    EnvironmentMutationResult,
)
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from contextual_world.structure.scene_schema import Home
from contextual_world.structure import build_scene
from contextual_world.world_old.user_probe.user_probe import ProbeApplyResult, UserProbe


DEFAULT_AGENT_WORN_ITEMS = ["jacket"]


@dataclass
class WorldEventChange:
    element_id: str
    field: str
    old: str
    new: str


@dataclass
class Agent:
    actor_id: str
    name: str
    center: tuple[float, float]
    size: tuple[float, float]
    facing: float
    current_area_id: str | None = None
    posture: str = "standing"
    interaction_elements: list[dict] = field(default_factory=list)
    interaction_method: str = ""
    gaze_target: str = ""
    worn_items: list[str] = field(default_factory=lambda: list(DEFAULT_AGENT_WORN_ITEMS))
    body_surface: str = "dry_clean"
    text_to_motion_description: str = ""


@dataclass
class WorldEvent:
    event_id: int
    turn: int
    kind: str
    summary: str
    actor_id: str = ""
    actor: str = ""
    target_ids: list[str] = field(default_factory=list)
    decision: str = ""
    changes: list[WorldEventChange] = field(default_factory=list)


class WorldRuntime:
    """Runtime facade for the mutable scene world.

    This class does not replace the existing world services. It keeps them
    wired to the same Home instance and gives callers one stable entrypoint for
    graph-based agent actions, user probes, direct mutations, and commonsense
    ticks.
    """

    def __init__(
        self,
        home: Home | None = None,
        *,
        scene_name: str = "unity_home",
        provider_name: str = "ollama",
        model: str | None = None,
    ) -> None:
        self.scene_name = scene_name if home is None else home.node_id
        self.home = home or build_scene(scene_name)
        self.provider_name = provider_name
        self.model = model

        self.physics = PhysicsEngine(self.home)
        self.mutation_api = EnvironmentMutationAPI(self.home)
        self.action_executor = WorldActionExecutor(
            provider_name=provider_name,
            model=model,
        )
        self.commonsense = CommonsenseEngine(
            self.home,
            provider_name=provider_name,
            model=model,
        )
        self.user_probe = UserProbe(
            self.home,
            provider_name=provider_name,
            model=model,
        )

        self.user_probe.mutation_api = self.mutation_api
        self._pending_interventions: list[RecentIntervention] = []
        self.events: list[WorldEvent] = []
        self.actors: dict[str, Agent] = {}
        self.turn = 0

    def register_actor(
        self,
        *,
        actor_id: str,
        name: str,
        center: tuple[float, float],
        size: tuple[float, float],
        facing: float,
        current_area_id: str | None = None,
        posture: str = "standing",
        interaction_elements: list[dict] | None = None,
        interaction_method: str = "",
        gaze_target: str = "",
        worn_items: list[str] | None = None,
        body_surface: str = "dry_clean",
        text_to_motion_description: str = "",
    ) -> Agent:
        if current_area_id is None:
            area = self.physics.get_area_for_point(*center)
            current_area_id = area.node_id if area is not None else None
        actor = Agent(
            actor_id=actor_id,
            name=name,
            center=center,
            size=size,
            facing=facing,
            current_area_id=current_area_id,
            posture=posture,
            interaction_elements=list(interaction_elements or []),
            interaction_method=interaction_method,
            gaze_target=gaze_target,
            worn_items=list(worn_items) if worn_items is not None else list(DEFAULT_AGENT_WORN_ITEMS),
            body_surface=body_surface,
            text_to_motion_description=text_to_motion_description,
        )
        self.actors[actor_id] = actor
        return actor

    def sync_actor(
        self,
        actor_id: str,
        *,
        center: tuple[float, float] | None = None,
        size: tuple[float, float] | None = None,
        facing: float | None = None,
        current_area_id: str | None = None,
        posture: str | None = None,
        interaction_elements: list[dict] | None = None,
        interaction_method: str | None = None,
        gaze_target: str | None = None,
        worn_items: list[str] | None = None,
        body_surface: str | None = None,
        text_to_motion_description: str | None = None,
    ) -> Agent | None:
        actor = self.actors.get(actor_id)
        if actor is None:
            return None
        if center is not None:
            actor.center = center
            if current_area_id is None:
                area = self.physics.get_area_for_point(*center)
                current_area_id = area.node_id if area is not None else None
        if size is not None:
            actor.size = size
        if facing is not None:
            actor.facing = facing
        if current_area_id is not None:
            actor.current_area_id = current_area_id
        if posture is not None:
            actor.posture = posture
        if interaction_elements is not None:
            actor.interaction_elements = list(interaction_elements)
        if interaction_method is not None:
            actor.interaction_method = interaction_method
        if gaze_target is not None:
            actor.gaze_target = gaze_target
        if worn_items is not None:
            actor.worn_items = list(worn_items)
        if body_surface is not None:
            actor.body_surface = body_surface
        if text_to_motion_description is not None:
            actor.text_to_motion_description = text_to_motion_description
        return actor

    def get_actor(self, actor_id: str) -> Agent | None:
        return self.actors.get(actor_id)

    def ensure_action_graph(self, actor_id: str):
        actor = self.actors.get(actor_id)
        if actor is None:
            raise ValueError(f"Unknown actor_id: {actor_id}")
        pipe = self.action_executor._ensure_graph_pipe(actor, self.physics)
        pipe.ensure_graph(actor, self.physics)
        return pipe

    def process_agent_action(
        self,
        *,
        actor_id: str,
        action_text: str,
    ) -> ActionResult:
        actor = self.actors.get(actor_id)
        if actor is None:
            raise ValueError(f"Unknown actor_id: {actor_id}")

        result = self.action_executor.execute(
            actor,
            self.physics,
            AgentDecision(
                action_type="action",
                reason=action_text,
                action_proposal_text=action_text,
            ),
        )
        transition_report = (
            result.environment_feedback.graph_transition_report
            if result.environment_feedback is not None
            else {}
        )
        self._append_event(
            kind="contextual_world_action",
            summary=result.feedback_text(),
            actor_id=actor_id,
            actor=actor.name,
            target_ids=self._target_ids_from_graph_transition(transition_report),
            decision="action",
            changes=[],
        )
        return result

    def apply_user_probe(
        self,
        user_command: str,
        *,
        actor_name: str,
        focus_element_name: str = "",
    ) -> ProbeApplyResult:
        result = self.user_probe.probe_and_apply(
            user_command,
            actor_name=actor_name,
            focus_element_name=focus_element_name,
        )
        self._append_event(
            kind="user_probe",
            summary=result.summary,
            actor_id=self._actor_id_for_name(actor_name),
            actor=actor_name,
            target_ids=[change.element_id for change in result.changes],
            decision="applied" if result.success else "failed",
            changes=[
                WorldEventChange(
                    element_id=change.element_id,
                    field="physical_status",
                    old=change.old_status,
                    new=change.new_status,
                )
                for change in result.changes
            ],
        )
        return result

    def apply_mutation(self, request: EnvironmentMutationRequest) -> EnvironmentMutationResult:
        result = self.mutation_api.apply(request)
        self._append_event(
            kind="direct_mutation",
            summary=result.summary,
            target_ids=[mutation.element_id for mutation in request.element_mutations],
            decision="applied" if result.success else "noop",
            changes=self._changes_from_mutation_result(result),
        )
        return result

    def record_intervention(
        self,
        *,
        actor_name: str,
        target_element_id: str,
        description: str,
    ) -> RecentIntervention:
        intervention = RecentIntervention(
            actor_name=actor_name,
            target_element_id=target_element_id,
            description=description,
        )
        self._pending_interventions.append(intervention)
        return intervention

    def advance_tick(
        self,
        *,
        recent_interventions: list[RecentIntervention] | None = None,
        consume_pending_interventions: bool = True,
    ) -> CommonsenseTickResult:
        if recent_interventions is None:
            interventions = list(self._pending_interventions)
            if consume_pending_interventions:
                self._pending_interventions.clear()
        else:
            interventions = list(recent_interventions)
        result = self.commonsense.advance_tick(recent_interventions=interventions)
        changes: list[WorldEventChange] = []
        target_ids: list[str] = []
        for update in result.applied_updates:
            target_ids.append(update.element_id)
            if update.from_physical_status != update.to_physical_status:
                changes.append(
                    WorldEventChange(
                        element_id=update.element_id,
                        field="physical_status",
                        old=update.from_physical_status,
                        new=update.to_physical_status,
                    )
                )
            if update.from_evolution_status != update.to_evolution_status:
                changes.append(
                    WorldEventChange(
                        element_id=update.element_id,
                        field="evolution_status",
                        old=update.from_evolution_status,
                        new=update.to_evolution_status,
                    )
                )
        if not target_ids:
            target_ids = [source.element_id for source in result.non_regular_sources]
        self._append_event(
            kind="commonsense_tick",
            summary=result.summary,
            target_ids=target_ids,
            decision="applied" if result.applied_updates else "noop",
            changes=changes,
        )
        return result

    def reset(self) -> None:
        self.home.reset_all_statuses()
        self.commonsense.reset()
        self._pending_interventions.clear()
        self.events.clear()
        self.turn = 0

    def _append_event(
        self,
        *,
        kind: str,
        summary: str,
        actor_id: str = "",
        actor: str = "",
        target_ids: list[str] | None = None,
        decision: str = "",
        changes: list[WorldEventChange] | None = None,
    ) -> WorldEvent:
        self.turn += 1
        event = WorldEvent(
            event_id=len(self.events) + 1,
            turn=self.turn,
            kind=kind,
            summary=summary,
            actor_id=actor_id,
            actor=actor,
            target_ids=list(dict.fromkeys(target_ids or [])),
            decision=decision,
            changes=changes or [],
        )
        self.events.append(event)
        return event

    def _actor_id_for_name(self, actor_name: str) -> str:
        for actor in self.actors.values():
            if actor.name == actor_name:
                return actor.actor_id
        return ""

    def _target_ids_from_graph_transition(self, report: dict) -> list[str]:
        ids: list[str] = []
        for node in report.get("created_temporary_nodes", []) or []:
            if isinstance(node, dict) and node.get("node_id"):
                ids.append(str(node["node_id"]))
        for item in report.get("updated_temporary_nodes", []) or []:
            if isinstance(item, dict) and item.get("node_id"):
                ids.append(str(item["node_id"]))
        for edge_key in ["removed_fact_edges", "added_fact_edges"]:
            for edge in report.get(edge_key, []) or []:
                if not isinstance(edge, dict):
                    continue
                for key in ["from_node_id", "to_node_id"]:
                    if edge.get(key):
                        ids.append(str(edge[key]))
        updated_agent = report.get("updated_agent_node")
        if isinstance(updated_agent, dict) and updated_agent.get("node_id"):
            ids.append(str(updated_agent["node_id"]))
        return list(dict.fromkeys(ids))

    def _changes_from_mutation_result(
        self,
        result: EnvironmentMutationResult | None,
    ) -> list[WorldEventChange]:
        if result is None:
            return []

        changes: list[WorldEventChange] = []
        for item in result.applied_changes:
            changes.extend(self._changes_from_applied_mutation(item))
        return changes

    def _changes_from_applied_mutation(self, item: AppliedElementMutation) -> list[WorldEventChange]:
        changes: list[WorldEventChange] = []
        if item.old_physical_status != item.new_physical_status:
            changes.append(
                WorldEventChange(
                    element_id=item.element_id,
                    field="physical_status",
                    old=item.old_physical_status,
                    new=item.new_physical_status,
                )
            )
        if item.old_evolution_status != item.new_evolution_status:
            changes.append(
                WorldEventChange(
                    element_id=item.element_id,
                    field="evolution_status",
                    old=item.old_evolution_status,
                    new=item.new_evolution_status,
                )
            )
        if item.old_interaction_status != item.new_interaction_status:
            changes.append(
                WorldEventChange(
                    element_id=item.element_id,
                    field="interaction_status",
                    old=item.old_interaction_status,
                    new=item.new_interaction_status,
                )
            )
        detail_keys = set(item.old_state_details) | set(item.new_state_details)
        for key in sorted(detail_keys):
            old = item.old_state_details.get(key, "")
            new = item.new_state_details.get(key, "")
            if old != new:
                changes.append(
                    WorldEventChange(
                        element_id=item.element_id,
                        field=f"state_details.{key}",
                        old=old or "unset",
                        new=new or "unset",
                    )
                )
        return changes

from __future__ import annotations

from dataclasses import dataclass, field

from contextual_world.structure.scene_schema import Home


@dataclass
class ElementMutation:
    element_id: str
    physical_status: str | None = None
    evolution_status: str | None = None
    interaction_status: str | None = None
    state_details: dict[str, str] | None = None


@dataclass
class EnvironmentMutationRequest:
    summary: str
    feedback_narration: str = ""
    source: str = ""
    element_mutations: list[ElementMutation] = field(default_factory=list)


@dataclass
class AppliedElementMutation:
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
class EnvironmentMutationResult:
    success: bool
    summary: str
    feedback_narration: str = ""
    applied_changes: list[AppliedElementMutation] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


class EnvironmentMutationAPI:
    def __init__(self, home: Home) -> None:
        self.home = home

    def apply(self, request: EnvironmentMutationRequest) -> EnvironmentMutationResult:
        applied_changes: list[AppliedElementMutation] = []
        skipped: list[str] = []
        seen: set[str] = set()

        for mutation in request.element_mutations:
            if mutation.element_id in seen:
                skipped.append(f"duplicate element mutation skipped: {mutation.element_id}")
                continue
            seen.add(mutation.element_id)

            element = self.home.find_element(mutation.element_id)
            if element is None:
                skipped.append(f"element not found: {mutation.element_id}")
                continue

            old_physical_status = element.physical_status
            old_evolution_status = element.evolution_status
            old_interaction_status = element.interaction_status
            old_state_details = dict(element.state_details)

            if mutation.physical_status is not None:
                element.set_physical_status(mutation.physical_status)
            if mutation.evolution_status is not None:
                element.set_evolution_status(mutation.evolution_status)
            if mutation.interaction_status is not None:
                element.set_interaction_status(mutation.interaction_status)
            if mutation.state_details:
                element.update_state_details(mutation.state_details)

            if (
                old_physical_status == element.physical_status
                and old_evolution_status == element.evolution_status
                and old_interaction_status == element.interaction_status
                and old_state_details == element.state_details
            ):
                continue

            applied_changes.append(
                AppliedElementMutation(
                    area_id=self._find_area_id_for_element(element.node_id) or "",
                    element_id=element.node_id,
                    element_name=element.name,
                    old_physical_status=old_physical_status,
                    new_physical_status=element.physical_status,
                    old_evolution_status=old_evolution_status,
                    new_evolution_status=element.evolution_status,
                    old_interaction_status=old_interaction_status,
                    new_interaction_status=element.interaction_status,
                    old_state_details=old_state_details,
                    new_state_details=dict(element.state_details),
                )
            )

        return EnvironmentMutationResult(
            success=bool(applied_changes),
            summary=request.summary,
            feedback_narration=request.feedback_narration,
            applied_changes=applied_changes,
            skipped=skipped,
        )

    def _find_area_id_for_element(self, element_id: str) -> str | None:
        for area in self.home.areas:
            if area.find_element(element_id) is not None:
                return area.node_id
        return None

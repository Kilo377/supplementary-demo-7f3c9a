from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import hypot

from llm.api_manager import APIManager
from contextual_world.world_old.prompts.commonsense_source_prompt import build_commonsense_source_prompt
from contextual_world.structure.scene_schema import Element, Home

ACTIVE_STATUS_KEYWORDS = (
    "fire",
    "burn",
    "smoke",
    "smolder",
    "spark",
    "leak",
    "flood",
    "melt",
    "overheat",
)


def infer_default_evolution_status(physical_status: str) -> str:
    lowered = physical_status.strip().lower()
    if lowered == "regular":
        return "stable"
    if any(keyword in lowered for keyword in ACTIVE_STATUS_KEYWORDS):
        return "changing"
    return "stable"


@dataclass
class CommonsenseObservation:
    summary: str
    candidate_effects: list[str] = field(default_factory=list)


@dataclass
class CommonsenseSource:
    area_id: str
    area_name: str
    element_id: str
    element_name: str
    physical_status: str
    evolution_status: str


@dataclass
class RecentIntervention:
    actor_name: str
    target_element_id: str
    description: str


@dataclass
class CommonsenseUpdate:
    element_id: str
    element_name: str
    area_id: str
    from_physical_status: str
    to_physical_status: str
    from_evolution_status: str
    to_evolution_status: str
    reason: str = ""


@dataclass
class SourceEvolutionResult:
    source_element_id: str
    source_element_name: str
    summary: str
    updates: list[CommonsenseUpdate] = field(default_factory=list)
    raw_response: str = ""
    skipped: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class CommonsenseTickResult:
    tick: int
    non_regular_sources: list[CommonsenseSource] = field(default_factory=list)
    source_results: list[SourceEvolutionResult] = field(default_factory=list)
    applied_updates: list[CommonsenseUpdate] = field(default_factory=list)
    summary: str = ""


class CommonsenseEngine:
    """LLM-backed per-tick commonsense evolution for changing physical states.

    TODO: migrate this mechanism to WorldGraph. The graph version should scan
    permanent_element / temporary_element / human_agent nodes and fact edges,
    then apply node and relation updates through WorldGraphTransitionApplier
    instead of mutating Home elements directly.
    """

    def __init__(self, home: Home, *, provider_name: str = "ollama", model: str | None = None) -> None:
        self.home = home
        self.provider_name = provider_name
        self.model = model
        self.current_tick = 0
        self.api = APIManager(provider_name=provider_name)

    def describe_world_risk(self) -> CommonsenseObservation:
        return CommonsenseObservation(
            summary="The common-sense rules directory is in place; subsequent experience-based state changes inferred by the LLM can be handled here.",
            candidate_effects=[],
        )

    def list_non_regular_sources(self) -> list[CommonsenseSource]:
        sources: list[CommonsenseSource] = []
        for area in self.home.areas:
            for element in area.elements:
                if element.physical_status == "regular":
                    continue
                if element.evolution_status != "changing":
                    continue
                sources.append(
                    CommonsenseSource(
                        area_id=area.node_id,
                        area_name=area.name,
                        element_id=element.node_id,
                        element_name=element.name,
                        physical_status=element.physical_status,
                        evolution_status=element.evolution_status,
                    )
                )
        return sources

    def build_source_prompt(
        self,
        *,
        tick: int,
        source: CommonsenseSource,
        local_elements: list[Element],
        recent_intervention: RecentIntervention | None = None,
    ) -> str:
        return build_commonsense_source_prompt(
            tick=tick,
            source=source.__dict__,
            local_elements=[
                {
                    "element_id": element.node_id,
                    "element_name": element.name,
                    "physical_status": element.physical_status,
                    "evolution_status": element.evolution_status,
                }
                for element in local_elements
            ],
            recent_intervention=recent_intervention.__dict__ if recent_intervention is not None else None,
        )

    def advance_tick(
        self,
        *,
        recent_interventions: list[RecentIntervention] | None = None,
    ) -> CommonsenseTickResult:
        self.current_tick += 1
        initial_sources = self.list_non_regular_sources()
        if not initial_sources:
            return CommonsenseTickResult(
                tick=self.current_tick,
                non_regular_sources=[],
                source_results=[],
                applied_updates=[],
                summary=(
                    f"Tick {self.current_tick}: No abnormal sources are currently changing in the world,"
                    "so the common-sense evolution does not trigger this round."
                ),
            )

        source_results: list[SourceEvolutionResult] = []
        applied_updates: list[CommonsenseUpdate] = []
        updated_element_ids: set[str] = set()
        intervention_map = {
            item.target_element_id: item
            for item in (recent_interventions or [])
        }

        for initial_source in initial_sources:
            live_source = self._refresh_source(initial_source.element_id)
            if live_source is None or live_source.physical_status == "regular":
                continue
            if live_source.evolution_status != "changing":
                continue

            result = self._evolve_single_source(
                live_source,
                recent_intervention=intervention_map.get(live_source.element_id),
                updated_element_ids=updated_element_ids,
            )
            source_results.append(result)
            applied_updates.extend(result.updates)
            updated_element_ids.update(update.element_id for update in result.updates)

        final_sources = self.list_non_regular_sources()
        if applied_updates:
            summaries = "；".join(
                result.summary for result in source_results if result.summary
            )
            summary = f"Tick {self.current_tick}: Completed commonsense evolution for {len(source_results)} anomaly sources. {summaries}"
        else:
            names = "、".join(source.element_name for source in final_sources[:4])
            summary = (
                f"Tick {self.current_tick}: Detected {len(final_sources)} anomaly sources, "
                f"no new physical_status / evolution_status updates this round. Current includes: {names}."
            )

        return CommonsenseTickResult(
            tick=self.current_tick,
            non_regular_sources=final_sources,
            source_results=source_results,
            applied_updates=applied_updates,
            summary=summary,
        )

    def reset(self) -> None:
        self.current_tick = 0

    def _evolve_single_source(
        self,
        source: CommonsenseSource,
        *,
        recent_intervention: RecentIntervention | None = None,
        updated_element_ids: set[str] | None = None,
    ) -> SourceEvolutionResult:
        local_elements = self._collect_local_elements(source.element_id, source.area_id)
        prompt = self.build_source_prompt(
            tick=self.current_tick,
            source=source,
            local_elements=local_elements,
            recent_intervention=recent_intervention,
        )
        try:
            raw = self.api.generate(prompt, model=self.model)
            parsed = json.loads(self._extract_json_text(raw))
        except Exception as error:
            return SourceEvolutionResult(
                source_element_id=source.element_id,
                source_element_name=source.element_name,
                summary=f"Commonsense evolution for {source.element_name} has temporarily failed to complete.",
                raw_response="" if "raw" not in locals() else raw,
                error=str(error),
            )

        updates = self._apply_updates(
            parsed.get("updates", []),
            updated_element_ids=updated_element_ids or set(),
        )
        return SourceEvolutionResult(
            source_element_id=source.element_id,
            source_element_name=source.element_name,
            summary=parsed.get("summary", "") or f"{source.element_name} shows no significant further changes this round.",
            updates=updates,
            raw_response=raw,
        )

    def _apply_updates(
        self,
        raw_updates: list[dict],
        *,
        updated_element_ids: set[str],
    ) -> list[CommonsenseUpdate]:
        applied: list[CommonsenseUpdate] = []
        for item in raw_updates:
            element_id = item.get("element_id", "") or ""
            new_status = item.get("to_physical_status", "") or ""
            new_evolution_status = item.get("to_evolution_status", "") or ""
            reason = item.get("reason", "") or ""
            if not element_id or not new_status:
                continue
            if element_id in updated_element_ids:
                continue

            element = self.home.find_element(element_id)
            if element is None:
                continue

            if not new_evolution_status:
                new_evolution_status = infer_default_evolution_status(new_status)
            else:
                new_evolution_status = infer_default_evolution_status(new_status)

            area_id = self._find_area_id_for_element(element.node_id) or ""
            old_status = element.physical_status
            old_evolution_status = element.evolution_status
            if old_status == new_status and old_evolution_status == new_evolution_status:
                continue
            element.set_status(new_status, evolution_status=new_evolution_status)
            updated_element_ids.add(element.node_id)
            applied.append(
                CommonsenseUpdate(
                    element_id=element.node_id,
                    element_name=element.name,
                    area_id=area_id,
                    from_physical_status=old_status,
                    to_physical_status=element.physical_status,
                    from_evolution_status=old_evolution_status,
                    to_evolution_status=element.evolution_status,
                    reason=reason,
                )
            )
        return applied

    def _collect_local_elements(
        self,
        source_element_id: str,
        area_id: str,
        *,
        max_items: int = 4,
    ) -> list[Element]:
        area = self.home.find_area(area_id)
        source = self.home.find_element(source_element_id)
        if area is None or source is None:
            return []

        ranked: list[tuple[float, Element]] = []
        for element in area.elements:
            if element.node_id == source_element_id:
                continue
            distance = hypot(
                element.center[0] - source.center[0],
                element.center[1] - source.center[1],
            )
            ranked.append((distance, element))
        ranked.sort(key=lambda item: item[0])
        return [element for _, element in ranked[:max_items]]

    def _refresh_source(self, element_id: str) -> CommonsenseSource | None:
        for area in self.home.areas:
            element = area.find_element(element_id)
            if element is None:
                continue
            return CommonsenseSource(
                area_id=area.node_id,
                area_name=area.name,
                element_id=element.node_id,
                element_name=element.name,
                physical_status=element.physical_status,
                evolution_status=element.evolution_status,
            )
        return None

    def _find_area_id_for_element(self, element_id: str) -> str | None:
        for area in self.home.areas:
            if area.find_element(element_id) is not None:
                return area.node_id
        return None

    def _extract_json_text(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        if "```json" in stripped:
            after = stripped.split("```json", 1)[1]
            return after.split("```", 1)[0].strip()
        if "```" in stripped:
            after = stripped.split("```", 1)[1]
            return after.split("```", 1)[0].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return stripped[start : end + 1]
        raise ValueError("No JSON object found in commonsense response.")

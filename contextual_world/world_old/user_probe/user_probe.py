from __future__ import annotations

import json
from dataclasses import dataclass, field

from llm.api_manager import APIManager
from contextual_world.world_old.environment_mutation_api import (
    ElementMutation,
    EnvironmentMutationAPI,
    EnvironmentMutationRequest,
)
from contextual_world.world_old.prompts.user_probe_prompt import build_user_probe_prompt
from contextual_world.structure.scene_schema import Home


@dataclass
class ProbeOperation:
    scope: str
    target_area_id: str | None = None
    target_element_ids: list[str] = field(default_factory=list)
    target_element_names: list[str] = field(default_factory=list)
    physical_status: str | None = None
    evolution_status: str | None = None
    interaction_status: str | None = None
    reason: str = ""


@dataclass
class ProbePlan:
    user_command: str
    summary: str
    operations: list[ProbeOperation] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class AppliedChange:
    element_id: str
    element_name: str
    old_status: str
    new_status: str
    area_id: str


@dataclass
class ProbeApplyResult:
    success: bool
    summary: str
    changes: list[AppliedChange] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    raw_response: str = ""
    error: str = ""


class UserProbe:
    """User-facing high-privilege environment entrypoint."""

    def __init__(self, home: Home, *, provider_name: str = "ollama", model: str | None = None) -> None:
        self.home = home
        self.provider_name = provider_name
        self.model = model
        self.api = APIManager(provider_name=provider_name)
        self.mutation_api = EnvironmentMutationAPI(home)

    def serialize_world_state(self) -> dict:
        areas = []
        for area in self.home.areas:
            elements = []
            for element in area.elements:
                elements.append(
                    {
                        "id": element.node_id,
                        "name": element.name,
                        "physical_status": element.physical_status,
                        "evolution_status": element.evolution_status,
                        "interaction_status": element.interaction_status,
                        "state_details": dict(element.state_details),
                        "movable": element.movable,
                    }
                )
            areas.append(
                {
                    "area_id": area.node_id,
                    "area_name": area.name,
                    "elements": elements,
                }
            )
        return {
            "home_id": self.home.node_id,
            "areas": areas,
        }

    def world_state_text(self) -> str:
        lines = []
        for area in self.home.areas:
            lines.append(f"[{area.node_id}] {area.name}")
            for element in area.elements:
                lines.append(
                    f"- id={element.node_id}, name={element.name}, "
                    f"physical_status={element.physical_status}, "
                    f"evolution_status={element.evolution_status}, "
                    f"interaction_status={element.interaction_status}, "
                    f"state_details={element.state_details}, "
                    f"movable={element.movable}"
                )
        return "\n".join(lines)

    def build_probe_prompt(self, user_command: str, *, actor_name: str, focus_element_name: str = "") -> str:
        return build_user_probe_prompt(
            user_command,
            actor_name=actor_name,
            world_state=self.serialize_world_state(),
            focus_element_name=focus_element_name,
        )

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
        raise ValueError("No JSON object found in LLM response.")

    def generate_plan(self, user_command: str, *, actor_name: str, focus_element_name: str = "") -> ProbePlan:
        prompt = self.build_probe_prompt(
            user_command,
            actor_name=actor_name,
            focus_element_name=focus_element_name,
        )
        raw = self.api.generate(prompt, model=self.model)
        parsed = json.loads(self._extract_json_text(raw))
        operations = []
        for item in parsed.get("operations", []):
            operations.append(
                ProbeOperation(
                    scope=item.get("scope", "element"),
                    target_area_id=item.get("target_area_id"),
                    target_element_ids=item.get("target_element_ids", []) or [],
                    target_element_names=item.get("target_element_names", []) or [],
                    physical_status=item.get("physical_status", item.get("status")),
                    evolution_status=item.get("evolution_status"),
                    interaction_status=item.get("interaction_status"),
                    reason=item.get("reason", ""),
                )
            )
        return ProbePlan(
            user_command=user_command,
            summary=parsed.get("summary", ""),
            operations=operations,
            raw_response=raw,
        )

    def build_mutation_request(self, plan: ProbePlan) -> tuple[EnvironmentMutationRequest, list[str]]:
        element_mutations: list[ElementMutation] = []
        skipped: list[str] = []
        seen: set[str] = set()

        for operation in plan.operations:
            if operation.scope == "area":
                if not operation.target_area_id:
                    skipped.append("area operation missing target_area_id")
                    continue
                area = self.home.find_area(operation.target_area_id)
                if area is None:
                    skipped.append(f"area not found: {operation.target_area_id}")
                    continue
                for element in area.elements:
                    if element.node_id in seen:
                        continue
                    seen.add(element.node_id)
                    element_mutations.append(
                        ElementMutation(
                            element_id=element.node_id,
                            physical_status=operation.physical_status,
                            evolution_status=operation.evolution_status,
                            interaction_status=operation.interaction_status,
                        )
                    )
                continue

            matched_any = False
            for element_id in operation.target_element_ids:
                element = self.home.find_element(element_id)
                if element is None:
                    skipped.append(f"element id not found: {element_id}")
                    continue
                if element.node_id in seen:
                    continue
                matched_any = True
                seen.add(element.node_id)
                element_mutations.append(
                    ElementMutation(
                        element_id=element.node_id,
                        physical_status=operation.physical_status,
                        evolution_status=operation.evolution_status,
                        interaction_status=operation.interaction_status,
                    )
                )

            for element_name in operation.target_element_names:
                element = self.home.find_element_by_name(element_name)
                if element is None:
                    skipped.append(f"element name not found: {element_name}")
                    continue
                if element.node_id in seen:
                    continue
                matched_any = True
                seen.add(element.node_id)
                element_mutations.append(
                    ElementMutation(
                        element_id=element.node_id,
                        physical_status=operation.physical_status,
                        evolution_status=operation.evolution_status,
                        interaction_status=operation.interaction_status,
                    )
                )

            if not matched_any:
                skipped.append(
                    "no targets matched for operation: "
                    f"{operation.reason or operation.physical_status or operation.interaction_status or 'unknown'}"
                )

        return (
            EnvironmentMutationRequest(
                summary=plan.summary,
                feedback_narration=plan.summary,
                source="user_probe",
                element_mutations=element_mutations,
            ),
            skipped,
        )

    def apply_plan(self, plan: ProbePlan) -> ProbeApplyResult:
        request, pre_skipped = self.build_mutation_request(plan)
        result = self.mutation_api.apply(request)
        changes = [
            AppliedChange(
                element_id=item.element_id,
                element_name=item.element_name,
                old_status=item.old_physical_status,
                new_status=item.new_physical_status,
                area_id=item.area_id,
            )
            for item in result.applied_changes
            if item.old_physical_status != item.new_physical_status
        ]
        return ProbeApplyResult(
            success=result.success,
            summary=result.summary,
            changes=changes,
            skipped=[*pre_skipped, *result.skipped],
            raw_response=plan.raw_response,
        )

    def probe_and_apply(
        self,
        user_command: str,
        *,
        actor_name: str,
        focus_element_name: str = "",
    ) -> ProbeApplyResult:
        try:
            plan = self.generate_plan(
                user_command,
                actor_name=actor_name,
                focus_element_name=focus_element_name,
            )
        except Exception as error:
            return ProbeApplyResult(
                success=False,
                summary="LLM 世界探针暂时未能生成修改方案。",
                skipped=[
                    "当前没有拿到可执行的结构化操作。",
                    "你可以先检查本地模型服务是否启动，再重试同一条自然语言命令。",
                ],
                error=str(error),
            )
        return self.apply_plan(plan)

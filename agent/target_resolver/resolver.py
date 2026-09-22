from __future__ import annotations

import json
from time import perf_counter, sleep

from agent.belief.spatial_memory.spatial_belief import SpatialBelief
from agent.target_resolver.prompt import build_target_resolution_prompt
from agent.target_resolver.types import TargetResolution, TargetResolutionError
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from llm.api_manager import APIManager


def resolve_target_with_llm(
    engine: PhysicsEngine,
    *,
    agent_name: str,
    action_text: str,
    current_area_id: str,
    current_area_name: str,
    provider_name: str = "ollama",
    model: str | None = None,
    spatial_belief: SpatialBelief | None = None,
    world_graph_context: dict | None = None,
    max_retries: int = 1,
) -> TargetResolution:
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.target_resolver",
    )
    known_world = _target_resolution_state(
        engine,
        spatial_belief=spatial_belief,
        current_area_id=current_area_id,
        action_text=action_text,
    )
    prompt = build_target_resolution_prompt(
        agent_name=agent_name,
        action_text=action_text,
        current_area_id=current_area_id,
        current_area_name=current_area_name,
        scene_elements_text=_scene_elements_text(engine),
        known_world=known_world,
        world_graph_context=world_graph_context,
    )
    raw = ""
    started = perf_counter()
    last_error: Exception | None = None
    for attempt in range(max(0, max_retries) + 1):
        try:
            raw = api.generate(prompt, model=model)
            last_error = None
            break
        except Exception as error:
            last_error = error
            if attempt >= max(0, max_retries):
                break
            sleep(0.5)
    if last_error is not None:
        raise TargetResolutionError(
            f"Target resolver LLM call failed: {last_error}",
            raw_response=raw,
            prompt=prompt,
            duration_seconds=round(perf_counter() - started, 6),
        ) from last_error

    try:
        parsed = json.loads(_extract_json_text(raw))
    except Exception as error:
        raise TargetResolutionError(
            "Failed to parse target resolver LLM output.",
            raw_response=raw,
            prompt=prompt,
            duration_seconds=round(perf_counter() - started, 6),
        ) from error

    result = TargetResolution(
        target_element_id=str(
            parsed.get("operation_target_element_id", parsed.get("target_element_id", "")) or ""
        ).strip(),
        target_element_name=str(
            parsed.get("operation_target_element_name", parsed.get("target_element_name", "")) or ""
        ).strip(),
        navigation_anchor_element_id=str(
            parsed.get("navigation_anchor_element_id", "") or ""
        ).strip(),
        navigation_anchor_element_name=str(
            parsed.get("navigation_anchor_element_name", "") or ""
        ).strip(),
        secondary_target_element_id=str(parsed.get("secondary_target_element_id", "") or "").strip(),
        arrival_completes_action=_parse_bool(parsed.get("arrival_completes_action")),
        reason=str(parsed.get("reason", "") or "").strip(),
        raw_response=raw,
        prompt=prompt,
        provider_name=api.provider_name,
        model=api.route.model or "provider default",
        duration_seconds=round(perf_counter() - started, 6),
    )
    return _validate_target_resolution(engine, result, current_area_id=current_area_id)


def resolve_target_fallback(
    engine: PhysicsEngine,
    *,
    action_text: str,
    current_area_id: str,
    spatial_belief: SpatialBelief | None = None,
) -> TargetResolution:
    if spatial_belief is not None:
        matches = spatial_belief.elements_mentioned_in(action_text, current_area_id=current_area_id)
        if matches:
            element = matches[0]
            return TargetResolution(
                target_element_id=element.element_id,
                target_element_name=element.name,
                navigation_anchor_element_id=element.element_id,
                navigation_anchor_element_name=element.name,
                reason="动作文本直接提到了这个场景元素。",
            )

    text = str(action_text or "")
    current_area = engine.get_area(current_area_id)
    areas = [current_area] if current_area is not None else []
    areas.extend(area for area in engine.home.areas if area is not current_area)
    candidates = []
    for area in areas:
        if area is None:
            continue
        for element in area.elements:
            name_index = text.rfind(element.name)
            id_index = text.rfind(element.node_id)
            index = max(name_index, id_index)
            if index < 0:
                continue
            candidates.append((area.node_id == current_area_id, len(element.name), index, element))
    if not candidates:
        return TargetResolution(reason="没有在动作文本中找到明确的场景元素。")
    candidates.sort(reverse=True, key=lambda item: item[:3])
    element = candidates[0][3]
    return TargetResolution(
        target_element_id=element.node_id,
        target_element_name=element.name,
        navigation_anchor_element_id=element.node_id,
        navigation_anchor_element_name=element.name,
        reason="动作文本直接提到了这个场景元素。",
    )


def _target_resolution_state(
    engine: PhysicsEngine,
    *,
    spatial_belief: SpatialBelief | None,
    current_area_id: str,
    action_text: str,
) -> dict:
    if spatial_belief is not None:
        return spatial_belief.to_target_resolution_state(
            current_area_id=current_area_id,
            thought_text=action_text,
        )

    return {
        "source": "world_scene",
        "areas": [
            {
                "area_id": area.node_id,
                "area_name": area.name,
                "elements": [
                    {
                        "id": element.node_id,
                        "name": element.name,
                        "physical_status": element.physical_status,
                        "evolution_status": element.evolution_status,
                        "interaction_status": element.interaction_status,
                        "state_details": dict(element.state_details),
                    }
                    for element in area.elements
                ],
            }
            for area in engine.home.areas
        ],
    }


def _scene_elements_text(engine: PhysicsEngine) -> str:
    items = []
    for area in engine.home.areas:
        for element in area.elements:
            items.append(f"{element.name}({element.node_id})")
    return "、".join(items) if items else "无"


def _validate_target_resolution(
    engine: PhysicsEngine,
    result: TargetResolution,
    *,
    current_area_id: str,
) -> TargetResolution:
    target_id = _resolve_element_id(
        engine,
        element_id=result.target_element_id,
        element_name=result.target_element_name,
        current_area_id=current_area_id,
    )
    secondary_id = _resolve_element_id(
        engine,
        element_id=result.secondary_target_element_id,
        element_name="",
        current_area_id=current_area_id,
    )
    navigation_anchor_id = _resolve_element_id(
        engine,
        element_id=result.navigation_anchor_element_id,
        element_name=result.navigation_anchor_element_name,
        current_area_id=current_area_id,
    )
    if not target_id:
        navigation_element = engine.get_element(navigation_anchor_id)
        return TargetResolution(
            navigation_anchor_element_id=navigation_anchor_id,
            navigation_anchor_element_name=(
                navigation_element.name
                if navigation_element is not None
                else result.navigation_anchor_element_name
            ),
            secondary_target_element_id=secondary_id,
            arrival_completes_action=result.arrival_completes_action,
            reason=result.reason,
            raw_response=result.raw_response,
            prompt=result.prompt,
            provider_name=result.provider_name,
            model=result.model,
            duration_seconds=result.duration_seconds,
            error=result.error,
        )
    element = engine.get_element(target_id)
    return TargetResolution(
        target_element_id=target_id,
        target_element_name=element.name if element is not None else result.target_element_name,
        navigation_anchor_element_id=navigation_anchor_id or target_id,
        navigation_anchor_element_name=(
            engine.get_element(navigation_anchor_id or target_id).name
            if engine.get_element(navigation_anchor_id or target_id) is not None
            else result.navigation_anchor_element_name
        ),
        secondary_target_element_id=secondary_id,
        arrival_completes_action=result.arrival_completes_action,
        reason=result.reason,
        raw_response=result.raw_response,
        prompt=result.prompt,
        provider_name=result.provider_name,
        model=result.model,
        duration_seconds=result.duration_seconds,
        error=result.error,
    )


def _resolve_element_id(
    engine: PhysicsEngine,
    *,
    element_id: str,
    element_name: str,
    current_area_id: str,
) -> str:
    cleaned_id = str(element_id or "").strip()
    if cleaned_id and engine.get_element(cleaned_id) is not None:
        return cleaned_id

    cleaned_name = str(element_name or "").strip()
    if not cleaned_name:
        return ""

    current_area = engine.get_area(current_area_id)
    if current_area is not None:
        for element in current_area.elements:
            if _name_matches(cleaned_name, element.name) or cleaned_name == element.node_id:
                return element.node_id

    for area in engine.home.areas:
        for element in area.elements:
            if _name_matches(cleaned_name, element.name) or cleaned_name == element.node_id:
                return element.node_id
    return ""


def _name_matches(value: str, element_name: str) -> bool:
    return value == element_name or value in element_name or element_name in value


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _extract_json_text(text: str) -> str:
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
    raise ValueError("No JSON object found in target resolver response.")

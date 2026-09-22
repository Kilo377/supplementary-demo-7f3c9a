from __future__ import annotations

from agent.desire import DesireState
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine

from .attention import build_attention
from .self_input import build_self_input
from .types import PerceiveResult
from .world_input import build_environment_input, build_visual_input, detect_changed_elements, initialize_spatial_belief


def perceive_agent(
    agent,
    engine: PhysicsEngine,
    *,
    desire_state: DesireState | None = None,
) -> PerceiveResult:
    if getattr(agent, "belief", None) is None:
        agent.belief = initialize_spatial_belief(engine)

    agent.perception_step += 1
    current_area = engine.get_area_for_point(*agent.center)
    if current_area is None:
        raise ValueError("Agent is outside known areas and cannot perceive.")

    first_time_visit = current_area.node_id not in agent.visited_area_ids
    visual_input = build_visual_input(agent, engine, current_area)
    visible_element_ids = {item.element_id for item in visual_input.visible_elements}
    changed_elements = detect_changed_elements(agent, current_area, visible_element_ids)

    result = PerceiveResult(
        agent_id=agent.node_id,
        agent_name=agent.name,
        step_id=agent.perception_step,
        first_time_visit=first_time_visit,
        environment_input=build_environment_input(
            agent,
            engine,
            current_area,
            changed_elements=changed_elements,
        ),
        self_input=build_self_input(agent, desire_state or getattr(agent, "desire_state", None)),
        visual_input=visual_input,
    )
    result.attention = build_attention(result, belief=agent.belief)

    agent.belief.refresh_visible_elements_from_world(
        current_area,
        visible_element_ids=visible_element_ids,
        seen_step=agent.perception_step,
    )
    agent.visited_area_ids.add(current_area.node_id)
    agent.current_area_id = current_area.node_id
    return result


def perceive_narrate_output(
    agent,
    engine: PhysicsEngine,
    *,
    use_llm: bool = False,
    provider_name: str = "ollama",
    model: str | None = None,
) -> str:
    result = perceive_agent(agent, engine)
    if not use_llm:
        return result.narration_text

    try:
        from agent.narration_llm import generate_narration_with_llm

        llm_text = generate_narration_with_llm(
            result,
            provider_name=provider_name,
            model=model,
        )
        if llm_text:
            return llm_text
    except Exception:
        pass

    return result.narration_text

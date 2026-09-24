from __future__ import annotations

from agent.belief.short_time_memory import MemoryEpisode, ShortTermMemory
from agent.belief.spatial_memory.spatial_belief import ElementBeliefSnapshot, SpatialBelief
from agent.intent.state import IntentState
from agent.intuition import IntuitionResult
from agent.think.types import ThinkResult


def build_action_proposal_prompt(
    *,
    agent_name: str,
    intent: IntentState,
    short_time_memory: ShortTermMemory | None,
    intuition: IntuitionResult | None,
    spatial_belief: SpatialBelief | None,
    current_area_id: str,
    scene_name: str,
    pre_think_intuition: IntuitionResult | None = None,
    think_result: ThinkResult | None = None,
) -> str:
    intent_text = str(getattr(intent, "intent_text", "") or "").strip()
    memory_text = format_short_time_memory_for_action_prompt(short_time_memory)
    spatial_text = format_spatial_belief_for_action_prompt(
        spatial_belief,
        current_area_id=current_area_id,
        scene_name=scene_name,
        agent_name=agent_name,
    )
    intuition_text = format_intuition_for_action_prompt(intuition)
    if should_include_think_context(pre_think_intuition, think_result):
        pre_think_text = format_intuition_for_action_prompt(pre_think_intuition)
        think_text = format_think_for_action_prompt(think_result)
        return f"""
{agent_name} wants:
{intent_text or "(No clear Intent.)"}

{agent_name} has already done:
{memory_text}

{spatial_text}

Previously, {agent_name}'s subconscious plan was:
{pre_think_text}

Then {agent_name} thought carefully:
{think_text}

Now {agent_name}'s Intuition plans to:
{intuition_text}

So what specific action should {agent_name} take to try to satisfy {agent_name}'s thoughts?

Requirements:
- Output only one specific action.
- The action must start with {agent_name}.
- This is the actual action to be executed by the world; do not explain the reasoning.
- Prioritize furniture or items that actually exist in the current room.
- Do not repeat actions that have already failed.
- Do not fabricate rooms, furniture, or tools that are not in the Spatial Belief.
- Do not output waiting, chatting, going to another room, or stopping to think; these are already handled by routing after Intuition and Think.
- If a tool is needed but currently unavailable, first perform cleaning actions that do not depend on tools.
""".strip()

    return f"""
{agent_name} wants:
{intent_text or "(No clear Intent.)"}

{agent_name} has already done:
{memory_text}

{spatial_text}

Now {agent_name}'s Intuition plans to:
{intuition_text}

So what specific action should {agent_name} take to try to satisfy {agent_name}'s thoughts?

Requirements:
- Output only one specific action.
- The action must start with {agent_name}.
- This is the actual action to be executed by the world; do not explain the reasoning.
- Prioritize furniture or items that actually exist in the current room.
- Do not repeat actions that have already failed.
- Do not fabricate rooms, furniture, or tools that are not in the Spatial Belief.
- Do not output waiting, chatting, going to another room, or stopping to think; these are already handled by routing after Intuition.
- If a tool is needed but currently unavailable, first perform cleaning actions that do not depend on tools.
""".strip()


def format_short_time_memory_for_action_prompt(memory: ShortTermMemory | None) -> str:
    episodes = [
        episode
        for episode in list(getattr(memory, "episodes", []) or [])
        if getattr(episode, "recallable", True)
    ]
    if not episodes:
        return "None yet."
    return "\n".join(_format_episode(episode) for episode in episodes)


def format_intuition_for_action_prompt(intuition: IntuitionResult | None) -> str:
    if intuition is None:
        return "(No clear Intuition; can only choose the next step based on Intent, short-term memory, and Spatial Belief.)"
    route = f"[{intuition.route}] " if intuition.route else ""
    return f"{route}{intuition.thought}".strip() or "(No clear Intuition.)"


def format_think_for_action_prompt(think_result: ThinkResult | None) -> str:
    if think_result is None:
        return "(No explicit Think performed.)"
    parts = [
        text
        for text in [
            think_result.thought.strip(),
            think_result.conclusion.strip(),
        ]
        if text
    ]
    return "\n".join(parts) or "(Think did not form clear content.)"


def should_include_think_context(
    pre_think_intuition: IntuitionResult | None,
    think_result: ThinkResult | None,
) -> bool:
    if think_result is None:
        return False
    if pre_think_intuition is None or pre_think_intuition.route != "think":
        return False
    return bool(think_result.thought.strip() or think_result.conclusion.strip())


def format_spatial_belief_for_action_prompt(
    spatial_belief: SpatialBelief | None,
    *,
    current_area_id: str,
    scene_name: str,
    agent_name: str,
) -> str:
    if spatial_belief is None:
        return f"{agent_name} is currently in {scene_name}, but Spatial Belief is temporarily unavailable."

    current_area = spatial_belief.get_area(current_area_id)
    current_area_name = current_area.area_name if current_area is not None else current_area_id or "Unknown area"
    lines = [f"{agent_name} is currently in {current_area_name} of {scene_name}."]
    if current_area is not None and current_area.elements:
        element_text = "、".join(
            _element_text(element)
            for element in current_area.elements.values()
            if element.name
        )
        lines.append(f"The furniture here includes: {element_text}.")
    else:
        lines.append("There is currently no available information about furniture or items here.")

    other_area_names = [
        area.area_name
        for area in spatial_belief.iter_areas()
        if area.area_id != current_area_id
    ]
    if other_area_names:
        lines.append(f"{agent_name} also knows that other rooms have: {'、'.join(other_area_names)}。")
    return "\n".join(lines)


def _format_episode(episode: MemoryEpisode) -> str:
    lines = [episode.header_text()]
    if episode.intended_action:
        lines.append(f"   intended: {episode.intended_action}")
    if episode.experienced_result:
        lines.append(f"   experienced: {episode.experienced_result}")
    if getattr(episode, "intuition_thought", ""):
        mode = f"{episode.intuition_mode}/" if getattr(episode, "intuition_mode", "") else ""
        route = f"[{mode}{episode.intuition_route}] " if getattr(episode, "intuition_route", "") or mode else ""
        lines.append(f"   intuition: {route}{episode.intuition_thought}")
    if getattr(episode, "arbiter_thought", ""):
        mode = f"[{episode.arbiter_mode}] " if getattr(episode, "arbiter_mode", "") else ""
        lines.append(f"   arbiter: {mode}{episode.arbiter_thought}")
    if episode.interacted_element_names:
        lines.append(f"   interacted: {'、'.join(episode.interacted_element_names)}")
    return "\n".join(lines)


def _element_text(element: ElementBeliefSnapshot) -> str:
    details = []
    if element.physical_status != "regular":
        details.append(f"Physical state={element.physical_status}")
    if element.evolution_status != "stable":
        details.append(f"Evolution status={element.evolution_status}")
    if element.interaction_status != "idle":
        details.append(f"Interaction status={element.interaction_status}")
    for key, value in sorted(element.state_details.items()):
        if str(value).strip():
            details.append(f"{key}={value}")
    if details:
        return f"{element.name}（{'，'.join(details)}）"
    return element.name

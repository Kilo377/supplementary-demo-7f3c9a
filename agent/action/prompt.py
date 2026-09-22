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
{agent_name} 想要：
{intent_text or "（没有明确 Intent。）"}

{agent_name} 因此已经做了：
{memory_text}

{spatial_text}

原来 {agent_name} 下意识的打算：
{pre_think_text}

然后 {agent_name} 仔细想了想：
{think_text}

现在 {agent_name} 的 Intuition 打算：
{intuition_text}

那么具体 {agent_name} 应该做什么，以努力满足 {agent_name} 的想法呢？

要求：
- 只输出一个具体动作。
- 动作用 {agent_name} 开头。
- 这是实际要交给 world 执行的动作，不要解释理由。
- 优先选择当前房间里真实存在的家具或物品。
- 不要重复已经失败的动作。
- 不要编造 Spatial Belief 里没有的房间、家具或工具。
- 不要输出等待、聊天、去另一个房间、停下来想；这些已经由 Intuition 和 Think 后路由处理。
- 如果需要工具但当前拿不到，先做不依赖工具的清理动作。
""".strip()

    return f"""
{agent_name} 想要：
{intent_text or "（没有明确 Intent。）"}

{agent_name} 因此已经做了：
{memory_text}

{spatial_text}

现在 {agent_name} 的 Intuition 打算：
{intuition_text}

那么具体 {agent_name} 应该做什么，以努力满足 {agent_name} 的想法呢？

要求：
- 只输出一个具体动作。
- 动作用 {agent_name} 开头。
- 这是实际要交给 world 执行的动作，不要解释理由。
- 优先选择当前房间里真实存在的家具或物品。
- 不要重复已经失败的动作。
- 不要编造 Spatial Belief 里没有的房间、家具或工具。
- 不要输出等待、聊天、去另一个房间、停下来想；这些已经由 Intuition 路由处理。
- 如果需要工具但当前拿不到，先做不依赖工具的清理动作。
""".strip()


def format_short_time_memory_for_action_prompt(memory: ShortTermMemory | None) -> str:
    episodes = [
        episode
        for episode in list(getattr(memory, "episodes", []) or [])
        if getattr(episode, "recallable", True)
    ]
    if not episodes:
        return "暂无。"
    return "\n".join(_format_episode(episode) for episode in episodes)


def format_intuition_for_action_prompt(intuition: IntuitionResult | None) -> str:
    if intuition is None:
        return "（没有明确 Intuition，只能根据 Intent、短期记忆和空间 Belief 选择下一步。）"
    route = f"[{intuition.route}] " if intuition.route else ""
    return f"{route}{intuition.thought}".strip() or "（没有明确 Intuition。）"


def format_think_for_action_prompt(think_result: ThinkResult | None) -> str:
    if think_result is None:
        return "（没有进行显式 Think。）"
    parts = [
        text
        for text in [
            think_result.thought.strip(),
            think_result.conclusion.strip(),
        ]
        if text
    ]
    return "\n".join(parts) or "（Think 没有形成明确内容。）"


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
        return f"{agent_name} 现在在 {scene_name}，但空间 Belief 暂时不可用。"

    current_area = spatial_belief.get_area(current_area_id)
    current_area_name = current_area.area_name if current_area is not None else current_area_id or "未知区域"
    lines = [f"{agent_name} 现在在 {scene_name} 的 {current_area_name}。"]
    if current_area is not None and current_area.elements:
        element_text = "、".join(
            _element_text(element)
            for element in current_area.elements.values()
            if element.name
        )
        lines.append(f"这里所有的家具有：{element_text}。")
    else:
        lines.append("这里暂时没有可用的家具或物品信息。")

    other_area_names = [
        area.area_name
        for area in spatial_belief.iter_areas()
        if area.area_id != current_area_id
    ]
    if other_area_names:
        lines.append(f"{agent_name} 还知道其他房间有：{'、'.join(other_area_names)}。")
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
        details.append(f"物理状态={element.physical_status}")
    if element.evolution_status != "stable":
        details.append(f"演化状态={element.evolution_status}")
    if element.interaction_status != "idle":
        details.append(f"交互状态={element.interaction_status}")
    for key, value in sorted(element.state_details.items()):
        if str(value).strip():
            details.append(f"{key}={value}")
    if details:
        return f"{element.name}（{'，'.join(details)}）"
    return element.name

from __future__ import annotations


def build_commonsense_source_prompt(
    *,
    tick: int,
    source: dict,
    local_elements: list[dict],
    recent_intervention: dict | None = None,
) -> str:
    local_lines = []
    for element in local_elements:
        local_lines.append(
            f"- {element.get('element_id', '')} / {element.get('element_name', '')}: "
            f"physical_status={element.get('physical_status', '')}, "
            f"evolution_status={element.get('evolution_status', '')}"
        )
    local_block = "\n".join(local_lines) if local_lines else "- 无额外局部元素"

    intervention_block = ""
    if recent_intervention is not None:
        intervention_block = f"""

最近一步 agent 干预：
- {recent_intervention.get('actor_name', '')} 对 {recent_intervention.get('target_element_id', '')} 做了这件事：{recent_intervention.get('description', '')}"""

    return f"""你是家庭世界中的常识物理演化器。

你的任务：
- 只围绕一个异常源元素，推演这一时间步最合理的 physical_status 变化
- 你可以让该异常源继续自我演化
- 你也可以让它影响附近原本 regular 的元素
- 只推进这一时间步，不要一口气推到最终结局

要求：
- 只修改 physical_status
- 不要修改 interaction_status
- 不要编造不存在的元素
- 输出必须是严格 JSON
- physical_status 使用简短、稳定、偏物理结果的英文状态词
- 如果这一时间步没有自然变化，可以返回空 updates
- 默认稍微保守：如果没有强烈常识证据表明它会继续恶化，就不要推进变化
- 扩散到附近元素时必须更保守，优先轻微影响，而且只影响非常接近的元素
- 同一时间步里，同一个元素最多更新一次

当前时间步：
{tick}

当前异常源：
- area_id: {source.get('area_id', '')}
- area_name: {source.get('area_name', '')}
- element_id: {source.get('element_id', '')}
- element_name: {source.get('element_name', '')}
- physical_status: {source.get('physical_status', '')}
- evolution_status: {source.get('evolution_status', '')}

局部相关元素：
{local_block}{intervention_block}

返回 JSON：
{{
  "summary": "这一时间步围绕该异常源发生了什么的简短中文摘要",
  "updates": [
    {{
      "element_id": "必须使用现有 element_id",
      "to_physical_status": "新的 physical_status",
      "to_evolution_status": "changing 或 stable",
      "reason": "为什么这一时间步会这样变化"
    }}
  ]
}}"""

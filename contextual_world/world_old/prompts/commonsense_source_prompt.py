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
    local_block = "\n".join(local_lines) if local_lines else "- No additional local elements"

    intervention_block = ""
    if recent_intervention is not None:
        intervention_block = f"""

The most recent agent intervention:
- {recent_intervention.get('actor_name', '')} did this to {recent_intervention.get('target_element_id', '')}: {recent_intervention.get('description', '')}"""

    return f"""You are the commonsense physics evolution engine in a household world.

Your task:
- Deduce the most reasonable physical_status change for this time step, centered on a single anomalous source element.
- You may allow the anomalous source to continue evolving on its own.
- You may also let it affect nearby elements that were previously regular.
- Advance only this one time step; do not jump straight to the final outcome.

Requirements:
- Only modify physical_status.
- Do not modify interaction_status.
- Do not fabricate non-existent elements.
- The output must be strictly JSON.
- Use short, stable, physics-result-oriented English status words for physical_status.
- If there is no natural change in this time step, you may return empty updates.
- Default to being slightly conservative: if there is no strong commonsense evidence that it will continue to worsen, do not advance the change.
- When spreading to nearby elements, be even more conservative: prioritize slight effects and only affect very close elements.
- In the same time step, each element can be updated at most once.

Current time step:
{tick}

Current anomalous source:
- area_id: {source.get('area_id', '')}
- area_name: {source.get('area_name', '')}
- element_id: {source.get('element_id', '')}
- element_name: {source.get('element_name', '')}
- physical_status: {source.get('physical_status', '')}
- evolution_status: {source.get('evolution_status', '')}

Locally relevant elements:
{local_block}{intervention_block}

Return JSON:
{{
  "summary": "A brief Chinese summary of what happened around this anomalous source in this time step",
  "updates": [
    {{
      "element_id": "Must use an existing element_id",
      "to_physical_status": "New physical_status",
      "to_evolution_status": "changing or stable",
      "reason": "Why this change occurs in this time step"
    }}
  ]
}}"""

from __future__ import annotations

import json
from dataclasses import dataclass

from agent.desire.desire_state import DesireState
from llm.api_manager import APIManager


@dataclass
class DesireUpdateResult:
    benefit: str
    cost: str
    desire_state: DesireState
    physiological_reason: str = ""
    internal_state_reason: str = ""
    mental_reason: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "benefit": self.benefit,
            "cost": self.cost,
            "desire_state": self.desire_state.to_dict(),
            "physiological_reason": self.physiological_reason,
            "internal_state_reason": self.internal_state_reason,
            "mental_reason": self.mental_reason,
        }


def build_desire_update_prompt(
    *,
    agent_name: str,
    desire_state: dict,
    intent_text: str,
    intent_status: str,
    intent_cycle_context: str,
    self_belief: str = "",
    personal_context: str = "",
    update_mode: str = "intent_end",
) -> str:
    personal_context_block = ""
    if personal_context.strip():
        personal_context_block = f"""
{agent_name} 的长期个人背景：
{personal_context.strip()}
"""
    is_step_update = update_mode == "every_step"
    update_timing_text = (
        "这次更新发生在刚完成的一步行动之后；intent 仍可能处于 active 状态。"
        if is_step_update
        else "这次更新发生在一个 intent 生命周期结束之后。"
    )
    context_label = "这一步行动及其最新结果" if is_step_update else "这个 intent 周期的上下文"
    self_belief_label = "这一步之后对自己状态的判断" if is_step_update else "在 intent 结束时对自己状态的判断"
    work_goal_rule = (
        "- 只有这一步的实际结果已经明确满足某个 work_goal 时，才能把它改成 true；"
        "不能因为 intent 还在进行、或仅仅朝目标移动，就提前完成目标"
        if is_step_update
        else "- 如果当 intent 周期所对应的行为和intent状态满足了某个 work_goal，把 completed 改成 true"
    )
    single_goal_rule = (
        "- 如果这一步只满足其中一个 work_goal，只把这个目标改成 true，其他目标保持 false"
        if is_step_update
        else "- 如果刚结束的 intent 只满足其中一个 work_goal，只把这个目标改成 true，其他目标保持 false"
    )
    return f"""你在更新一个 human agent 的 Desire state。

Desire 不是 intent。Desire 表示 {agent_name} 的需求、心情和正事目标是否被满足。
{update_timing_text}

当前 Desire state：
{json.dumps(desire_state, ensure_ascii=False, indent=2)}

刚结束的 intent：
{intent_text}

intent 生命周期状态：
{intent_status}

{context_label}：
{intent_cycle_context}

{agent_name} {self_belief_label}：
{self_belief or "无。"}
{personal_context_block}

更新规则：
1. physiological_state 是 0-10 的生理需求压力量表；分数越高，需求压力越强。
   - hunger: 0=完全不饿，10=非常饿
   - thirst: 0=完全不渴，10=非常渴
   - hygiene: 0=干净/不需要清洁，10=很脏/很需要清洁
   - 更新要保守：无关项尽量不变
   - 完成吃饭、喝水或清洁时，对应压力可以明显下降

2. internal_state 是 0-10 的可计算内部状态量表。
   - stress: 0=没有压力，10=压力极大
   - tension: 0=完全放松，10=非常紧张
   - fatigue: 0=完全不累，10=非常疲劳
   - depletion: 自控资源耗竭，不等于身体疲劳；3=正常基线，7=明显耗竭，10=极强耗竭
   - cognitive_load: 当前认知负荷，不等于紧张；3=正常基线，7=明显升高，10=极强负荷
   - stress 在内部状态门控中以3为正常基线、7为明显升高；不要仅因一次普通动作就大幅上调。
   - 持续抑制冲动、费力自控可能提高depletion；并行处理或复杂思考可能提高cognitive_load。没有相关证据时保持原值，休息或负担解除时可以降低。
   - 普通行动通常最多变化 1；强烈、持续或明确恢复性的经历才允许更明显变化
   - 身体劳动通常提高 fatigue，休息通常降低 fatigue
   - 失败和受阻可能提高 stress 或 tension，顺利完成事情可能降低它们

3. mental 是纯自然语言心理状态。
   - 输出一个新的自然语言 mental
   - 可以给简短 reason
   - 保持连续性，不要突然改成人格、长期价值观或新的目标
   - 可以自然体现压力、紧张和疲劳，但不要暴露量表数值

4. work_goal 是 Desire 里的正事目标，不是 intent。
   - 每个 work_goal 只有 text 和 completed
   - work_goal 可能同时有多个；它们彼此独立，必须逐项判断
   - 返回的 work_goal 数组必须保留当前 Desire state 里的所有原有目标，不要合并、拆分、改写或漏掉 text
   - 只能判断这个 desire 是否已经被满足
   - 不要给 work_goal 加 reason、status、active、blocked、abandoned
   - 如果一个 work_goal 已经 completed=true，保持 true
   {work_goal_rule}
   {single_goal_rule}
   - 如果所有 work_goal 都 completed=true，外层仿真可以结束

5. 不要发明新的 Desire 维度。
6. 不要把 intent lifecycle 状态直接复制成 work_goal 状态。
7. 只返回 JSON，不要解释推理过程。

返回 JSON 格式：
{{
  "benefit": "这个 intent 周期带来的收益，一句中文",
  "cost": "这个 intent 周期带来的代价，一句中文",
  "physiological_state_update": {{
    "after": {{
      "hunger": 0,
      "thirst": 0,
      "hygiene": 0
    }},
    "reason": "为什么这样更新生理状态"
  }},
  "internal_state_update": {{
    "after": {{
      "stress": 0,
      "tension": 0,
      "fatigue": 0,
      "depletion": 3,
      "cognitive_load": 3
    }},
    "reason": "为什么这样更新内部状态"
  }},
  "mental_update": {{
    "after": "新的自然语言心情状态",
    "reason": "为什么这样更新 mental"
  }},
  "work_goal": [
    {{
      "text": "原有 work goal text",
      "completed": false
    }}
  ]
}}
"""


def update_desire_state(
    *,
    agent_name: str,
    desire_state: DesireState,
    intent_text: str,
    intent_status: str,
    intent_cycle_context: str,
    self_belief: str = "",
    provider_name: str = "ollama",
    model: str | None = None,
    personal_context: str = "",
    update_mode: str = "intent_end",
) -> DesireUpdateResult:
    prompt = build_desire_update_prompt(
        agent_name=agent_name,
        desire_state=desire_state.to_dict(),
        intent_text=intent_text,
        intent_status=intent_status,
        intent_cycle_context=intent_cycle_context,
        self_belief=self_belief,
        personal_context=personal_context,
        update_mode=update_mode,
    )
    try:
        raw = APIManager(
            provider_name=provider_name,
            task_name="agent.desire_update",
        ).generate(prompt, model=model)
        parsed = json.loads(_extract_json_text(raw))
        updated = _desire_state_from_update(
            before=desire_state,
            parsed=parsed,
        )
        return DesireUpdateResult(
            benefit=parsed.get("benefit", "") or "",
            cost=parsed.get("cost", "") or "",
            desire_state=updated,
            physiological_reason=_update_reason(parsed, "physiological_state_update"),
            internal_state_reason=_update_reason(parsed, "internal_state_update"),
            mental_reason=_update_reason(parsed, "mental_update"),
            raw_response=raw,
        )
    except Exception as error:
        return DesireUpdateResult(
            benefit="",
            cost=f"Desire update failed: {error}",
            desire_state=desire_state,
            raw_response="",
        )


def _desire_state_from_update(*, before: DesireState, parsed: dict) -> DesireState:
    before_data = before.to_dict()
    physiological_after = (
        parsed.get("physiological_state_update", {}).get("after", {})
        if isinstance(parsed.get("physiological_state_update", {}), dict)
        else {}
    )
    internal_state_after = (
        parsed.get("internal_state_update", {}).get("after", {})
        if isinstance(parsed.get("internal_state_update", {}), dict)
        else {}
    )
    mental_after = (
        parsed.get("mental_update", {}).get("after", "")
        if isinstance(parsed.get("mental_update", {}), dict)
        else ""
    )
    work_goal_after = parsed.get("work_goal", before_data.get("work_goal", []))
    merged_work_goal = _merge_work_goal_updates(
        before_goals=before_data.get("work_goal", []),
        updated_goals=work_goal_after,
    )
    return DesireState.from_dict(
        {
            "physiological_state": _merge_numeric_state(
                before_data.get("physiological_state", {}),
                physiological_after,
            ),
            "internal_state": _merge_numeric_state(
                before_data.get("internal_state", {}),
                internal_state_after,
            ),
            "mental": mental_after or before_data.get("mental", ""),
            "work_goal": merged_work_goal,
        }
    )


def _update_reason(parsed: dict, key: str) -> str:
    value = parsed.get(key, {})
    if not isinstance(value, dict):
        return ""
    return str(value.get("reason", "") or "").strip()


def _merge_numeric_state(before: dict, after) -> dict:
    merged = dict(before or {})
    if isinstance(after, dict):
        merged.update(after)
    return merged


def _merge_work_goal_updates(*, before_goals: list, updated_goals) -> list[dict]:
    if not isinstance(updated_goals, list):
        updated_goals = []

    completed_by_text: dict[str, bool] = {}
    for item in updated_goals:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        completed_by_text[text] = bool(item.get("completed", False))

    merged = []
    for item in before_goals:
        if not isinstance(item, dict):
            text = str(item).strip()
            before_completed = False
        else:
            text = str(item.get("text", "")).strip()
            before_completed = bool(item.get("completed", False))
        if not text:
            continue
        merged.append(
            {
                "text": text,
                "completed": before_completed or completed_by_text.get(text, False),
            }
        )
    return merged


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
    raise ValueError("No JSON object found in desire update response.")

from __future__ import annotations

from .types import StateTransitionPromptContext


def build_state_transition_prompt(
    *,
    agent_name: str,
    action_text: str,
    intent_text: str = "",
    spatial_belief_text: str = "",
    time_text: str = "",
    physical_state_text: str = "",
    internal_state_text: str = "",
    recent_experience_text: str = "",
) -> str:
    name = str(agent_name or "Agent").strip()
    action = str(action_text or "").strip()
    sections = [f"你是 {name}。"]
    _append_section(sections, "你当前想要：", intent_text)
    sections.append(f"你打算做：{action}")
    _append_section(sections, "你记得你家里是：", spatial_belief_text)
    _append_section(sections, "现在是：", time_text)
    _append_section(sections, f"{name} 当前的身体状态是：", physical_state_text)
    _append_section(sections, f"{name} 当前感受到的内部状态是：", internal_state_text)
    _append_section(sections, f"{name} 最近经历了：", recent_experience_text)
    sections.append(_transition_instructions(name))
    return "\n\n".join(sections).strip()


def build_state_transition_prompt_from_context(
    context: StateTransitionPromptContext,
) -> str:
    return build_state_transition_prompt(
        agent_name=context.agent_name,
        action_text=context.action_text,
        intent_text=context.intent_text,
        spatial_belief_text=context.spatial_belief_text,
        time_text=context.time_text,
        physical_state_text=context.physical_state_text,
        internal_state_text=context.internal_state_text,
        recent_experience_text=context.recent_experience_text,
    )


def _append_section(sections: list[str], heading: str, content: str) -> None:
    cleaned = str(content or "").strip()
    if cleaned:
        sections.append(f"{heading}\n\n{cleaned}")


def _transition_instructions(name: str) -> str:
    return f"""
预测一下，在你做完这个动作，你下一刻状态是怎样的？

需要考虑：

1. 推测动作后 {name} 所在的房间，以及靠近、拿着、注视或正在交互的对象。
2. 推测姿态、朝向、身体动作和交互状态可能发生的变化。
3. 推测疲劳、压力、紧张、饥饿、口渴和卫生状态是否发生即时变化。
4. 推测环境中哪些元素可能发生变化。
5. 根据现实世界中的完整动作过程估计耗时。
6. 只预测这个动作直接造成的下一刻状态，不继续替 {name} 安排后续行动。
7. 判断动作完成后的状态是否已经满足、近似满足，或者仍未满足当前 Intent。这个判断可以宽松一些：如果实践上已经基本达到目的，不必要求穷举所有细节。

请严格输出以下 JSON：

{{
  "transition_outcome": {{
    "status": "success | partial | failed",
    "description": "动作最可能产生的直接结果",
    "reason": "形成这一预测的主要依据"
  }},
  "intent_satisfaction": {{
    "status": "satisfied | approximately_satisfied | not_satisfied",
    "reason": "为什么动作后的状态已经满足、近似满足或仍未满足当前 Intent"
  }},
  "elapsed_seconds": 0,
  "next_self_state": {{
    "area": "动作后所在的房间",
    "near_element": "动作后靠近的元素，没有则为空",
    "posture": "动作后的姿态",
    "facing_or_gaze": "动作后的朝向或注视对象",
    "holding": ["动作后拿着的物品"],
    "interacting_with": ["动作后正在交互的元素"],
    "worn_items_change": "穿戴变化，没有则为空",
    "body_surface_change": "身体表面变化，没有则为空"
  }},
  "internal_state_changes": {{
    "hunger": 0,
    "thirst": 0,
    "hygiene": 0,
    "stress": 0,
    "tension": 0,
    "fatigue": 0,
    "mental_change": "即时的主观心理变化，没有则为空"
  }},
  "spatial_belief_updates": [
    {{
      "element": "发生变化的元素",
      "state_change": "状态变化"
    }}
  ],
  "expected_feedback": "{name}预计自己紧接着会经历到的结果",
  "uncertainty": "预测中无法确认的部分，没有则为空"
}}
""".strip()

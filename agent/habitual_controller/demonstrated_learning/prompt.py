from __future__ import annotations

import json

from agent.habitual_controller.cue_memory import CueMemoryRecord

from .types import DemonstratedActionObservation


def build_demonstrated_habit_learning_prompt(
    *,
    agent_name: str,
    intent_text: str,
    observations: list[DemonstratedActionObservation],
    existing_persona_habits: list[CueMemoryRecord] | None = None,
    existing_demonstrated_habits: list[CueMemoryRecord] | None = None,
) -> str:
    payload = [observation.to_prompt_dict() for observation in observations]
    existing = [
        {
            "response_key": record.response_key,
            "response_text": record.response_text,
            "required_cue_ids": list(record.required_cue_ids),
        }
        for record in list(existing_persona_habits or [])
    ]
    demonstrated = [
        {
            "response_key": record.response_key,
            "response_text": record.response_text,
            "required_cue_ids": list(record.required_cue_ids),
            "source_intent": record.metadata.get("source_intent", ""),
        }
        for record in list(existing_demonstrated_habits or [])
    ]
    return f"""
有一个人叫 {agent_name}。{agent_name} 原本为了实现下面这个目标，进行了一系列有意识的 Goal-Directed 行为：

{intent_text}

下面列出了其中成功落实的动作。每个动作都附带动作发生前真实存在的 Context Features：

{json.dumps(payload, ensure_ascii=False, indent=2)}

这个角色原本已经具有以下 Persona Habits：

{json.dumps(existing, ensure_ascii=False, indent=2)}

这个角色已经从过去的受控任务中学到以下 Demonstrated Habits：

{json.dumps(demonstrated, ensure_ascii=False, indent=2)}

请判断其中哪些动作适合在相同或相近的稳定任务中，作为由情境快速准备的习惯性反应。

判断原则：
- 选择可重复、可执行，并且能够由动作前情境稳定提示的行为。
- 上一个成功动作可以是重要的顺序 Cue。
- 地点、可见物体、物体状态、身体状态也可以成为 Cue。
- 不要仅因为动作曾经发生就学习它；一次性的权衡、偶然补救和过度依赖特殊措辞的动作可以拒绝。
- selected_cue_ids 只能逐字选择该 observation 的 available_context_features 中真实存在的 feature_id。
- 通常选择一到三个最有提示作用的 Cue，不要把所有 Features 都作为必要条件。
- 除非时间本身对行为必要，否则不要选择 time_period Cue。
- 这些行为原本是有意识的 Goal-Directed 行为，因此形成的反应全部是 conscious。
- canonical_action_signature 使用简短、稳定的英文 snake_case，表达动作及核心对象，例如 place_salmon_on_counter。
- habitual_action 是脱离本次任务叙事后仍可直接执行的单一动作，用 {agent_name} 开头；不要包含“然后继续”、目标说明或下一步计划。
- 如果动作与某条 Persona Habit 的 response 实质相同，canonical_action_signature 必须复用它现有的 response_key，帮助检索阶段识别重复。
- 如果当前目标下的动作与某条 Demonstrated Habit 实质相同，必须复用它的 response_key；这是强化旧联结，不是创建新行为。

每个 observation 都必须返回一个判断。只返回 JSON：

{{
  "judgments": [
    {{
      "observation_id": "原 observation_id",
      "should_learn": true,
      "reason": "简短判断理由",
      "selected_cue_ids": ["真实存在的 feature_id"],
      "canonical_action_signature": "snake_case",
      "habitual_action": "{agent_name}执行的单一具体动作",
      "context_text": "用一句自然中文描述这些 Cue 形成的情境"
    }}
  ]
}}
""".strip()

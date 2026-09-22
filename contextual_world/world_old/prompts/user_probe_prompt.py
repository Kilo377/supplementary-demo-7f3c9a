from __future__ import annotations

import json


def build_user_probe_prompt(
    user_command: str,
    *,
    actor_name: str,
    world_state: dict,
    focus_element_name: str = "",
) -> str:
    world_json = json.dumps(world_state, ensure_ascii=False, indent=2)
    focus_text = ""
    if focus_element_name:
        focus_text = f"\n当前 {actor_name} 已经走到了目标附近，当前主要交互目标是：{focus_element_name}\n"
    return f"""你是一个家庭场景世界探针规划器。

你的任务：
1. 阅读当前世界状态
2. 理解用户的自然语言命令
3. 输出严格 JSON，表示应该修改哪些元素的状态

要求：
- 只能基于给定世界中的 area 和 element 做匹配
- 不要编造不存在的 element id
- 如果用户说“让厨房整个烧起来”，可以使用 area 级操作
- 如果用户说“让炉子着火”，应尽量定位到最相关元素，例如灶台、锅等
- 用户命令也可能是在描述 agent 已经执行的交互动作，例如“{actor_name}坐上了单人沙发并调整了抱枕1的位置”
- 对这类交互动作，请把它理解成已经发生的环境变化，并把相关元素 physical_status 改成更贴切的短英文状态词
- 例如可以使用：occupied, adjusted, held, opened, closed, regular, on_fire, leaking, withered, wet, broken, smoky
- 如果动作意味着把异常恢复正常，例如“关掉漏水的花洒”或“扑灭灶台上的火”，可以把对应状态改回 regular，或改成更合理的结果状态
- 对交互动作，默认只修改直接相关的 1 到 3 个元素，不要把整个 area 一起改掉
- 只有在用户明确说“整个厨房”“整个客厅”这类整片区域时，才允许使用 area 级操作
- 对交互动作，优先使用 element 级操作，不要把无关家具一起改状态
- physical_status 使用英文短词，例如：on_fire, leaking, withered, wet, broken, smoky
- interaction_status 只在非常明确的人物使用语境下才修改，例如 in_use、idle
- summary 必须是一句简短中文结果，像“{actor_name}已经坐下，并调整了抱枕位置。”
- summary 不要写推理过程，不要解释为什么
- 只返回 JSON，不要解释，不要 markdown
- 默认的正常状态是 regular

JSON schema:
{{
  "summary": "中文简述",
  "operations": [
    {{
      "scope": "element" | "area",
      "target_area_id": "可选",
      "target_element_ids": ["可选，优先使用 id"],
      "target_element_names": ["可选，若拿不准 id 可以给 name"],
      "physical_status": "可选状态词",
      "evolution_status": "可选 changing 或 stable",
      "interaction_status": "可选状态词",
      "reason": "为什么这样匹配"
    }}
  ]
}}

当前世界状态:
{world_json}
{focus_text}

用户命令:
{user_command}
"""

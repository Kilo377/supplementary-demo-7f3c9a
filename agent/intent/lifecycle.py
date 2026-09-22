from __future__ import annotations

import json
from dataclasses import dataclass

from agent.intent.state import IntentState
from llm.api_manager import APIManager


@dataclass
class IntentLifecycleResult:
    intent_status: str
    reason: str
    updated_intent: str
    raw_response: str = ""


def build_intent_lifecycle_prompt(
    *,
    agent_name: str,
    personality: str,
    current_state_belief: str,
    intent: IntentState,
    intent_decided_time: str,
    current_time: str,
    short_time_memory_text: str,
    force_reflect: bool = False,
) -> str:
    force_reflect_text = ""
    if force_reflect:
        force_reflect_text = "\n你已经为这件事做了若干步。现在认真判断它是否已经到达一个你在现实生活中能够接受的终点，不要仅仅因为还能继续做类似动作就保持 active。"

    personality_text = personality.strip().rstrip("。.!！") or "会根据当下情况做判断"
    state_text = current_state_belief.strip() or "你刚刚完成了眼前这一步。"

    return f"""你叫 {agent_name}，现在的状态是：
{state_text}

你觉得你是一个{personality_text}的人。

就在{intent_decided_time}，你决定：
{intent.intent_text}

现在是{current_time}，你做了这些事情：
{short_time_memory_text}
{force_reflect_text}

你觉得已经完成你想做的了吗？
如果没完成，你还想继续做吗？

请填写状态和理由：

- active：当前仍有一个明确、具体、尚未完成，而且由现有环境或近期经历支持的下一步。
  不能只因为理论上还能继续做更多事情，就保持 active。

- satisfied：根据当前可见环境、已经完成的动作和近期反馈，
  你已经觉得满足了这个 Intent，这同样取决你的性格，举个一个完成任务的例子，如果你是一个将就的人，那可能草草了事。如果你是一个比较负责的人，可能会对自己有更高的标准。

- deferred：Intent 尚未充分满足，但当前受到疲劳、时间、环境缺失、
  路径受阻或其他现实条件限制，适合以后继续。

- abandoned：你已经明确不再打算实现这个 Intent。
  任务较长、暂时休息或环境信息不足，本身不等于 abandoned。

要求：
- reason 使用第一人称，用一句简短中文说明你的真实判断
- 先按原始 Intent 的范围判断，再看已经做过的事情是否足够满足它
- 如果只是完成了一个房间、一件家具或一个局部步骤，通常不应判为 satisfied，除非原始 Intent 本来就只要求这个局部
- 如果仍保留这个 Intent 但想调整它的表达，填写 updated_intent，否则留空
- 只返回 JSON，不要解释推理过程

返回格式：
{{
  "intent_status": "active | satisfied | deferred | abandoned",
  "reason": "一句简短中文理由",
  "updated_intent": "若需调整 intent，则填写；否则为空字符串"
}}
"""


def evaluate_intent_lifecycle(
    *,
    agent_name: str,
    personality: str,
    current_state_belief: str,
    intent: IntentState,
    intent_decided_time: str,
    current_time: str,
    short_time_memory_text: str,
    force_reflect: bool = False,
    provider_name: str = "ollama",
    model: str | None = None,
) -> IntentLifecycleResult:
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.intent_lifecycle",
    )
    prompt = build_intent_lifecycle_prompt(
        agent_name=agent_name,
        personality=personality,
        current_state_belief=current_state_belief,
        intent=intent,
        intent_decided_time=intent_decided_time,
        current_time=current_time,
        short_time_memory_text=short_time_memory_text,
        force_reflect=force_reflect,
    )
    raw = api.generate(prompt, model=model)
    parsed = json.loads(_extract_json_text(raw))
    status = str(parsed.get("intent_status", "active") or "active").strip().lower()
    if status not in {"active", "satisfied", "deferred", "abandoned"}:
        status = "active"
    return IntentLifecycleResult(
        intent_status=status,
        reason=parsed.get("reason", "") or "",
        updated_intent=parsed.get("updated_intent", "") or "",
        raw_response=raw,
    )


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
    raise ValueError("No JSON object found in intent reflection response.")

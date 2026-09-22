from __future__ import annotations

import json
from dataclasses import dataclass

from agent.intent.state import IntentState
from llm.api_manager import APIManager


@dataclass
class IntentProgressResult:
    progress_item: str
    raw_response: str = ""


def build_intent_progress_prompt(
    *,
    agent_name: str,
    intent: IntentState,
    action_proposal: str,
    execution_feedback: str,
) -> str:
    progress_text = "\n".join(f"- {item}" for item in intent.progress) or "暂无。"
    return f"""这是 {agent_name} 短期做的事情。
它只记录了和这个 intent 直接相关、已经发生过的行动进展。

{agent_name} 这个阶段想要做的事情是：
{intent.intent_text}

为了完成这个事情，已有进度：
{progress_text}

刚刚提出的动作：
{action_proposal}

环境反馈：
{execution_feedback}

请把刚刚真正推进了 intent 的部分，压缩成一条短期进度。

要求：
- 只记录已经发生的事情
- 只记录和 active intent 相关的事情
- 如果刚刚没有推进 intent，progress_item 返回空字符串
- 不要写计划、原因、推理
- 不要重复已有进度里已经表达过的内容
- 只返回 JSON

返回格式：
{{
  "progress_item": "一条简短进度；如果没有进展则为空字符串"
}}
"""


def summarize_intent_progress(
    *,
    agent_name: str,
    intent: IntentState,
    action_proposal: str,
    execution_feedback: str,
    provider_name: str = "ollama",
    model: str | None = None,
) -> IntentProgressResult:
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.intent_progress",
    )
    prompt = build_intent_progress_prompt(
        agent_name=agent_name,
        intent=intent,
        action_proposal=action_proposal,
        execution_feedback=execution_feedback,
    )
    raw = api.generate(prompt, model=model)
    parsed = json.loads(_extract_json_text(raw))
    return IntentProgressResult(
        progress_item=(parsed.get("progress_item", "") or "").strip(),
        raw_response=raw,
    )


def fallback_progress_item(
    *,
    agent_name: str,
    intent: IntentState,
    action_proposal: str,
    execution_feedback: str,
) -> str:
    text = execution_feedback.strip() or action_proposal.strip()
    if not text:
        return ""
    if any(word in text for word in ["找不到", "不能", "无法", "没有", "失败", "记错", "摔"]):
        return ""
    if not _looks_related_to_intent(intent.intent_text, text):
        return ""
    return text.rstrip("。") + "。"


def append_progress(intent: IntentState, item: str, *, max_items: int = 12) -> None:
    cleaned = item.strip()
    if not cleaned:
        return
    if not cleaned.endswith(("。", "！", "？")):
        cleaned = f"{cleaned}。"
    if _is_duplicate_progress(intent.progress, cleaned):
        return
    intent.progress.append(cleaned)
    if len(intent.progress) > max_items:
        intent.progress[:] = intent.progress[-max_items:]


def format_progress_for_prompt(intent: IntentState) -> str:
    if not intent.progress:
        return "暂无。"
    return "\n".join(f"- {item}" for item in intent.progress)


def _looks_related_to_intent(intent_text: str, text: str) -> bool:
    if any(word in intent_text for word in ["做饭", "填饱肚子", "吃"]):
        return any(word in text for word in ["厨房", "冰箱", "食材", "操作台", "灶台", "锅", "水槽", "砧板", "切", "清洗", "加热", "煮", "炒", "吃"])
    if any(word in intent_text for word in ["喝水", "口渴", "找点水"]):
        return any(word in text for word in ["水", "杯", "冰箱", "水槽", "喝"])
    return True


def _is_duplicate_progress(existing: list[str], item: str) -> bool:
    normalized = _normalize(item)
    return any(normalized == _normalize(old) for old in existing)


def _normalize(text: str) -> str:
    return text.replace("已经", "").replace("了", "").replace("。", "").replace(" ", "")


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
    raise ValueError("No JSON object found in intent progress response.")

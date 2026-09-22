from __future__ import annotations

import re

from agent.belief.short_time_memory import ShortTermMemory
from agent.belief.spatial_memory.spatial_belief import SpatialBelief
from agent.intent.state import IntentState
from agent.intuition import IntuitionResult
from agent.think.types import ThinkResult
from llm.api_manager import APIManager

from .prompt import build_action_proposal_prompt
from .types import ActionProposalResult


DEBUG_ACTION_PROPOSAL_PROMPT = False


def set_debug_action_proposal_prompt(enabled: bool) -> None:
    global DEBUG_ACTION_PROPOSAL_PROMPT
    DEBUG_ACTION_PROPOSAL_PROMPT = enabled


def propose_next_action(
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
    provider_name: str = "ollama",
    model: str | None = None,
) -> ActionProposalResult:
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.action_proposal",
    )
    prompt = build_action_proposal_prompt(
        agent_name=agent_name,
        intent=intent,
        short_time_memory=short_time_memory,
        intuition=intuition,
        pre_think_intuition=pre_think_intuition,
        think_result=think_result,
        spatial_belief=spatial_belief,
        current_area_id=current_area_id,
        scene_name=scene_name,
    )
    if DEBUG_ACTION_PROPOSAL_PROMPT:
        print("=" * 72)
        print("Action Proposal Prompt")
        print(prompt)
        print("=" * 72, flush=True)
    raw = api.generate(prompt, model=model)
    text = _clean_action_text(raw, agent_name=agent_name)
    if text.startswith("你"):
        text = f"{agent_name}{text[1:]}"
    elif text.startswith(("他", "她")):
        text = f"{agent_name}{text[1:]}"
    return ActionProposalResult(
        action_text=text,
        provider_name=api.provider_name,
        model=api.route.model,
    )


def _clean_action_text(text: str, *, agent_name: str) -> str:
    cleaned = _strip_code_fence(text.strip())
    for marker in [
        f"{agent_name}接下来先打算做的是：",
        f"{agent_name}接下来先打算做的是:",
        f"{agent_name}接下来先做的是：",
        f"{agent_name}接下来先做的是:",
    ]:
        if marker in cleaned:
            cleaned = cleaned.rsplit(marker, 1)[1].strip()
            break

    bold_candidates = re.findall(r"\*\*(.*?)\*\*", cleaned, flags=re.S)
    if bold_candidates:
        candidates = [_remove_markdown(candidate.strip()) for candidate in bold_candidates]
        action_candidates = [candidate for candidate in candidates if _looks_like_action_candidate(candidate, agent_name)]
        named = [candidate for candidate in action_candidates if agent_name in candidate]
        if named:
            cleaned = named[-1].strip()
        elif action_candidates:
            cleaned = action_candidates[0].strip()
        else:
            cleaned = candidates[0].strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    lines = [_remove_markdown(line) for line in lines]
    action_lines = [line for line in lines if _looks_like_action_line(line, agent_name)]
    if action_lines:
        cleaned = action_lines[-1]
    elif lines:
        cleaned = lines[0]

    cleaned = _remove_markdown(cleaned)
    cleaned = re.sub(r"（[^）]*或[^）]*）", "", cleaned)
    cleaned = cleaned.strip(" ：:。")
    cleaned = _trim_to_one_action(cleaned)
    if agent_name not in cleaned:
        cleaned = f"{agent_name}{cleaned}"
    if not cleaned.endswith(("。", "！", "？")):
        cleaned = f"{cleaned}。"
    return cleaned


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    parts = stripped.split("```")
    if len(parts) >= 3:
        return parts[1].replace("json", "", 1).strip()
    return stripped.replace("```", "").strip()


def _remove_markdown(text: str) -> str:
    cleaned = re.sub(r"^\s*[-*]\s*", "", text)
    cleaned = re.sub(r"^\s*\d+[.)、]\s*", "", cleaned)
    cleaned = cleaned.replace("**", "").replace("*", "").replace("`", "")
    return cleaned.strip()


def _looks_like_action_line(text: str, agent_name: str) -> bool:
    if not _looks_like_action_candidate(text, agent_name):
        return False
    return agent_name in text


def _looks_like_action_candidate(text: str, agent_name: str) -> bool:
    if not text or any(word in text for word in ["推理", "逻辑", "目标分析", "当前状态", "环境匹配", "排除干扰"]):
        return False
    if text.strip(" ：:。") in {
        f"{agent_name}接下来先打算做的是",
        f"{agent_name}接下来先做的是",
    }:
        return False
    if text.startswith(("因此", "所以", "通常", "考虑到", "综合", "注：", "如果", "若")):
        return False
    action_verbs = ["走", "伸手", "拿", "放", "打开", "关", "查看", "坐", "站", "转身", "靠近", "清洗", "倒", "按", "拉开", "整理", "检查", "擦", "收拾"]
    return agent_name in text or any(verb in text for verb in action_verbs)


def _trim_to_one_action(text: str) -> str:
    sentence = re.split(r"[。！？]\s*", text, maxsplit=1)[0].strip()
    if not sentence:
        sentence = text.strip()
    splitters = [
        "，然后",
        "然后",
        "，并且",
        "并且",
        "，并",
        "并",
        "，同时",
        "同时",
        "，接着",
        "接着",
        "，以便",
        "以便",
        "或者",
        "或",
    ]
    changed = True
    while changed:
        changed = False
        for splitter in splitters:
            if splitter in sentence:
                sentence = sentence.split(splitter, 1)[0].strip()
                changed = True
                break
    return sentence

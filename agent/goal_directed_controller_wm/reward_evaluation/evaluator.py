from __future__ import annotations

import json
import re

from agent.goal_directed_controller_wm.state_transition import (
    StateTransitionRolloutNode,
    StateTransitionRolloutResult,
    build_state_transition_context,
)
from agent.goal_directed_controller_wm.state_transition.json_output import parse_json_object
from agent.intent.state import IntentState
from llm.api_manager import APIManager

from .prompt import build_reward_evaluation_prompt
from .types import RewardEvaluation, RewardEvaluationResult


def evaluate_action_rewards(
    agent,
    *,
    intent: IntentState,
    rollout: StateTransitionRolloutResult,
    provider_name: str = "ollama",
    model: str | None = None,
) -> RewardEvaluationResult:
    action_ids = [root.path_text for root in rollout.roots]
    prompt = build_reward_evaluation_prompt(
        agent_name=str(getattr(agent, "name", "Agent") or "Agent"),
        intent_text=intent.intent_text,
        personality_text=str(getattr(agent, "personality", "") or "").strip(),
        current_state_text=_current_state_text(agent, intent_text=intent.intent_text),
        action_chains_text=format_action_chains_for_reward(rollout),
    )
    raw_response = ""
    api = APIManager(
        provider_name=provider_name,
        task_name="agent.world_model.reward_evaluation",
    )
    try:
        raw_response = api.generate(
            prompt,
            model=model,
        )
        parsed = parse_json_object(raw_response)
        evaluations = _parse_evaluations(parsed, expected_action_ids=action_ids)
        return RewardEvaluationResult(
            evaluations=evaluations,
            prompt=prompt,
            raw_response=raw_response,
            provider_name=api.provider_name,
            model=api.route.model,
        )
    except Exception as error:
        return RewardEvaluationResult(
            prompt=prompt,
            raw_response=raw_response,
            provider_name=api.provider_name,
            model=api.route.model,
            error=f"Reward evaluation failed: {error}",
        )


def format_action_chains_for_reward(rollout: StateTransitionRolloutResult) -> str:
    return "\n\n".join(_format_action_chain(root) for root in rollout.roots)


def _format_action_chain(root: StateTransitionRolloutNode) -> str:
    lines = [
        f"动作链 {root.path_text}",
        f"被评分的初始动作：{root.candidate.action_text}",
    ]
    total_seconds = 0
    node = root
    while node is not None:
        lines.append(f"第{node.depth}步动作：{node.candidate.action_text}")
        result = node.transition_result
        if result.error or result.transition is None:
            lines.append(f"第{node.depth}步预测失败：{result.error or '没有预测结果'}")
        else:
            transition = result.transition
            total_seconds += transition.elapsed_seconds
            lines.append(
                f"第{node.depth}步预测结果："
                f"{transition.outcome.description or '(empty)'}"
            )
            lines.append(
                f"第{node.depth}步目标状态："
                f"{transition.intent_satisfaction.status}；"
                f"{transition.intent_satisfaction.reason or '(empty)'}"
            )
            lines.append(
                "第"
                f"{node.depth}步预测后的自身状态："
                + json.dumps(
                    transition.to_dict()["next_self_state"],
                    ensure_ascii=False,
                )
            )
            lines.append(
                "第"
                f"{node.depth}步内部状态变化："
                + json.dumps(
                    transition.to_dict()["internal_state_changes"],
                    ensure_ascii=False,
                )
            )
            if transition.spatial_belief_updates:
                lines.append(
                    f"第{node.depth}步环境变化："
                    + json.dumps(
                        transition.to_dict()["spatial_belief_updates"],
                        ensure_ascii=False,
                    )
                )
            if transition.uncertainty:
                lines.append(f"第{node.depth}步不确定性：{transition.uncertainty}")
        node = node.children[0] if node.children else None
    lines.append(f"累计预测耗时：{total_seconds}秒")
    lines.append(f"动作链停止原因：{_terminal_node(root).stop_reason or 'unknown'}")
    return "\n".join(lines)


def _current_state_text(agent, *, intent_text: str) -> str:
    context = build_state_transition_context(
        agent,
        action_text="",
        intent_text=intent_text,
        recent_experience_count=0,
    )
    sections = []
    if context.time_text:
        sections.append(f"时间：{context.time_text}")
    if context.physical_state_text:
        sections.append(f"身体状态：\n{context.physical_state_text}")
    if context.internal_state_text:
        sections.append(f"内部状态：\n{context.internal_state_text}")
    return "\n\n".join(sections)


def _parse_evaluations(
    parsed: dict,
    *,
    expected_action_ids: list[str],
) -> list[RewardEvaluation]:
    raw_items = parsed.get("evaluations", [])
    if not isinstance(raw_items, list):
        raise ValueError("evaluations is not a list.")
    expected = set(expected_action_ids)
    evaluations: dict[str, RewardEvaluation] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        action_id = _normalize_action_id(item.get("action_id"))
        if action_id not in expected or action_id in evaluations:
            continue
        evaluations[action_id] = RewardEvaluation(
            action_id=action_id,
            reward=_score(item.get("reward")),
            reason=str(item.get("reason", "") or "").strip(),
        )
    missing = [action_id for action_id in expected_action_ids if action_id not in evaluations]
    if missing:
        raise ValueError(f"Missing reward evaluations: {', '.join(missing)}")
    return [evaluations[action_id] for action_id in expected_action_ids]


def _score(value) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(10, score))


def _normalize_action_id(value) -> str:
    text = str(value or "").strip()
    match = re.search(r"\d+(?:\.\d+)*", text)
    return match.group(0) if match else text


def _terminal_node(root: StateTransitionRolloutNode) -> StateTransitionRolloutNode:
    node = root
    while node.children:
        node = node.children[0]
    return node

from __future__ import annotations

import json

from llm.api_manager import APIManager

from .prompt import build_cumulative_reward_prompt
from .types import CumulativeRewardResult


def calculate_cumulative_reward(
    intent_satisfaction: float,
    execution_steps: int,
) -> float:
    steps = max(0, int(execution_steps))
    if steps == 0:
        return 0.0
    satisfaction = min(10.0, max(0.0, float(intent_satisfaction)))
    return satisfaction / steps


def aggregate_cumulative_rewards(
    results: list[CumulativeRewardResult],
) -> dict:
    total_steps = sum(result.execution_steps for result in results)
    total_satisfaction = sum(result.intent_satisfaction for result in results)
    errors = [result.error for result in results if result.error]
    return {
        "intent_count": len(results),
        "total_intent_satisfaction": total_satisfaction,
        "execution_steps": total_steps,
        "cumulative_reward": (
            total_satisfaction / total_steps
            if total_steps > 0
            else 0.0
        ),
        "reason": f"汇总了{len(results)}个Intent的模拟结束结算。",
        "intent_results": [result.to_dict() for result in results],
        "error": "；".join(errors),
    }


def evaluate_cumulative_reward(
    *,
    agent_name: str,
    intent_text: str,
    results: list,
    home,
    final_desire_state: dict | None = None,
    provider_name: str = "ollama",
    model: str | None = None,
) -> CumulativeRewardResult:
    steps = len(results)
    final_status = (
        str(getattr(results[-1], "intent_status", "active") or "active")
        if results
        else "active"
    )
    if steps == 0:
        return CumulativeRewardResult(
            intent_text=intent_text,
            intent_satisfaction=0.0,
            execution_steps=0,
            cumulative_reward=0.0,
            reason="模拟没有产生实际执行步骤。",
            final_intent_status=final_status,
            provider_name=provider_name,
            model=model,
        )

    prompt = build_cumulative_reward_prompt(
        agent_name=agent_name,
        intent_text=intent_text,
        final_intent_status=final_status,
        actual_trajectory_text=format_actual_trajectory(results),
        final_state_text=format_final_state(
            results,
            home=home,
            final_desire_state=final_desire_state,
        ),
    )
    raw_response = ""
    api = APIManager(
        provider_name=provider_name,
        task_name="evaluation.cumulative_reward",
    )
    try:
        raw_response = api.generate(
            prompt,
            model=model,
        )
        parsed = json.loads(_extract_json_text(raw_response))
        satisfaction = _satisfaction(_satisfaction_value(parsed))
        return CumulativeRewardResult(
            intent_text=intent_text,
            intent_satisfaction=satisfaction,
            execution_steps=steps,
            cumulative_reward=calculate_cumulative_reward(satisfaction, steps),
            reason=str(parsed.get("reason", "") or "").strip(),
            final_intent_status=final_status,
            prompt=prompt,
            raw_response=raw_response,
            provider_name=api.provider_name,
            model=api.route.model,
        )
    except Exception as error:
        return CumulativeRewardResult(
            intent_text=intent_text,
            intent_satisfaction=0.0,
            execution_steps=steps,
            cumulative_reward=0.0,
            final_intent_status=final_status,
            prompt=prompt,
            raw_response=raw_response,
            provider_name=api.provider_name,
            model=api.route.model,
            error=f"Cumulative reward evaluation failed: {error}",
        )


def format_actual_trajectory(results: list) -> str:
    lines = []
    for index, result in enumerate(results, start=1):
        proposal = str(getattr(result, "action_proposal_text", "") or "").strip()
        feedback = _actual_feedback(result)
        lines.append(f"{index}. 实际动作：{proposal or '(没有动作描述)'}")
        lines.append(f"   实际结果：{feedback or '(没有World反馈)'}")
    return "\n".join(lines)


def format_final_state(
    results: list,
    *,
    home,
    final_desire_state: dict | None,
) -> str:
    final_result = results[-1]
    payload = {
        "agent_state": dict(getattr(final_result, "actor_state", {}) or {}),
        "self_belief": str(getattr(final_result, "self_belief", "") or ""),
        "desire_state": final_desire_state or {},
        "world_state": _world_state(home),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _world_state(home) -> list[dict]:
    areas = []
    for area in getattr(home, "areas", ()) or ():
        elements = []
        for element in getattr(area, "elements", ()) or ():
            elements.append(
                {
                    "name": element.name,
                    "physical_status": element.physical_status,
                    "evolution_status": element.evolution_status,
                    "interaction_status": element.interaction_status,
                    "state_details": dict(element.state_details),
                }
            )
        areas.append({"area": area.name, "elements": elements})
    return areas


def _actual_feedback(result) -> str:
    feedback = getattr(result, "environment_feedback", {}) or {}
    if isinstance(feedback, dict):
        summary = str(feedback.get("perception_summary", "") or "").strip()
        if summary:
            return summary
    return str(
        getattr(result, "execution_narration", "")
        or getattr(result, "action_text", "")
        or ""
    ).strip()


def _satisfaction(value) -> float:
    try:
        return min(10.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        raise ValueError("intent_satisfaction must be a number from 0 to 10.")


def _satisfaction_value(parsed: dict):
    for key in (
        "intent_satisfaction",
        "agent_satisfaction",
        "satisfaction",
    ):
        if key in parsed:
            return parsed[key]
    return None


def _extract_json_text(text: str) -> str:
    stripped = str(text or "").strip()
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
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    raise ValueError("No JSON object found in cumulative reward response.")

from __future__ import annotations

from typing import Any

from .desire_update_prompt import DesireUpdateResult, update_desire_state


def apply_desire_update(
    *,
    agent: Any,
    intent_text: str,
    intent_status: str,
    update_mode: str,
    provider_name: str,
    model: str | None,
    context_count: int,
) -> tuple[DesireUpdateResult, dict]:
    """Apply one Desire update and return a visualization-friendly record."""
    before = agent.desire_state
    before_dict = before.to_dict()
    memory = getattr(agent, "short_time_memory", None)
    context = memory.format_for_prompt(count=context_count) if memory is not None else ""
    long_term_memory = getattr(agent, "long_term_memory", None)
    retrieve = getattr(long_term_memory, "retrieve", None)
    personal_context = retrieve(query=intent_text) if callable(retrieve) else ""
    self_state = getattr(agent, "self_state", None)
    self_belief = str(getattr(self_state, "self_belief", "") or "")

    update = update_desire_state(
        agent_name=agent.name,
        desire_state=before,
        intent_text=intent_text,
        intent_status=intent_status,
        intent_cycle_context=context,
        self_belief=self_belief,
        provider_name=provider_name,
        model=model,
        personal_context=personal_context,
        update_mode=update_mode,
    )
    agent.desire_state = update.desire_state
    record = {
        "intent_text": intent_text,
        "intent_status": intent_status,
        "update_mode": update_mode,
        "benefit": update.benefit,
        "cost": update.cost,
        "physiological_reason": update.physiological_reason,
        "internal_state_reason": update.internal_state_reason,
        "mental_reason": update.mental_reason,
        "before": before_dict,
        "after": agent.desire_state.to_dict(),
    }
    return update, record

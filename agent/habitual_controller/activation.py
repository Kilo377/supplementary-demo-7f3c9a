from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from agent.habitual_controller.cue_memory import CueMemory
from agent.habitual_controller.cue_extraction import CueExtractionResult
from agent.habitual_controller.cue_retrieval import APIEmbeddingClient, CueRetrievalSettings, retrieve_cue_memory

from .types import HabitualActivationResult, PreparedHabitualResponse


def activate_habitual(
    agent,
    context: CueExtractionResult,
    *,
    activation_threshold: float = 0.35,
    embedding_provider_name: str = "ollama",
    embedding_model: str = "bge-m3:latest",
) -> HabitualActivationResult:
    now = getattr(getattr(agent, "time_belief", None), "current_datetime", None) or datetime.now()
    last_executed_at = getattr(agent, "habitual_last_executed_at", {})
    last_considered_at = getattr(agent, "habitual_last_considered_at", {})
    persona_memory = agent.long_term_memory.cue_memory
    demonstrated_memory = agent.long_term_memory.demonstrated_habit_memory
    merged_records, sources_by_key = _merge_memory_pools(
        persona_memory,
        demonstrated_memory,
    )
    available_memory = CueMemory(
        records=[
            record
            for record in merged_records
            if not _is_on_cooldown(
                record,
                now=now,
                last_executed_at=last_executed_at,
                last_considered_at=last_considered_at,
            )
        ],
        habit_strength_threshold=persona_memory.habit_strength_threshold,
    )
    settings = CueRetrievalSettings(
        activation_threshold=min(1.0, max(0.0, float(activation_threshold))),
        embedding_provider_name=embedding_provider_name,
        embedding_model=embedding_model,
    )
    try:
        retrieval = retrieve_cue_memory(
            available_memory,
            context,
            embedder=APIEmbeddingClient(embedding_provider_name, embedding_model),
            settings=settings,
        )
    except Exception:
        return HabitualActivationResult(retrieval_status="embedding_unavailable")

    tendencies: list[PreparedHabitualResponse] = []
    for candidate in retrieval.candidates:
        if candidate.familiarity_status != "confirmed":
            continue
        record = candidate.record
        tendencies.append(
            PreparedHabitualResponse(
                association_id=record.association_id,
                response_key=record.response_key,
                response_text=record.response_text,
                source_dimension=record.source_dimension,
                trigger_cue_ids=tuple(record.required_cue_ids or candidate.matching_cue_ids),
                habit_strength=record.habit_strength,
                activation=min(1.0, candidate.retrieval_score),
                awareness_type=record.awareness_type,
                estimated_duration=record.estimated_duration,
                required_body_resources=tuple(record.required_body_resources),
                cooldown_seconds=record.cooldown_seconds,
                memory_sources=sources_by_key.get(
                    record.canonical_association_key(),
                    (),
                ),
            )
        )
    tendencies.sort(key=lambda item: (item.activation, item.habit_strength), reverse=True)
    return HabitualActivationResult(
        tendencies=tendencies,
        retrieval_status=retrieval.status,
    )


def _merge_memory_pools(
    persona_memory: CueMemory,
    demonstrated_memory: CueMemory,
) -> tuple[list, dict[str, tuple[str, ...]]]:
    grouped = defaultdict(list)
    sources = defaultdict(set)
    for source, memory in (
        ("persona", persona_memory),
        ("demonstrated", demonstrated_memory),
    ):
        for record in memory.established_habits:
            key = record.canonical_association_key()
            grouped[key].append(record)
            sources[key].add(source)
    records = [
        max(group, key=lambda item: item.habit_strength)
        for group in grouped.values()
    ]
    return records, {
        key: tuple(sorted(values))
        for key, values in sources.items()
    }


def _latest_time(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _is_on_cooldown(
    record,
    *,
    now: datetime,
    last_executed_at: dict[str, datetime],
    last_considered_at: dict[str, datetime],
) -> bool:
    last_at = _latest_time(
        last_executed_at.get(record.association_id),
        last_considered_at.get(record.association_id),
    )
    return bool(
        last_at is not None
        and (now - last_at).total_seconds() < record.cooldown_seconds
    )

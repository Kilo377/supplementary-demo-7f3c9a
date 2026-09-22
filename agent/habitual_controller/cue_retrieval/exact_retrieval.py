from __future__ import annotations

from agent.habitual_controller.cue_memory import CueMemory

from .types import CueRetrievalCandidate


def retrieve_by_exact_cues(
    memory: CueMemory,
    cue_ids: list[str],
    *,
    limit: int,
) -> list[CueRetrievalCandidate]:
    query = set(cue_ids)
    visible_ids = {cue.split(":", 1)[1] for cue in query if cue.startswith("visible:")}
    visible_types = {cue.split(":", 1)[1] for cue in query if cue.startswith("visible_type:")}
    candidates: list[CueRetrievalCandidate] = []
    for record in memory.established_habits:
        has_visual_index = bool(record.visual_element_ids or record.visual_semantic_keys)
        object_match = bool(visible_ids & set(record.visual_element_ids)) or bool(
            visible_types & set(record.visual_semantic_keys)
        )
        if has_visual_index and not object_match:
            continue
        if record.cue_prefixes and not any(
            cue.startswith(prefix)
            for cue in query
            for prefix in record.cue_prefixes
        ):
            continue
        required = set(record.required_cue_ids)
        if required and not required.issubset(query):
            continue
        if not required and not record.cue_prefixes:
            continue
        prefix_matches = {
            cue
            for cue in query
            if any(cue.startswith(prefix) for prefix in record.cue_prefixes)
        }
        matching = (query & set(record.cue_ids)) | prefix_matches
        pattern_size = max(1, len(set(record.cue_ids)) + len(record.cue_prefixes))
        similarity = min(1.0, len(matching) / pattern_size)
        candidates.append(
            CueRetrievalCandidate(
                record=record,
                retrieval_method="exact_visual" if has_visual_index else "exact_structured",
                matching_cue_ids=tuple(sorted(matching)),
                similarity=similarity,
                retrieval_score=record.habit_strength * similarity,
                familiarity_status="not_required",
            )
        )
    candidates.sort(
        key=lambda item: (item.retrieval_score, item.similarity, item.record.habit_strength),
        reverse=True,
    )
    return candidates[:limit]

from __future__ import annotations

from agent.habitual_controller.cue_memory import CueMemory

from .embedding import Embedder, canonical_context_text, cosine_similarity, ensure_record_embedding
from .settings import CueRetrievalSettings
from .types import CueRetrievalCandidate


def retrieve_by_context_embedding(
    memory: CueMemory,
    cue_ids: list[str],
    *,
    embedder: Embedder,
    settings: CueRetrievalSettings,
) -> tuple[list[float], list[CueRetrievalCandidate]]:
    query_embedding = embedder.embed(canonical_context_text(cue_ids))
    candidates: list[CueRetrievalCandidate] = []
    query = set(cue_ids)
    for record in memory.established_habits:
        record_embedding = ensure_record_embedding(
            record,
            embedder=embedder,
            embedding_version=settings.embedding_version,
        )
        similarity = cosine_similarity(query_embedding, record_embedding)
        if similarity < settings.similarity_threshold:
            continue
        matching = query & set(record.cue_ids)
        candidates.append(
            CueRetrievalCandidate(
                record=record,
                retrieval_method="semantic_embedding",
                matching_cue_ids=tuple(sorted(matching)),
                similarity=similarity,
                retrieval_score=similarity * record.habit_strength,
            )
        )
    candidates.sort(
        key=lambda item: (item.retrieval_score, item.similarity, item.record.habit_strength),
        reverse=True,
    )
    return query_embedding, candidates[: settings.limit]

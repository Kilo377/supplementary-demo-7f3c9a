from __future__ import annotations

from agent.habitual_controller.cue_memory import CueMemory
from agent.habitual_controller.cue_extraction import CueExtractionResult

from .embedding import Embedder, ensure_memory_embeddings
from .dimension_embedding_retrieval import retrieve_by_dimension_embeddings
from .dimension_retrieval import retrieve_by_structured_dimensions
from .exact_retrieval import retrieve_by_exact_cues
from .semantic_retrieval import retrieve_by_context_embedding
from .settings import CueRetrievalSettings
from .types import CueRetrievalResult


def retrieve_cue_memory(
    memory: CueMemory,
    context: CueExtractionResult,
    *,
    embedder: Embedder | None = None,
    settings: CueRetrievalSettings | None = None,
) -> CueRetrievalResult:
    config = settings or CueRetrievalSettings()
    cue_ids = context.structured_feature_ids or [item.cue.cue_id for item in context.active_cues]
    if config.method not in {"dimension_embedding", "structured_dimension", "exact", "embedding", "exact_then_embedding"}:
        raise ValueError(f"Unknown cue retrieval method: {config.method}")

    if config.method == "dimension_embedding":
        if embedder is None:
            return CueRetrievalResult(
                retrieval_method="dimension_embedding_strength",
                similarity_threshold=config.activation_threshold,
                habit_strength_threshold=memory.habit_strength_threshold,
                status="embedding_unavailable",
            )
        candidates = retrieve_by_dimension_embeddings(
            memory,
            context,
            embedder=embedder,
            settings=config,
        )
        return CueRetrievalResult(
            candidates=candidates,
            retrieval_method="dimension_embedding_strength",
            similarity_threshold=config.activation_threshold,
            habit_strength_threshold=memory.habit_strength_threshold,
            status="response_prepared" if candidates else "no_response_prepared",
        )

    if config.method == "structured_dimension":
        candidates = retrieve_by_structured_dimensions(
            memory,
            context,
            activation_threshold=config.activation_threshold,
            limit=config.limit,
        )
        return CueRetrievalResult(
            candidates=candidates,
            retrieval_method="structured_dimension",
            similarity_threshold=config.activation_threshold,
            habit_strength_threshold=memory.habit_strength_threshold,
            status="candidates_found" if candidates else "no_match",
        )

    if config.method in {"exact", "exact_then_embedding"}:
        exact = retrieve_by_exact_cues(memory, cue_ids, limit=config.limit)
        if exact:
            methods = {item.retrieval_method for item in exact}
            return CueRetrievalResult(
                candidates=exact,
                retrieval_method=next(iter(methods)) if len(methods) == 1 else "exact",
                similarity_threshold=config.similarity_threshold,
                habit_strength_threshold=memory.habit_strength_threshold,
                status="candidates_found",
            )
        if config.method == "exact":
            return CueRetrievalResult(
                retrieval_method="exact",
                similarity_threshold=config.similarity_threshold,
                habit_strength_threshold=memory.habit_strength_threshold,
                status="no_match",
            )

    if embedder is None:
        return CueRetrievalResult(
            retrieval_method="semantic_embedding",
            similarity_threshold=config.similarity_threshold,
            habit_strength_threshold=memory.habit_strength_threshold,
            status="embedding_unavailable",
        )
    ensure_memory_embeddings(memory, embedder=embedder, embedding_version=config.embedding_version)
    query_embedding, semantic = retrieve_by_context_embedding(
        memory,
        cue_ids,
        embedder=embedder,
        settings=config,
    )
    return CueRetrievalResult(
        candidates=semantic,
        retrieval_method="semantic_embedding",
        query_embedding=query_embedding,
        similarity_threshold=config.similarity_threshold,
        habit_strength_threshold=memory.habit_strength_threshold,
        status="candidates_found" if semantic else "below_similarity_threshold",
    )


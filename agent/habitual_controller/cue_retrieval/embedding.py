from __future__ import annotations

from math import sqrt
from typing import Protocol

from agent.habitual_controller.cue_memory import CueMemory, CueMemoryRecord
from llm.api_manager import APIManager


_TEXT_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}


class Embedder(Protocol):
    model_name: str

    def embed(self, text: str) -> list[float]: ...


class APIEmbeddingClient:
    def __init__(self, provider_name: str = "ollama", model_name: str = "") -> None:
        self.api = APIManager(
            provider_name=provider_name,
            task_name="agent.habitual_embedding",
        )
        if model_name:
            self.api.provider.embedding_model = model_name
        self.model_name = str(getattr(self.api.provider, "embedding_model", "") or provider_name)

    def embed(self, text: str) -> list[float]:
        return [float(item) for item in self.api.embed(text)]


def canonical_context_text(cue_ids: list[str]) -> str:
    return "\n".join(sorted({str(cue_id).strip() for cue_id in cue_ids if str(cue_id).strip()}))


def ensure_record_embedding(
    record: CueMemoryRecord,
    *,
    embedder: Embedder,
    embedding_version: str,
) -> list[float]:
    if (
        record.context_embedding
        and record.embedding_model == embedder.model_name
        and record.embedding_version == embedding_version
    ):
        return record.context_embedding
    embedding = embedder.embed(canonical_context_text(record.cue_ids))
    record.context_embedding = embedding
    record.embedding_model = embedder.model_name
    record.embedding_dimensions = len(embedding)
    record.embedding_version = embedding_version
    return embedding


def ensure_memory_embeddings(
    memory: CueMemory,
    *,
    embedder: Embedder,
    embedding_version: str,
) -> None:
    for record in memory.established_habits:
        ensure_record_embedding(record, embedder=embedder, embedding_version=embedding_version)


def ensure_record_dimension_embedding(
    record: CueMemoryRecord,
    *,
    dimension: str,
    text: str,
    embedder: Embedder,
    embedding_version: str,
) -> list[float]:
    if record.embedding_model != embedder.model_name or record.embedding_version != embedding_version:
        record.dimension_embeddings.clear()
        record.dimension_embedding_texts.clear()
    if (
        record.dimension_embeddings.get(dimension)
        and record.dimension_embedding_texts.get(dimension) == text
        and record.embedding_model == embedder.model_name
        and record.embedding_version == embedding_version
    ):
        return record.dimension_embeddings[dimension]
    embedding = embed_text_cached(text, embedder=embedder)
    record.dimension_embeddings[dimension] = embedding
    record.dimension_embedding_texts[dimension] = text
    record.embedding_model = embedder.model_name
    record.embedding_dimensions = len(embedding)
    record.embedding_version = embedding_version
    return embedding


def embed_text_cached(text: str, *, embedder: Embedder) -> list[float]:
    key = (embedder.model_name, text)
    if key not in _TEXT_EMBEDDING_CACHE:
        _TEXT_EMBEDDING_CACHE[key] = embedder.embed(text)
    return _TEXT_EMBEDDING_CACHE[key]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

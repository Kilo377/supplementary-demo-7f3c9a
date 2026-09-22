from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CueRetrievalSettings:
    similarity_threshold: float = 0.72
    activation_threshold: float = 0.35
    limit: int = 8
    embedding_version: str = "dimension-text-v1"
    embedding_provider_name: str = "ollama"
    embedding_model: str = "bge-m3:latest"
    same_category_compatibility: float = 0.65
    unrelated_feature_compatibility: float = 0.20
    uncertain_state_compatibility: float = 0.60
    selection_temperature: float = 2.0
    seed: int | None = None
    method: str = "dimension_embedding"

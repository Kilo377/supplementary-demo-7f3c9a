from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import prod
import random

from agent.habitual_controller.cue_memory import CueMemory, CueMemoryRecord
from agent.habitual_controller.cue_extraction import ContextFeature, CueExtractionResult

from .embedding import Embedder, cosine_similarity, embed_text_cached, ensure_record_dimension_embedding
from .settings import CueRetrievalSettings
from .types import CueRetrievalCandidate


@dataclass(frozen=True)
class _Requirement:
    cue_id: str
    dimension: str
    prefix_match: bool = False


def retrieve_by_dimension_embeddings(
    memory: CueMemory,
    context: CueExtractionResult,
    *,
    embedder: Embedder,
    settings: CueRetrievalSettings,
) -> list[CueRetrievalCandidate]:
    """Algorithm B: dimension evidence prepares one response through an S-R race."""
    observed = _features_by_dimension(context)
    candidates = [
        candidate
        for record in memory.established_habits
        if (candidate := _score_record(record, observed, embedder=embedder, settings=settings)) is not None
    ]
    eligible = [
        candidate
        for candidate in candidates
        if candidate.similarity >= settings.activation_threshold
    ]
    if not eligible:
        return []

    response_probabilities = [
        _threshold_excess_probability(candidate.similarity, settings.activation_threshold)
        * candidate.record.habit_strength
        for candidate in eligible
    ]
    rng = random.Random(settings.seed) if settings.seed is not None else random.Random()
    trigger_probability = max(response_probabilities, default=0.0)
    if trigger_probability <= 0.0 or rng.random() >= trigger_probability:
        return []

    selected_index = rng.choices(
        range(len(eligible)),
        weights=_tempered_odds(response_probabilities, settings.selection_temperature),
        k=1,
    )[0]
    selected = eligible[selected_index]
    selected_probability = response_probabilities[selected_index]
    return [
        CueRetrievalCandidate(
            record=selected.record,
            retrieval_method="dimension_embedding_strength",
            matching_cue_ids=selected.matching_cue_ids,
            similarity=selected.similarity,
            retrieval_score=selected_probability,
            familiarity_status="confirmed",
            evidence={
                **selected.evidence,
                "trigger_probability": trigger_probability,
                "response_probability": selected_probability,
                "eligible_count": len(eligible),
            },
        )
    ]


def _score_record(
    record: CueMemoryRecord,
    observed: dict[str, list[ContextFeature]],
    *,
    embedder: Embedder,
    settings: CueRetrievalSettings,
) -> CueRetrievalCandidate | None:
    requirements = _requirements_by_dimension(record)
    if not requirements:
        return None

    dimension_scores: dict[str, float] = {}
    raw_similarities: dict[str, float] = {}
    structured_compatibilities: dict[str, float] = {}
    matching_ids: set[str] = set()
    for dimension, conditions in requirements.items():
        features = observed.get(dimension, [])
        if not features:
            return None
        compatibility_scores: list[float] = []
        for condition in conditions:
            compatibility, matching_feature = _best_condition_match(condition, features, settings)
            compatibility_scores.append(compatibility)
            if compatibility > 0.0 and matching_feature is not None:
                matching_ids.add(matching_feature.feature_id)
        structured = _geometric_mean(compatibility_scores)
        if structured <= 0.0:
            return None

        memory_text = _memory_dimension_text(record, dimension, conditions)
        situation_text = _situation_dimension_text(features)
        memory_embedding = ensure_record_dimension_embedding(
            record,
            dimension=dimension,
            text=memory_text,
            embedder=embedder,
            embedding_version=settings.embedding_version,
        )
        situation_embedding = embed_text_cached(situation_text, embedder=embedder)
        raw_similarity = max(0.0, cosine_similarity(memory_embedding, situation_embedding))
        raw_similarities[dimension] = raw_similarity
        structured_compatibilities[dimension] = structured
        dimension_scores[dimension] = raw_similarity * structured

    cue_similarity = _geometric_mean(list(dimension_scores.values()))
    return CueRetrievalCandidate(
        record=record,
        retrieval_method="dimension_embedding_strength",
        matching_cue_ids=tuple(sorted(matching_ids)),
        similarity=cue_similarity,
        retrieval_score=cue_similarity,
        familiarity_status="confirmed",
        evidence={
            "dimension_scores": dimension_scores,
            "raw_similarities": raw_similarities,
            "structured_compatibilities": structured_compatibilities,
            "habit_strength": record.habit_strength,
        },
    )


def _features_by_dimension(context: CueExtractionResult) -> dict[str, list[ContextFeature]]:
    grouped: dict[str, list[ContextFeature]] = defaultdict(list)
    for feature in context.features:
        if feature.temporal_state == "offset":
            continue
        grouped[_dimension(feature.feature_id)].append(feature)
    return dict(grouped)


def _requirements_by_dimension(record: CueMemoryRecord) -> dict[str, list[_Requirement]]:
    grouped: dict[str, list[_Requirement]] = defaultdict(list)
    for cue_id in record.required_cue_ids:
        grouped[_dimension(cue_id)].append(_Requirement(cue_id, _dimension(cue_id)))
    for prefix in record.cue_prefixes:
        grouped[_dimension(prefix)].append(_Requirement(prefix, _dimension(prefix), prefix_match=True))
    return dict(grouped)


def _condition_compatibility(
    condition: _Requirement,
    feature: ContextFeature,
    settings: CueRetrievalSettings,
) -> float:
    feature_id = feature.feature_id
    if condition.prefix_match and feature_id.startswith(condition.cue_id):
        return 1.0
    if _matches_exact(condition.cue_id, feature_id):
        return 1.0
    if _materially_incompatible_identity(condition.cue_id, feature_id):
        return 0.0
    if _incompatible_state(condition.cue_id, feature_id):
        return 0.0
    if _state_is_unobserved(condition.cue_id, feature_id):
        return settings.uncertain_state_compatibility
    if _feature_category(condition.cue_id) == _feature_category(feature_id):
        return settings.same_category_compatibility
    return settings.unrelated_feature_compatibility


def _materially_incompatible_identity(condition: str, feature_id: str) -> bool:
    hard_identity_prefixes = ("entered:", "location:", "posture:", "wearing:")
    return any(condition.startswith(prefix) and condition != feature_id for prefix in hard_identity_prefixes)


def _best_condition_match(
    condition: _Requirement,
    features: list[ContextFeature],
    settings: CueRetrievalSettings,
) -> tuple[float, ContextFeature | None]:
    expected_state = _visual_state_parts(condition.cue_id)
    if expected_state is not None:
        expected_object, expected_key, expected_value = expected_state
        same_state_key = [
            (feature, _visual_state_parts(feature.feature_id))
            for feature in features
        ]
        same_state_key = [
            (feature, parts)
            for feature, parts in same_state_key
            if parts is not None and parts[0] == expected_object and parts[1] == expected_key
        ]
        if same_state_key:
            exact = next(
                (feature for feature, parts in same_state_key if parts[2] == expected_value),
                None,
            )
            return (1.0, exact) if exact is not None else (0.0, same_state_key[0][0])
        object_feature = next(
            (feature for feature in features if feature.feature_id == f"visible_type:{expected_object}"),
            None,
        )
        if object_feature is not None:
            return settings.uncertain_state_compatibility, object_feature

    scored = [
        (_condition_compatibility(condition, feature, settings), feature)
        for feature in features
    ]
    return max(scored, key=lambda item: item[0]) if scored else (0.0, None)


def _matches_exact(condition: str, feature_id: str) -> bool:
    if condition == feature_id:
        return True
    if condition.endswith(":contains:water"):
        base = condition.removesuffix(":water")
        return feature_id in {condition, f"{base}:half_water", f"{base}:full_water"}
    return False


def _incompatible_state(condition: str, feature_id: str) -> bool:
    expected = _visual_state_parts(condition)
    actual = _visual_state_parts(feature_id)
    if expected is None or actual is None:
        return False
    expected_object, expected_key, expected_value = expected
    actual_object, actual_key, actual_value = actual
    return (
        expected_object == actual_object
        and expected_key == actual_key
        and expected_value != actual_value
    )


def _state_is_unobserved(condition: str, feature_id: str) -> bool:
    expected = _visual_state_parts(condition)
    if expected is None:
        return False
    expected_object, _, _ = expected
    return feature_id == f"visible_type:{expected_object}"


def _visual_state_parts(cue_id: str) -> tuple[str, str, str] | None:
    prefix = "visible_type_state:"
    if not cue_id.startswith(prefix):
        return None
    parts = cue_id[len(prefix):].split(":", 2)
    return tuple(parts) if len(parts) == 3 else None


def _dimension(cue_id: str) -> str:
    if cue_id.startswith(("visible:", "visible_type:", "visible_state:", "visible_type_state:")):
        return "visual_object"
    if cue_id.startswith(("entered:", "completed:", "failed:")):
        return "sequence_event"
    if cue_id.startswith("time_period:"):
        return "temporal"
    if cue_id.startswith("location:"):
        return "spatial"
    if cue_id.startswith("feeling:mental:"):
        return "affective"
    if cue_id.startswith("feeling:"):
        return "internal_body"
    if cue_id.startswith(("posture:", "wearing:")):
        return "body_resource"
    return "other"


def _feature_category(cue_id: str) -> str:
    if cue_id.startswith("visible_type_state:"):
        parts = cue_id.split(":", 3)
        return f"visual_state:{parts[1]}:{parts[2]}" if len(parts) > 2 else "visual_state"
    if cue_id.startswith("visible_type:"):
        semantic_type = cue_id.split(":", 1)[1]
        return _visual_semantic_category(semantic_type)
    if cue_id.startswith("visible_state:"):
        return "visual_state"
    if cue_id.startswith("visible:"):
        return "visual_instance"
    if cue_id.startswith("location:"):
        return "spatial_location"
    if cue_id.startswith("time_period:"):
        return "time_period"
    if cue_id.startswith("entered:"):
        return "room_entry"
    if cue_id.startswith("completed:"):
        return "completed_action"
    if cue_id.startswith("failed:"):
        return "failed_action"
    if cue_id.startswith("feeling:mental:"):
        return "negative_arousal"
    if cue_id.startswith("feeling:"):
        return "body_feeling"
    return cue_id.split(":", 1)[0]


def _visual_semantic_category(semantic_type: str) -> str:
    categories = {
        "cup": "drinking_vessel",
        "water_container": "drinking_vessel",
        "remote_control": "household_controller",
        "mirror": "reflection_surface",
        "coffee_machine": "kitchen_appliance",
    }
    return categories.get(semantic_type, f"visual_object:{semantic_type}")


def _memory_dimension_text(
    record: CueMemoryRecord,
    dimension: str,
    conditions: list[_Requirement],
) -> str:
    configured = (record.metadata.get("dimension_texts", {}) or {}).get(dimension)
    if configured:
        return str(configured)
    cue_text = "；".join(_describe_cue_id(item.cue_id) for item in conditions)
    return " ".join(part for part in (record.context_text, cue_text) if part).strip()


def _situation_dimension_text(features: list[ContextFeature]) -> str:
    texts = []
    for feature in features:
        text = str(feature.text or feature.value or _describe_cue_id(feature.feature_id)).strip()
        if text and text not in texts:
            texts.append(text)
    return "；".join(texts)


def _describe_cue_id(cue_id: str) -> str:
    replacements = {
        "visible_type": "看见",
        "visible_type_state": "看见物体状态",
        "location": "位于",
        "time_period": "时间是",
        "entered": "刚进入",
        "completed": "刚完成",
        "failed": "刚才失败",
        "feeling": "感觉",
        "contains": "内容物",
        "water": "水",
        "empty": "空",
        "mild": "轻微",
        "moderate": "中等",
        "strong": "明显",
        "anxiety": "焦虑",
        "stress": "压力",
        "frustration": "沮丧",
        "fatigue": "疲劳",
    }
    return " ".join(replacements.get(part, part.replace("_", " ")) for part in cue_id.rstrip(":").split(":"))


def _threshold_excess_probability(score: float, threshold: float) -> float:
    threshold = min(1.0, max(0.0, threshold))
    if threshold >= 1.0:
        return 1.0 if score >= 1.0 else 0.0
    return min(1.0, max(0.0, (score - threshold) / (1.0 - threshold)))


def _probability_odds(probability: float) -> float:
    if probability >= 1.0:
        return 1e12
    if probability <= 0.0:
        return 0.0
    return probability / (1.0 - probability)


def _tempered_odds(probabilities: list[float], temperature: float) -> list[float]:
    bounded_temperature = max(1e-6, float(temperature))
    return [
        _probability_odds(probability) ** (1.0 / bounded_temperature)
        for probability in probabilities
    ]


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    bounded = [min(1.0, max(0.0, float(value))) for value in values]
    return prod(bounded) ** (1.0 / len(bounded))

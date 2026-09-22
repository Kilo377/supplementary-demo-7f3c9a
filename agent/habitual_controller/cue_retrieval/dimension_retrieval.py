from __future__ import annotations

from collections import defaultdict
from math import prod

from agent.habitual_controller.cue_memory import CueMemory
from agent.habitual_controller.cue_extraction import CueExtractionResult

from .types import CueRetrievalCandidate


def retrieve_by_structured_dimensions(
    memory: CueMemory,
    context: CueExtractionResult,
    *,
    activation_threshold: float,
    limit: int,
) -> list[CueRetrievalCandidate]:
    feature_ids = set(context.structured_feature_ids)
    candidates: list[CueRetrievalCandidate] = []
    for record in memory.established_habits:
        requirements = _requirements(record.required_cue_ids, record.cue_prefixes)
        if not requirements:
            continue
        dimension_scores: list[float] = []
        matching: set[str] = set()
        for conditions in requirements.values():
            condition_scores = []
            for condition, prefix_match in conditions:
                matches = {
                    feature_id
                    for feature_id in feature_ids
                    if _matches(condition, feature_id, prefix_match=prefix_match)
                }
                condition_scores.append(1.0 if matches else 0.0)
                matching.update(matches)
            dimension_scores.append(sum(condition_scores) / len(condition_scores))
        cue_similarity = _geometric_mean(dimension_scores)
        retrieval_score = cue_similarity * record.habit_strength
        if retrieval_score < activation_threshold:
            continue
        candidates.append(
            CueRetrievalCandidate(
                record=record,
                retrieval_method="structured_dimension",
                matching_cue_ids=tuple(sorted(matching)),
                similarity=cue_similarity,
                retrieval_score=retrieval_score,
                familiarity_status="not_required",
            )
        )
    candidates.sort(
        key=lambda item: (item.retrieval_score, item.similarity, item.record.habit_strength),
        reverse=True,
    )
    return candidates[:limit]


def _requirements(required_ids: list[str], prefixes: list[str]) -> dict[str, list[tuple[str, bool]]]:
    grouped: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for cue_id in required_ids:
        grouped[_dimension(cue_id)].append((cue_id, False))
    for prefix in prefixes:
        grouped[_dimension(prefix)].append((prefix, True))
    return dict(grouped)


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


def _matches(condition: str, feature_id: str, *, prefix_match: bool) -> bool:
    if prefix_match:
        return feature_id.startswith(condition)
    if condition == feature_id:
        return True
    if condition.endswith(":contains:water"):
        base = condition.removesuffix(":water")
        return feature_id in {condition, f"{base}:half_water", f"{base}:full_water"}
    return False


def _geometric_mean(values: list[float]) -> float:
    return prod(values) ** (1.0 / len(values)) if values else 0.0

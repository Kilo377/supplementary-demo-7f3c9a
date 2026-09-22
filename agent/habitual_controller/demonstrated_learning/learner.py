from __future__ import annotations

from datetime import datetime
import hashlib
import re
from uuid import uuid4

from agent.goal_directed_controller_wm.state_transition.json_output import parse_json_object
from agent.habitual_controller.cue_memory import CueMemory, CueMemoryRecord
from llm.api_manager import APIManager

from .prompt import build_demonstrated_habit_learning_prompt
from .types import (
    DemonstratedActionObservation,
    DemonstratedHabitLearningResult,
    DemonstratedHabitRejection,
)


def learn_demonstrated_habits(
    *,
    agent_name: str,
    intent_text: str,
    observations: list[DemonstratedActionObservation],
    memory: CueMemory,
    persona_memory: CueMemory | None = None,
    initial_habit_strength: float = 0.90,
    provider_name: str = "ollama",
    model: str | None = None,
    run_id: str = "",
) -> DemonstratedHabitLearningResult:
    if not observations:
        return DemonstratedHabitLearningResult()
    prompt = build_demonstrated_habit_learning_prompt(
        agent_name=agent_name,
        intent_text=intent_text,
        observations=observations,
        existing_persona_habits=(
            persona_memory.established_habits
            if persona_memory is not None
            else []
        ),
        existing_demonstrated_habits=memory.established_habits,
    )
    result = DemonstratedHabitLearningResult(prompt=prompt)
    try:
        api = APIManager(
            provider_name=provider_name,
            task_name="agent.demonstrated_habit_learning",
        )
        raw = api.generate(prompt, model=model)
        result.raw_response = raw
        parsed = parse_json_object(raw)
    except Exception as error:
        result.error = f"Demonstrated habit learning failed: {error}"
        return result

    observations_by_id = {
        observation.observation_id: observation
        for observation in observations
    }
    judgments = parsed.get("judgments", [])
    if not isinstance(judgments, list):
        judgments = []
    judged_ids = set()
    strength = min(1.0, max(memory.habit_strength_threshold, float(initial_habit_strength)))
    current_run_id = run_id or uuid4().hex[:12]
    for judgment in judgments:
        if not isinstance(judgment, dict):
            continue
        observation_id = str(judgment.get("observation_id", "") or "").strip()
        observation = observations_by_id.get(observation_id)
        if observation is None or observation_id in judged_ids:
            continue
        judged_ids.add(observation_id)
        reason = str(judgment.get("reason", "") or "").strip()
        if not _bool_value(judgment.get("should_learn")):
            result.rejections.append(
                DemonstratedHabitRejection(observation_id, observation.action_text, reason)
            )
            continue
        selected_cues = _valid_selected_cues(judgment, observation)
        action_signature = _canonical_action_signature(
            judgment.get("canonical_action_signature")
        )
        if not selected_cues or not action_signature:
            result.rejections.append(
                DemonstratedHabitRejection(
                    observation_id,
                    observation.action_text,
                    "LLM没有返回可用的真实 Cue 或规范动作签名。",
                )
            )
            continue
        canonical_key = _association_key(selected_cues, action_signature)
        source_run_id = observation.source_run_id or current_run_id
        habitual_action = str(judgment.get("habitual_action", "") or "").strip()
        record = CueMemoryRecord(
            association_id=f"demonstrated_{canonical_key}",
            cue_ids=list(selected_cues),
            required_cue_ids=list(selected_cues),
            cue_prefixes=[],
            context_text=str(judgment.get("context_text", "") or "").strip(),
            response_key=action_signature,
            response_text=habitual_action or observation.action_text,
            habit_strength=strength,
            source_dimension="demonstrated_history",
            awareness_type="conscious",
            estimated_duration=observation.estimated_duration,
            cooldown_seconds=0,
            reinforcement_count=1,
            last_reinforced_at=datetime.now(),
            metadata={
                "memory_pool": "demonstrated",
                "canonical_association_key": canonical_key,
                "canonical_action_signature": action_signature,
                "learning_reason": reason,
                "source_intent": intent_text,
                "source_step_id": observation.step_id,
                "source_runs": [source_run_id],
            },
        )
        _reuse_existing_task_association_key(
            memory,
            record,
            intent_text=intent_text,
        )
        stored, created = memory.upsert_established(record)
        result.records.append(stored)
        if created:
            result.created_count += 1
        else:
            result.reinforced_count += 1

    for observation in observations:
        if observation.observation_id not in judged_ids:
            result.rejections.append(
                DemonstratedHabitRejection(
                    observation.observation_id,
                    observation.action_text,
                    "LLM没有返回这个动作的判断。",
                )
            )
    return result


def _valid_selected_cues(
    judgment: dict,
    observation: DemonstratedActionObservation,
) -> tuple[str, ...]:
    selected = judgment.get("selected_cue_ids", [])
    if not isinstance(selected, list):
        return ()
    available = observation.available_cue_ids
    selected_cues = tuple(dict.fromkeys(
        str(item).strip()
        for item in selected
        if str(item).strip() in available
    ))
    if observation.source_run_id:
        location_cue = next(
            (
                feature.feature_id
                for feature in observation.cue_features
                if feature.feature_type == "location"
                or feature.feature_id.startswith("location:")
            ),
            "",
        )
        if location_cue and location_cue not in selected_cues:
            selected_cues = (location_cue, *selected_cues)
    return selected_cues[:3]


def _canonical_action_signature(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _association_key(cue_ids: tuple[str, ...], action_signature: str) -> str:
    payload = f"{'|'.join(sorted(cue_ids))}=>{action_signature}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _reuse_existing_task_association_key(
    memory: CueMemory,
    record: CueMemoryRecord,
    *,
    intent_text: str,
) -> None:
    for existing in memory.records:
        if existing.response_key != record.response_key:
            continue
        if str(existing.metadata.get("source_intent", "") or "") != intent_text:
            continue
        record.metadata["canonical_association_key"] = (
            existing.canonical_association_key()
        )
        return


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes"}

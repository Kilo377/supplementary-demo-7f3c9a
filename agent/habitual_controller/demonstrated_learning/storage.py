from __future__ import annotations

import json
from pathlib import Path

from agent.habitual_controller.cue_memory import CueMemory, CueMemoryRecord


def load_demonstrated_habit_memory(
    path: str | Path,
    *,
    agent_name: str,
    habit_strength_threshold: float,
) -> CueMemory:
    source = Path(path)
    memory = CueMemory(habit_strength_threshold=habit_strength_threshold)
    if not source.exists():
        return memory
    payload = json.loads(source.read_text(encoding="utf-8"))
    stored_name = str(payload.get("agent_name", "") or "").strip()
    if stored_name and stored_name != agent_name:
        raise ValueError(
            f"Demonstrated habit memory belongs to {stored_name}, not {agent_name}."
        )
    for item in list(payload.get("records", []) or []):
        if isinstance(item, dict):
            memory.upsert_established(
                CueMemoryRecord.from_dict(item, agent_name=agent_name)
            )
    return memory


def save_demonstrated_habit_memory(
    path: str | Path,
    *,
    agent_name: str,
    intent_text: str,
    memory: CueMemory,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "demonstrated-habit-memory.v1",
        "agent_name": agent_name,
        "last_intent": intent_text,
        "habit_strength_threshold": memory.habit_strength_threshold,
        "records": [record.to_dict() for record in memory.records],
    }
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


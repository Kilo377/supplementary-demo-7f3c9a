from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


DEFAULT_BEHAVIOR_LOG = str(Path(__file__).resolve().parents[2] / "behavior_log.json")


def build_behavior_log(results: Iterable[object]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for result in results:
        behavior = str(
            getattr(result, "execution_narration", "")
            or getattr(result, "action_text", "")
            or ""
        ).strip()
        if not behavior:
            continue
        records.append(
            {
                "time": _result_time(result),
                "behavior": behavior,
            }
        )
    return records


def write_behavior_log(
    results: Iterable[object],
    *,
    output_path: str | Path = DEFAULT_BEHAVIOR_LOG,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_behavior_log(results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _result_time(result: object) -> str:
    clock_time = str(getattr(result, "clock_time", "") or "").strip()
    if clock_time:
        return clock_time
    time_text = str(getattr(result, "time_text", "") or "").strip()
    if time_text:
        return time_text
    return ""

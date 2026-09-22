from __future__ import annotations

import atexit
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


GENERIC_WORLD_FAILURE_MARKERS = (
    "这一步没有得到稳定的环境反馈，动作暂时没有落实",
    "这一步没有得到稳定的环境支持判断，动作暂时没有落实",
)

SYSTEM_FAILURE_ROUTES = {
    "contextual_world_error",
    "world_transition_check",
    "world_node_support",
}

DEFAULT_FAILURE_OUTPUT = Path(__file__).resolve().parent / "world_failure_cases.json"


class WorldFailureSession:
    """Persist contextual-world failures for any runtime entry point."""

    def __init__(
        self,
        *,
        scene: str,
        provider: str,
        model: str,
        output_path: str | Path = DEFAULT_FAILURE_OUTPUT,
    ) -> None:
        self.started_at = datetime.now().astimezone()
        self.scene = scene
        self.provider = provider
        self.model = model
        self.output_path = _timestamped_output_path(
            Path(output_path),
            ended_at=self.started_at,
            include_microseconds=True,
        )
        self.world_action_attempts = 0
        self.failures: list[dict] = []
        self._lock = Lock()
        self._closed = False
        atexit.register(self.close)

    def record_result(self, result: Any) -> Path | None:
        with self._lock:
            if self._closed:
                return None
            self.world_action_attempts += 1
            record = _failure_record(result, turn_index=self.world_action_attempts)
            if record is None:
                return None
            return self._append_failure(record)

    def observe_action_attempt(self) -> None:
        with self._lock:
            if not self._closed:
                self.world_action_attempts += 1

    def record_failure(self, record: dict) -> Path | None:
        with self._lock:
            if self._closed:
                return None
            return self._append_failure(dict(record))

    def _append_failure(self, record: dict) -> Path:
        record.setdefault("failure_index", len(self.failures) + 1)
        record.setdefault("world_action_attempt", self.world_action_attempts or None)
        record["recorded_at"] = datetime.now().astimezone().isoformat()
        self.failures.append(record)
        self._write(ended_at=datetime.now().astimezone())
        return self.output_path

    def close(self) -> Path | None:
        with self._lock:
            if self._closed:
                return self.output_path if self.failures else None
            self._closed = True
            if not self.failures:
                return None
            self._write(ended_at=datetime.now().astimezone())
            return self.output_path

    def _write(self, *, ended_at: datetime) -> None:
        failure_counts_by_module: dict[str, int] = {}
        for failure in self.failures:
            module = str(failure.get("failed_module", "") or "unknown")
            failure_counts_by_module[module] = failure_counts_by_module.get(module, 0) + 1
        payload = {
            "schema_version": "world_failure_cases.v3",
            "run_started_at": self.started_at.isoformat(),
            "run_ended_at": ended_at.isoformat(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scene": self.scene,
            "provider": self.provider,
            "model": self.model,
            "world_action_attempts": self.world_action_attempts,
            "failure_count": len(self.failures),
            "failure_counts_by_module": failure_counts_by_module,
            "failures": self.failures,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(f"{self.output_path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.output_path)


def write_world_failure_export(
    results: list[Any],
    *,
    output_path: str | Path,
    scene: str,
    provider: str,
    model: str,
    run_error: str = "",
) -> tuple[Path, int]:
    run_ended_at = datetime.now().astimezone()
    failures = []
    for turn_index, result in enumerate(results, start=1):
        record = _failure_record(result, turn_index=turn_index)
        if record is not None:
            failures.append(record)
    failure_counts_by_module: dict[str, int] = {}
    for failure in failures:
        module = str(failure.get("failed_module", "") or "unknown")
        failure_counts_by_module[module] = failure_counts_by_module.get(module, 0) + 1

    payload = {
        "schema_version": "world_failure_cases.v1",
        "run_ended_at": run_ended_at.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene": scene,
        "provider": provider,
        "model": model,
        "total_action_turns": len(results),
        "failure_count": len(failures),
        "failure_counts_by_module": failure_counts_by_module,
        "run_error": run_error,
        "failures": failures,
    }
    output = _timestamped_output_path(Path(output_path), ended_at=run_ended_at)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, len(failures)


def _failure_record(result: Any, *, turn_index: int) -> dict | None:
    feedback = _dict(getattr(result, "environment_feedback", {}))
    report = _dict(feedback.get("graph_transition_report"))
    world_event = _dict(report.get("world_action_event"))
    execution_debug = _dict(getattr(result, "execution_debug", {}))
    route = str(feedback.get("route", "") or execution_debug.get("route", "") or "").strip()
    feedback_text = str(
        feedback.get("perception_summary", "")
        or world_event.get("actual_event", "")
        or getattr(result, "execution_narration", "")
        or getattr(result, "action_text", "")
        or ""
    ).strip()
    execution_kind = str(getattr(result, "execution_kind", "") or "")
    has_generic_feedback = any(marker in feedback_text for marker in GENERIC_WORLD_FAILURE_MARKERS)
    has_system_failure = (
        has_generic_feedback
        or route in SYSTEM_FAILURE_ROUTES
        or execution_kind == "action_error"
        or bool(execution_debug.get("failed_module"))
        or bool(execution_debug.get("error"))
    )
    if not has_system_failure:
        return None

    rejection = _dict(world_event.get("rejection"))
    warnings = [str(item) for item in (report.get("warnings") or [])]
    failed_module = str(execution_debug.get("failed_module", "") or route or "unknown")
    error = str(execution_debug.get("error", "") or rejection.get("reason", "") or "").strip()
    if not error and warnings:
        error = "; ".join(warnings)

    return {
        "turn_index": turn_index,
        "engine_step_id": getattr(result, "step_id", None),
        "failure_kind": "pipeline_error",
        "failed_module": failed_module,
        "error": error,
        "world_route": route,
        "execution_kind": execution_kind,
        "action_proposal": str(getattr(result, "action_proposal_text", "") or ""),
        "intent": {
            "text": str(getattr(result, "intent_text", "") or ""),
            "status": str(getattr(result, "intent_status", "") or ""),
            "action_index": getattr(result, "intent_action_index", None),
        },
        "cognition": {
            "intuition_route": str(getattr(result, "intuition_route", "") or ""),
            "intuition_thought": str(getattr(result, "intuition_thought", "") or ""),
            "think_thought": str(getattr(result, "think_thought", "") or ""),
            "think_conclusion": str(getattr(result, "think_conclusion", "") or ""),
        },
        "target_resolution": {
            "action_type": str(getattr(result, "action_type", "") or ""),
            "decision_reason": str(getattr(result, "decision_text", "") or ""),
            "target_area_id": getattr(result, "target_area_id", None),
            "target_element_id": getattr(result, "target_element_id", None),
            "navigation_target_element_id": getattr(result, "navigation_target_element_id", None),
            "secondary_target_element_id": getattr(result, "secondary_target_element_id", None),
        },
        "world_feedback": feedback_text,
        "world_judgments": {
            "node_support": _dict(execution_debug.get("support_result")),
            "interaction_focus": _dict(execution_debug.get("focus_result")),
            "state_transition": _dict(execution_debug.get("world_state_transition")),
            "transition_check": _dict(execution_debug.get("transition_check")),
            "world_action_event": world_event,
            "warnings": warnings,
        },
        "contextual_world_trace": list(execution_debug.get("contextual_world_trace") or []),
        "graph_transition_report": report,
        "actor_state": _dict(getattr(result, "actor_state", {})),
        "final_position": list(getattr(result, "final_position", ()) or ()),
    }


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _timestamped_output_path(
    base_path: Path,
    *,
    ended_at: datetime,
    include_microseconds: bool = False,
) -> Path:
    expanded = base_path.expanduser()
    suffix = expanded.suffix or ".json"
    stem = expanded.stem if expanded.suffix else expanded.name
    timestamp_format = "%Y%m%d_%H%M%S_%f" if include_microseconds else "%Y%m%d_%H%M%S"
    timestamp = ended_at.strftime(timestamp_format)
    candidate = expanded.with_name(f"{stem}_{timestamp}{suffix}").resolve()
    index = 1
    while candidate.exists():
        candidate = expanded.with_name(f"{stem}_{timestamp}_{index:02d}{suffix}").resolve()
        index += 1
    return candidate

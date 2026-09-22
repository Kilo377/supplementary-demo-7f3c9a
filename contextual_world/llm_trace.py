from __future__ import annotations

import traceback
from datetime import datetime
from time import perf_counter
from typing import Callable, TypeVar

from llm.api_manager import APIManager

from contextual_world.types import ContextualWorldModuleError, ContextualWorldTraceStep


ResultT = TypeVar("ResultT")


def run_traced_llm_module(
    *,
    module_name: str,
    variables: dict,
    prompt: str,
    provider_name: str,
    model: str | None,
    parse_result: Callable[[str], ResultT],
) -> tuple[ResultT, ContextualWorldTraceStep]:
    started_at = datetime.now().astimezone().isoformat()
    started = perf_counter()
    raw_output = ""
    phase = "llm_call"
    attempts: list[dict] = []
    attempt_started = perf_counter()
    llm = APIManager(
        provider_name=provider_name,
        task_name=f"world.{module_name.removeprefix('world_')}",
    )
    try:
        raw_output = llm.generate(prompt, model=model)
        attempts = _llm_attempts_for_trace(llm.call_attempts, raw_output=raw_output)
        phase = "parse_output"
        result = parse_result(raw_output)
    except Exception as error:
        error_traceback = traceback.format_exc()
        attempts = _llm_attempts_for_trace(llm.call_attempts, raw_output=raw_output)
        if phase == "parse_output" or not attempts:
            attempts.append({
                "attempt": len(attempts) + 1,
                "phase": phase,
                "duration_seconds": round(perf_counter() - attempt_started, 6),
                "raw_output": raw_output,
                "error_type": type(error).__name__,
                "error": str(error),
            })
        trace_step = ContextualWorldTraceStep(
            module_name=module_name,
            variables=variables,
            prompt=prompt,
            raw_output=raw_output,
            error=error_traceback,
            provider_name=llm.provider_name,
            model=llm.route.model or "provider default",
            started_at=started_at,
            duration_seconds=round(perf_counter() - started, 6),
            phase=phase,
            error_type=type(error).__name__,
            attempts=attempts,
        )
        raise ContextualWorldModuleError(str(error), trace_step=trace_step) from error

    parsed_output = result.to_dict() if hasattr(result, "to_dict") else {}
    return result, ContextualWorldTraceStep(
        module_name=module_name,
        variables=variables,
        prompt=prompt,
        raw_output=raw_output,
        parsed_output=parsed_output,
        provider_name=llm.provider_name,
        model=llm.route.model or "provider default",
        started_at=started_at,
        duration_seconds=round(perf_counter() - started, 6),
        phase="complete",
        attempts=attempts,
    )


def _llm_attempts_for_trace(
    call_attempts: list[dict],
    *,
    raw_output: str,
) -> list[dict]:
    result: list[dict] = []
    for index, call in enumerate(call_attempts, start=1):
        succeeded = call.get("status") == "success"
        result.append({
            "attempt": index,
            "phase": "complete" if succeeded else "llm_call",
            "provider_name": str(call.get("provider_name", "") or ""),
            "model": str(call.get("model", "") or ""),
            "duration_seconds": call.get("duration_seconds"),
            "raw_output": raw_output if succeeded else "",
            "error_type": str(call.get("error_type", "") or ""),
            "error": str(call.get("error", "") or ""),
        })
    return result

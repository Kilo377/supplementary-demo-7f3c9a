from __future__ import annotations

"""JSON parsing helpers for model-generated state transitions."""

import json
import re


def parse_json_object(text: str) -> dict:
    payload = extract_json_text(text)
    last_error: Exception | None = None
    for candidate in (payload, _repair_common_json_quoting(payload)):
        try:
            parsed = json.loads(candidate, strict=False)
        except (json.JSONDecodeError, TypeError) as error:
            last_error = error
            continue
        if not isinstance(parsed, dict):
            raise ValueError("Model output is not a JSON object.")
        return parsed
    raise ValueError(f"Failed to parse model JSON output: {last_error}")


def extract_json_text(text: str) -> str:
    stripped = str(text or "").strip().lstrip("\ufeff")
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    if "```json" in stripped:
        after = stripped.split("```json", 1)[1]
        return after.split("```", 1)[0].strip()
    if "```" in stripped:
        after = stripped.split("```", 1)[1]
        return after.split("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    raise ValueError("No JSON object found in model response.")


def _repair_common_json_quoting(text: str) -> str:
    repaired = re.sub(r"([:\[,]\s*)“", r'\1"', text)
    repaired = re.sub(r"”(?=\s*[,}\]])", '"', repaired)
    return repaired

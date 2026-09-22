from __future__ import annotations


WORLD_MODEL_MODES = ("simple", "complex")
DEFAULT_WORLD_MODEL_MODE = "complex"


def normalize_world_model_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in WORLD_MODEL_MODES:
        return mode
    return DEFAULT_WORLD_MODEL_MODE

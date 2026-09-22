from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import sleep

from llm.api_manager import APIManager
from contextual_world.world_old.prompts.element_support_prompt import build_agent_environment_element_support_prompt


@dataclass
class ElementSupportResult:
    route: str
    support_kind: str
    reason: str
    permanent_targets: list[dict] = field(default_factory=list)
    temporary_targets: list[dict] = field(default_factory=list)
    temporary_element_creations: list[dict] = field(default_factory=list)
    temporary_element_updates: list[dict] = field(default_factory=list)
    context_facts: list[str] = field(default_factory=list)
    raw_response: str = ""


class ElementSupportError(Exception):
    def __init__(self, message: str, raw_response: str = "", prompt: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.prompt = prompt


def judge_agent_environment_element_support(
    *,
    agent_name: str,
    action_proposal: str,
    agent_state_for_support: dict,
    current_agent_facts: list[dict],
    permanent_elements: list[dict],
    temporary_elements: list[dict],
    provider_name: str = "ollama",
    model: str | None = None,
    print_output: bool = True,
    max_retries: int = 1,
) -> ElementSupportResult:
    prompt = build_agent_environment_element_support_prompt(
        agent_name=agent_name,
        action_proposal=action_proposal,
        agent_state_for_support=agent_state_for_support,
        current_agent_facts=current_agent_facts,
        permanent_elements=permanent_elements,
        temporary_elements=temporary_elements,
    )
    api = APIManager(provider_name=provider_name)
    raw = ""
    last_error: Exception | None = None
    for attempt in range(max(0, max_retries) + 1):
        try:
            raw = api.generate(prompt, model=model)
            last_error = None
            break
        except Exception as error:
            last_error = error
            if attempt >= max(0, max_retries):
                break
            sleep(0.5)
    if last_error is not None:
        raise ElementSupportError(
            f"Element support LLM call failed: {last_error}",
            raw_response=raw,
            prompt=prompt,
        ) from last_error
    check_element_support_output(raw, enabled=print_output)
    try:
        parsed = json.loads(_extract_json_text(raw))
    except Exception as error:
        raise ElementSupportError(
            "Failed to parse element support LLM output.",
            raw_response=raw,
            prompt=prompt,
        ) from error
    return ElementSupportResult(
        route=_normalize_route(parsed.get("route")),
        support_kind=_normalize_support_kind(parsed.get("support_kind")),
        reason=str(parsed.get("reason", "") or ""),
        permanent_targets=_dict_list(parsed.get("permanent_targets")),
        temporary_targets=_dict_list(parsed.get("temporary_targets")),
        temporary_element_creations=_dict_list(parsed.get("temporary_element_creations")),
        temporary_element_updates=_dict_list(parsed.get("temporary_element_updates")),
        context_facts=[
            str(item).strip()
            for item in (parsed.get("context_facts", []) or [])
            if str(item).strip()
        ],
        raw_response=raw,
    )


def check_element_support_output(raw_output: str, *, enabled: bool = True) -> None:
    # TODO: Replace this with a meta-check that evaluates whether the support
    # judgment is internally consistent before graph transition planning uses it.
    if not enabled:
        return
    print("Element support output:")
    print(raw_output)


def _normalize_route(value) -> str:
    route = str(value or "").strip()
    if route in {"agent_body_action", "element_interaction", "reject"}:
        return route
    return "reject"


def _normalize_support_kind(value) -> str:
    support_kind = str(value or "").strip()
    allowed = {
        "no_element",
        "permanent_element",
        "existing_temporary_element",
        "new_temporary_element",
        "mixed",
        "reject",
    }
    if support_kind in allowed:
        return support_kind
    return "reject"


def _dict_list(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
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
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    raise ValueError("No JSON object found in element support response.")

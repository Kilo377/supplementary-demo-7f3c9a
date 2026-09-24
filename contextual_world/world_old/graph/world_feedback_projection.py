from __future__ import annotations

from core.feedback_types import EnvironmentFeedback
from llm.api_manager import APIManager
from contextual_world.world_old.prompts.world_feedback_summary_prompt import build_world_feedback_summary_prompt

from .world_graph import WorldGraph


def build_feedback_current_state(
    graph: WorldGraph,
    *,
    agent_node_id: str = "agent_01",
    transition_report: dict | None = None,
) -> dict:
    relevant_ids = {agent_node_id}
    transition_report = transition_report or {}
    for edge in graph.edges_for_node(agent_node_id, edge_kind="fact"):
        relevant_ids.add(edge.from_node_id)
        relevant_ids.add(edge.to_node_id)
    for key in ("created_temporary_nodes", "updated_temporary_nodes", "updated_element_nodes"):
        for item in transition_report.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id", "") or "").strip()
            if node_id:
                relevant_ids.add(node_id)
            for state_key in ("state", "new_state", "old_state"):
                state = item.get(state_key)
                if not isinstance(state, dict):
                    continue
                for ref_key in ("anchor_element_id", "anchor_area_id"):
                    ref_id = str(state.get(ref_key, "") or "").strip()
                    if ref_id:
                        relevant_ids.add(ref_id)
    for key in ("added_fact_edges", "removed_fact_edges"):
        for edge in transition_report.get(key, []) or []:
            if not isinstance(edge, dict):
                continue
            from_id = str(edge.get("from_node_id", "") or edge.get("subject_id", "") or "").strip()
            to_id = str(edge.get("to_node_id", "") or edge.get("object_id", "") or "").strip()
            if from_id:
                relevant_ids.add(from_id)
            if to_id:
                relevant_ids.add(to_id)

    relevant_nodes = [
        graph.nodes[node_id].to_dict()
        for node_id in sorted(relevant_ids)
        if node_id in graph.nodes
    ]
    current_fact_edges = [
        edge.to_dict()
        for edge in graph.edges.values()
        if edge.edge_kind == "fact"
        and (edge.from_node_id in relevant_ids or edge.to_node_id in relevant_ids)
    ]
    relevant_affiliation_edges = [
        edge.to_dict()
        for edge in graph.edges.values()
        if edge.edge_kind == "affiliation"
        and (edge.from_node_id in relevant_ids or edge.to_node_id in relevant_ids)
    ]
    return {
        "agent_node": graph.nodes.get(agent_node_id).to_dict() if agent_node_id in graph.nodes else {},
        "relevant_nodes": relevant_nodes,
        "current_fact_edges": current_fact_edges,
        "relevant_affiliation_edges": relevant_affiliation_edges,
        "recent_transition_report": transition_report,
    }


def build_environment_feedback(
    *,
    current_state: dict,
    route: str = "",
    agent_name: str = "agent",
    agent_state_patch: dict | None = None,
    graph_transition_report: dict | None = None,
    world_state_diff: dict | None = None,
) -> EnvironmentFeedback:
    report = graph_transition_report or current_state.get("recent_transition_report", {}) or {}
    agent_state = _extract_agent_state(current_state)
    environment_changes = _environment_changes_from_report(report)
    current_elements = _current_elements_from_state(current_state)
    return EnvironmentFeedback(
        route=route,
        agent_state=agent_state,
        environment_changes=environment_changes,
        current_elements=current_elements,
        perception_summary=_compose_perception_summary(
            agent_name=agent_name,
            agent_state=agent_state,
            environment_changes=environment_changes,
            current_elements=current_elements,
        ),
        agent_state_patch=dict(agent_state_patch or {}),
        graph_transition_report=dict(report),
        world_state_diff=dict(world_state_diff or {}),
    )


def generate_world_feedback(
    *,
    agent_name: str,
    action_proposal: str,
    current_state: dict,
    route: str = "",
    agent_state_patch: dict | None = None,
    graph_transition_report: dict | None = None,
    world_state_diff: dict | None = None,
    provider_name: str = "ollama",
    model: str | None = None,
    print_output: bool = True,
    use_llm_summary: bool = False,
) -> EnvironmentFeedback:
    feedback = build_environment_feedback(
        current_state=current_state,
        route=route,
        agent_name=agent_name,
        agent_state_patch=agent_state_patch,
        graph_transition_report=graph_transition_report,
        world_state_diff=world_state_diff,
    )
    if use_llm_summary:
        try:
            feedback.perception_summary = generate_world_feedback_summary(
                agent_name=agent_name,
                action_proposal=action_proposal,
                route=route,
                world_state_diff=feedback.world_state_diff,
                provider_name=provider_name,
                model=model,
                print_output=print_output,
            )
        except Exception as error:
            feedback.graph_transition_report.setdefault("warnings", []).append(
                f"world feedback summary failed: {error}"
            )
    if print_output:
        print("Environment feedback projection:")
        print(feedback.to_dict())
    return feedback


def build_world_feedback_prompt(
    *,
    agent_name: str,
    action_proposal: str,
    current_state: dict,
) -> str:
    fallback = build_environment_feedback(
        current_state=current_state,
        agent_name=agent_name,
    )
    return build_world_feedback_summary_prompt(
        agent_name=agent_name,
        action_proposal=action_proposal,
        route=fallback.route,
        world_state_diff=fallback.world_state_diff,
    )


def generate_world_feedback_summary(
    *,
    agent_name: str,
    action_proposal: str,
    route: str,
    world_state_diff: dict | None = None,
    provider_name: str = "ollama",
    model: str | None = None,
    print_output: bool = True,
) -> str:
    prompt = build_world_feedback_summary_prompt(
        agent_name=agent_name,
        action_proposal=action_proposal,
        route=route,
        world_state_diff=world_state_diff,
    )
    raw = APIManager(provider_name=provider_name).generate(prompt, model=model)
    if print_output:
        print("World feedback summary output:")
        print(raw)
    return raw.strip()


def check_world_feedback_output(raw_output: str, *, enabled: bool = True) -> None:
    if enabled:
        print(raw_output)


def _extract_agent_state(current_state: dict) -> dict:
    agent_node = current_state.get("agent_node", {}) or {}
    state = agent_node.get("state", {}) if isinstance(agent_node, dict) else {}
    return dict(state) if isinstance(state, dict) else {}


def _current_elements_from_state(current_state: dict) -> list[dict]:
    elements = []
    for node in current_state.get("relevant_nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_type = node.get("node_type", "")
        if node_type not in {"permanent_element", "temporary_element"}:
            continue
        state = node.get("state", {}) or {}
        if node_type == "temporary_element" and state.get("visible") is False:
            continue
        elements.append(node)
    return elements


def _environment_changes_from_report(report: dict) -> list[dict]:
    changes: list[dict] = []
    for node in report.get("created_temporary_nodes", []) or []:
        if isinstance(node, dict):
            changes.append({"change_type": "temporary_created", **node})
    for item in report.get("updated_temporary_nodes", []) or []:
        if isinstance(item, dict):
            changes.append({"change_type": "temporary_updated", **item})
    for edge in report.get("added_fact_edges", []) or []:
        if isinstance(edge, dict):
            changes.append({"change_type": "fact_added", **edge})
    for edge in report.get("removed_fact_edges", []) or []:
        if isinstance(edge, dict):
            changes.append({"change_type": "fact_removed", **edge})
    if report.get("updated_agent_node"):
        changes.append({
            "change_type": "agent_state_updated",
            "node": report["updated_agent_node"],
        })
    for item in report.get("updated_element_nodes", []) or []:
        if isinstance(item, dict):
            changes.append({"change_type": "element_state_updated", **item})
    return changes


def _compose_perception_summary(
    *,
    agent_name: str,
    agent_state: dict,
    environment_changes: list[dict],
    current_elements: list[dict],
) -> str:
    parts = []
    node_names = _node_name_map(current_elements, environment_changes)
    feeling = _agent_state_sentence(
        agent_state,
        current_elements=current_elements,
        node_names=node_names,
        agent_name=agent_name,
    )
    if feeling:
        parts.append(feeling)
    change_text = _environment_change_sentence(
        environment_changes,
        current_elements=current_elements,
        agent_name=agent_name,
    )
    if change_text:
        parts.append(change_text)
    return " ".join(parts).strip()


def _agent_state_sentence(
    agent_state: dict,
    *,
    current_elements: list[dict],
    node_names: dict[str, str],
    agent_name: str,
) -> str:
    details = []
    posture = agent_state.get("posture")
    if posture and str(posture) != "standing":
        details.append(_posture_label(str(posture)))
    body_surface = agent_state.get("body_surface")
    if body_surface and str(body_surface) not in {"dry_clean"}:
        details.append(f"Body surface: {_body_surface_label(str(body_surface))}")
    method = agent_state.get("interaction_method")
    if method and not _looks_like_holding_method(str(method)):
        details.append(str(method))
    gaze_target = agent_state.get("gaze_target")
    if gaze_target:
        details.append(f"Looking at {node_names.get(str(gaze_target), 'target nearby')}")
    if not details:
        return ""
    return f"{agent_name}" + "，".join(details) + "。"


def _environment_change_sentence(
    changes: list[dict],
    *,
    current_elements: list[dict],
    agent_name: str,
) -> str:
    texts = []
    node_names = _node_name_map(current_elements, changes)
    created_node_ids = set()
    for change in changes:
        change_type = change.get("change_type", "")
        if change_type == "temporary_created":
            name = change.get("name", "A temporary object")
            node_id = str(change.get("node_id", "") or "")
            if node_id:
                created_node_ids.add(node_id)
            status = (change.get("state", {}) or {}).get("status", "")
            if status == "held":
                texts.append(f"Holding {name} in hand")
            elif status:
                texts.append(f"{name}{_temporary_status_label(str(status))}")
            else:
                texts.append(f"{name} has appeared")
        elif change_type == "temporary_updated":
            name = change.get("name") or change.get("node_id") or "A temporary object"
            new_state = change.get("new_state", {}) or {}
            status = new_state.get("status", "")
            visible = new_state.get("visible", None)
            if status in {"consumed", "disposed", "discarded"} or visible is False:
                texts.append(f"{name} is no longer visible")
            elif status:
                texts.append(f"{name} now has status: {_temporary_status_label(str(status))}")
        elif change_type == "fact_added":
            relation = change.get("relation", "")
            object_id = str(change.get("to_node_id", "") or change.get("object_id", "") or "")
            if relation == "holding" and object_id not in created_node_ids:
                object_name = node_names.get(object_id, "An item")
                texts.append(f"Holding {object_name}")
            elif relation == "placed_on":
                texts.append("An item has been placed on a relevant surface")
        elif change_type == "fact_removed":
            relation = change.get("relation", "")
            if relation == "holding":
                texts.append("No longer holding a relevant item")
        elif change_type == "agent_state_updated":
            continue
        elif change_type == "element_state_updated":
            node_id = str(change.get("node_id", "") or "")
            name = node_names.get(node_id, node_id or "Relevant object")
            for text in _element_state_change_texts(name, change):
                texts.append(text)
    if not texts:
        return ""
    return f"{agent_name}" + "；".join(_agent_experience_phrase(text) for text in texts[:4]) + "。"


def _element_state_change_texts(name: str, change: dict) -> list[str]:
    old_state = change.get("old_state", {}) or {}
    new_state = change.get("new_state", {}) or {}
    coarse_device_text = _coarse_device_state_change_text(name, old_state, new_state)
    if coarse_device_text:
        return [coarse_device_text]

    texts = []
    old_interaction = old_state.get("interaction_status")
    new_interaction = new_state.get("interaction_status")
    if old_interaction != new_interaction and new_interaction:
        if new_interaction == "in_use":
            texts.append(f"{name} is being used")
        else:
            texts.append(f"{name}'s state changed to {new_interaction}")

    old_details = old_state.get("state_details", {}) or {}
    new_details = new_state.get("state_details", {}) or {}
    for key in sorted(set(old_details) | set(new_details)):
        old_value = old_details.get(key)
        new_value = new_details.get(key)
        if old_value == new_value or not new_value:
            continue
        if key == "door_state" and new_value == "open":
            texts.append(f"The door of {name} is now open")
        elif key == "door_state" and new_value == "closed":
            texts.append(f"The door of {name} is now closed")
        elif key == "flow_state" and new_value == "on":
            texts.append(f"{name} is dispensing water")
        elif key == "power_state" and new_value in {"on", "running"}:
            texts.append(f"{name} has been activated")
        elif key == "surface_state" and new_value == "clean":
            texts.append(f"The surface of {name} is now clean")
        else:
            texts.append(f"{name}'s {key} changed to {new_value}")
    return texts


def _coarse_device_state_change_text(name: str, old_state: dict, new_state: dict) -> str:
    old_details = old_state.get("state_details", {}) or {}
    new_details = new_state.get("state_details", {}) or {}
    old_evolution = str(old_state.get("evolution_status", "") or "")
    new_evolution = str(new_state.get("evolution_status", "") or "")
    old_power = str(old_details.get("power_state", "") or "")
    new_power = str(new_details.get("power_state", "") or "")

    if "Washing machine" in name:
        if _became_active(old_evolution, new_evolution, old_power, new_power):
            return f"Confirm {name} has started washing"
        if old_details.get("contains") != new_details.get("contains") and new_details.get("contains"):
            return f"Confirm the clothes to be washed have been placed in {name}"
        if old_details.get("door_state") != new_details.get("door_state"):
            return f"Confirm {name} is ready for continued use"
        return ""

    if "microwave oven" in name:
        if _became_active(old_evolution, new_evolution, old_power, new_power) or new_evolution == "heating":
            return f"Confirm {name} has started heating"
        if old_details.get("contains") != new_details.get("contains") and new_details.get("contains"):
            return f"Confirm the food has been placed in {name}"
        return ""

    if "computer" in name or name.lower() in {"computer", "pc"}:
        if _became_active(old_evolution, new_evolution, old_power, new_power):
            return f"Confirm {name} is open"
        if new_power == "off" and old_power and old_power != "off":
            return f"Confirm {name} is closed"
        return ""

    return ""


def _became_active(
    old_evolution: str,
    new_evolution: str,
    old_power: str,
    new_power: str,
) -> bool:
    active_values = {"on", "running", "heating", "washing"}
    return (
        (new_power in active_values and old_power != new_power)
        or (new_evolution in active_values and old_evolution != new_evolution)
    )


def _agent_experience_phrase(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    direct_prefixes = ("in hand", "oneself", "currently", "Confirm", "see", "Noticed", "No longer", "Already")
    if stripped.startswith(direct_prefixes):
        return stripped
    return f"Noticed {stripped}"


def _current_elements_sentence(elements: list[dict]) -> str:
    names = []
    for element in elements:
        name = element.get("name")
        if name:
            names.append(str(name))
    if not names:
        return ""
    return "Currently relevant perceptible objects are:" + "、".join(names[:6]) + "。"


def _node_name_map(elements: list[dict], changes: list[dict]) -> dict[str, str]:
    names: dict[str, str] = {}
    for element in elements:
        if not isinstance(element, dict):
            continue
        node_id = str(element.get("node_id", "") or "")
        name = str(element.get("name", "") or "")
        if node_id and name:
            names[node_id] = name
    for change in changes:
        if not isinstance(change, dict):
            continue
        node_id = str(change.get("node_id", "") or "")
        name = str(change.get("name", "") or "")
        if node_id and name:
            names[node_id] = name
    return names


def _looks_like_holding_method(text: str) -> bool:
    return any(word in text for word in ("Holding", "Picking up", "Gripping", "Carrying (in hands)", "Cradling"))


def _posture_label(value: str) -> str:
    return {
        "standing": "standing",
        "sitting": "sitting",
        "lying": "lying down",
        "crouching": "crouching",
        "walking": "Walking",
    }.get(value, value)


def _body_surface_label(value: str) -> str:
    return {
        "dry_clean": "Dry and clean",
        "wet": "Wet",
        "dirty": "A bit dirty",
        "soapy": "Foamy",
        "dry_dirty": "Dry but a bit dirty",
    }.get(value, value)


def _temporary_status_label(value: str) -> str:
    return {
        "held": "Held in hand",
        "placed": "Placed",
        "in_use": "Being used",
        "open": "In the open state",
        "closed": "In the closed state",
        "consumed": "Already consumed",
        "disposed": "Already processed",
        "discarded": "Already discarded",
    }.get(value, f"The status is {value}")

from __future__ import annotations

from contextual_world.modules.state_transition.types import WorldStateTransitionResult

from .types import WorldTransitionCheckResult


def check_world_state_transition_result(
    *,
    transition_result: WorldStateTransitionResult,
    allowed_node_ids: set[str],
    removable_fact_keys: set[tuple[str, str, str]],
    node_reference: dict[str, dict] | None = None,
) -> WorldTransitionCheckResult:
    issues: list[str] = []
    node_reference = node_reference or {}
    delta = transition_result.next_state_delta or {}
    accepted = bool(transition_result.accepted)
    for item in delta.get("node_updates", []) or []:
        node_id = str(item.get("node_id", "") or "").strip()
        if node_id not in allowed_node_ids:
            issues.append(f"node update outside focused subgraph: {node_id}")
        if not accepted:
            node_type = str(item.get("node_type", "") or node_reference.get(node_id, {}).get("node_type", "") or "")
            if node_type != "human_agent":
                issues.append(f"rejected transition updates external node: {node_id}")
    for item in delta.get("fact_edges_to_add", []) or []:
        subject_id = str(item.get("subject_id", "") or "").strip()
        object_id = str(item.get("object_id", "") or "").strip()
        if subject_id not in allowed_node_ids or object_id not in allowed_node_ids:
            issues.append(f"fact edge add outside focused subgraph: {subject_id} -> {object_id}")
        if not accepted:
            issues.append(f"rejected transition adds fact edge: {subject_id} -> {object_id}")
    valid_removals = []
    for item in delta.get("fact_edges_to_remove", []) or []:
        key = (
            str(item.get("subject_id", "") or "").strip(),
            str(item.get("relation", "") or "").strip(),
            str(item.get("object_id", "") or "").strip(),
        )
        if key in removable_fact_keys:
            valid_removals.append(item)
    delta["fact_edges_to_remove"] = valid_removals
    return WorldTransitionCheckResult(
        check_status="rejected" if issues else "accepted",
        issues=issues,
    )

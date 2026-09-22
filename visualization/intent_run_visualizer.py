from __future__ import annotations

import json
from pathlib import Path

from engine.interaction_engine import EngineStepResult
from contextual_world.world_old.spatial_rules.physics_rules import get_physics_profile
from contextual_world.structure.scene_schema import Home


OUTPUT_HTML = "intent_action_run.html"
SCALE = 90
PADDING = 30

def _scene_size(home: Home) -> tuple[float, float]:
    x_values = []
    y_values = []
    for area in home.areas:
        x_min, y_min, x_max, y_max = area.bounds
        x_values.extend([x_min, x_max])
        y_values.extend([y_min, y_max])
    return max(x_values), max(y_values)


def _element_payload(home: Home) -> list[dict]:
    areas = []
    area_colors = getattr(home, "rendering", {}).get("area_colors", {})
    for area in home.areas:
        elements = []
        for element in area.elements:
            profile = get_physics_profile(element)
            elements.append(
                {
                    "id": element.node_id,
                    "name": element.name,
                    "center": list(element.center),
                    "size": list(element.size),
                    "blocking": profile.is_blocking,
                    "physical_status": element.physical_status,
                    "evolution_status": element.evolution_status,
                    "interaction_status": element.interaction_status,
                    "state_details": dict(element.state_details),
                }
            )
        areas.append(
            {
                "id": area.node_id,
                "name": area.name,
                "bounds": list(area.bounds),
                "color": area_colors.get(area.node_id, "#ded6c8"),
                "elements": elements,
            }
        )
    return areas


def _change_payload(result: EngineStepResult) -> list[dict]:
    changes = []
    for item in result.world_changes:
        fields = []
        if item.old_physical_status != item.new_physical_status:
            fields.append(["physical_status", item.old_physical_status, item.new_physical_status])
        if item.old_evolution_status != item.new_evolution_status:
            fields.append(["evolution_status", item.old_evolution_status, item.new_evolution_status])
        if item.old_interaction_status != item.new_interaction_status:
            fields.append(["interaction_status", item.old_interaction_status, item.new_interaction_status])
        detail_keys = set(item.old_state_details) | set(item.new_state_details)
        for key in sorted(detail_keys):
            old = item.old_state_details.get(key, "")
            new = item.new_state_details.get(key, "")
            if old != new:
                fields.append([f"state_details.{key}", old or "unset", new or "unset"])
        for field, old, new in fields:
            changes.append(
                {
                    "area_id": item.area_id,
                    "element_id": item.element_id,
                    "element_name": item.element_name,
                    "field": field,
                    "old": old,
                    "new": new,
                }
            )
    return changes


def _world_summary(result: EngineStepResult) -> str:
    feedback = getattr(result, "environment_feedback", {}) or {}
    if not isinstance(feedback, dict):
        return ""
    return str(feedback.get("perception_summary", "") or "").strip()


def _control_source(result: EngineStepResult) -> tuple[str, str]:
    if str(getattr(result, "execution_kind", "") or "") == "move_precondition":
        return "positioning", "World Positioning"
    mode = str(getattr(result, "arbiter_mode", "") or "").strip()
    has_habit = bool(str(getattr(result, "habitual_response_key", "") or "").strip())
    if has_habit and mode in {"fast_path", "habitual"}:
        return "habitual", "Habitual"
    if has_habit and mode in {"combine", "sequence"}:
        return "combined", "Goal + Habit"
    if str(getattr(result, "computation_architecture", "") or "") == "world_model":
        return "world_model", "Goal-Directed / World Model"
    return "goal_directed", "Goal-Directed"


def build_payload(
    home: Home,
    *,
    initial_position: tuple[float, float],
    results: list[EngineStepResult],
    desire_state: dict | None = None,
    initial_desire_state: dict | None = None,
    desire_updates: list[dict] | None = None,
    cumulative_reward: dict | None = None,
    world_failure_report: dict | None = None,
) -> dict:
    steps = []
    previous_position = initial_position
    width, height = _scene_size(home)
    for turn_index, result in enumerate(results, start=1):
        control_source, control_label = _control_source(result)
        target_ids = [
            target_id
            for target_id in [
                result.target_element_id,
                result.navigation_target_element_id,
                result.secondary_target_element_id,
            ]
            if target_id
        ]
        change_targets = [change.element_id for change in result.world_changes]
        world_summary = _world_summary(result)
        steps.append(
            {
                "step_id": result.step_id,
                "turn_index": turn_index,
                "action_index": result.intent_action_index,
                "counts_as_intent_action": getattr(
                    result,
                    "counts_as_intent_action",
                    True,
                ),
                "agent_name": result.agent_name,
                "from_position": list(previous_position),
                "to_position": list(result.final_position),
                "intent_text": result.intent_text,
                "intent_status": result.intent_status,
                "intent_progress": result.intent_progress,
                "desire_state": getattr(result, "desire_state", None),
                "desire_update": getattr(result, "desire_update", None),
                "time_text": getattr(result, "time_text", ""),
                "self_belief": getattr(result, "self_belief", ""),
                "proposal": result.action_proposal_text,
                "control_source": control_source,
                "control_label": control_label,
                "computation_architecture": getattr(
                    result,
                    "computation_architecture",
                    "intuition",
                ),
                "world_model_trace": getattr(result, "world_model_trace", {}) or {},
                "action_generation_trace": getattr(
                    result,
                    "action_generation_trace",
                    {},
                ) or {},
                "decision": result.decision_text,
                "action_type": result.action_type,
                "execution_kind": getattr(result, "execution_kind", "") or result.action_type,
                "target_area_id": result.target_area_id,
                "target_ids": list(dict.fromkeys([*target_ids, *change_targets])),
                "action_text": result.action_text,
                "world_summary": world_summary,
                "feedback": world_summary or result.execution_narration or result.action_text,
                "estimated_duration": result.estimated_duration,
                "temporary_element_changes": result.temporary_element_changes,
                "actor_state": getattr(result, "actor_state", {}),
                "actor_state_update": getattr(result, "actor_state_update", {}),
                "changes": _change_payload(result),
            }
        )
        previous_position = result.final_position

    return {
        "home": {
            "width": width,
            "height": height,
            "areas": _element_payload(home),
            "portals": list(getattr(home, "portals", [])),
        },
        "initial_position": list(initial_position),
        "agent": {
            "name": results[0].agent_name if results else "Agent",
            "facing": (results[0].actor_state or {}).get("facing", 90.0) if results else 90.0,
            "field_of_view_degrees": (results[0].actor_state or {}).get("field_of_view_degrees", 160.0) if results else 160.0,
        },
        "steps": steps,
        "desire": {
            "initial": initial_desire_state or desire_state or {},
            "current": desire_state or initial_desire_state or {},
            "updates": desire_updates or [],
        },
        "cumulative_reward": cumulative_reward or {},
        "world_failure_report": world_failure_report or {},
        "scale": SCALE,
        "padding": PADDING,
    }


def build_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Intent Action Replay</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2528;
      --muted: #687076;
      --paper: #f6f0e7;
      --panel: #fffaf3;
      --line: #d8c7b5;
      --accent: #d6672d;
      --blue: #2267b5;
      --target: #f2c94c;
      --change: #d64545;
    }}
    * {{ box-sizing: border-box; }}
    html {{
      min-height: 100%;
      overflow-y: auto;
    }}
    body {{
      margin: 0;
      min-height: 100%;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
      overflow-x: auto;
    }}
    .page-tabs {{
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 10px 18px;
      border-bottom: 1px solid var(--line);
      background: #fffaf3;
    }}
    .page-tab {{
      width: auto;
      min-width: 110px;
      padding: 0 14px;
      background: transparent;
      border-color: transparent;
      color: var(--muted);
      font-weight: 650;
    }}
    .page-tab.active {{
      color: var(--ink);
      background: #f0e2d3;
      border-color: #cdb49d;
    }}
    .page-view[hidden] {{ display: none; }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(560px, 1fr) 8px minmax(860px, 46vw);
      align-items: start;
    }}
    .resizer {{
      background: linear-gradient(90deg, transparent, rgba(120, 92, 64, 0.28), transparent);
      cursor: col-resize;
      height: 100vh;
      min-height: 100vh;
      position: sticky;
      top: 0;
      touch-action: none;
    }}
    .resizer:hover, .resizer.dragging {{
      background: rgba(214, 103, 45, 0.34);
    }}
    .stage {{
      padding: 18px;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 12px;
      position: sticky;
      top: 0;
      align-self: start;
    }}
    .map-wrap {{
      position: relative;
      min-height: 0;
      background: #fbf7f0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 18px 50px rgba(62, 43, 24, 0.13);
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
      min-height: 620px;
    }}
    .side {{
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      display: grid;
      grid-template-columns: minmax(320px, 1fr) 8px minmax(460px, 680px);
      grid-template-rows: auto auto;
      gap: 14px;
      min-width: 0;
      min-height: 100vh;
      overflow: visible;
      align-content: start;
    }}
    .side-header {{
      grid-column: 1 / -1;
    }}
    .detail-column {{
      min-width: 0;
      overflow: visible;
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding-right: 2px;
    }}
    .side-split-resizer {{
      min-height: 0;
      border-radius: 6px;
      cursor: col-resize;
      background: linear-gradient(90deg, transparent, rgba(120, 92, 64, 0.18), transparent);
      touch-action: none;
    }}
    .side-split-resizer:hover, .side-split-resizer.dragging {{
      background: rgba(214, 103, 45, 0.28);
    }}
    .timeline-column {{
      min-width: 420px;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: visible;
    }}
    h1 {{
      margin: 0;
      font-size: 21px;
      font-weight: 750;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.64);
      padding: 12px;
    }}
    .step-title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      font-size: 14px;
      color: var(--muted);
      margin-top: 2px;
    }}
    .text {{
      font-size: 14px;
      line-height: 1.55;
      white-space: pre-wrap;
    }}
    .big {{
      font-size: 16px;
      line-height: 1.5;
      font-weight: 650;
    }}
    .state-gauges {{
      display: grid;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .state-gauge {{
      display: grid;
      grid-template-columns: 72px minmax(100px, 1fr) 42px;
      gap: 8px;
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }}
    .state-gauge-track {{
      height: 16px;
      background: #e4ddd5;
      border: 1px solid #b9aa9b;
      border-radius: 3px;
      overflow: hidden;
      box-shadow: inset 0 1px 3px rgba(55, 39, 27, 0.18);
    }}
    .state-gauge-fill {{
      height: 100%;
      border-radius: 2px;
      background-color: #2f7d67;
      background-image: repeating-linear-gradient(
        90deg,
        transparent 0,
        transparent calc(10% - 1px),
        rgba(255, 255, 255, 0.42) calc(10% - 1px),
        rgba(255, 255, 255, 0.42) 10%
      );
      transition: width 180ms ease, background-color 180ms ease;
    }}
    .state-gauge-fill.low {{ background-color: #2f7d67; }}
    .state-gauge-fill.medium {{ background-color: #d39a27; }}
    .state-gauge-fill.high {{ background-color: #c84b3c; }}
    .state-gauge-fill.unknown {{ background-color: #9b948d; }}
    .state-gauge-value {{
      text-align: right;
      color: #3f3329;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 42px 42px 1fr 72px;
      gap: 8px;
      align-items: center;
    }}
    button, select {{
      height: 36px;
      border: 1px solid #bfa58b;
      border-radius: 7px;
      background: #fff8ef;
      color: #382719;
      cursor: pointer;
      font-size: 14px;
    }}
    button:hover, select:hover {{ border-color: var(--accent); }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    .timeline {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      overflow: visible;
      padding-right: 0;
      min-height: 0;
      width: 100%;
    }}
    .timeline button {{
      width: 100%;
      height: auto;
      min-height: 0;
      text-align: left;
      padding: 10px 12px;
      border-color: #decbb9;
      background: #fffaf3;
      line-height: 1.45;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .timeline-proposal {{
      display: block;
      color: #6b5540;
      font-size: 13px;
      line-height: 1.36;
      margin-bottom: 6px;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .timeline-source {{
      display: inline-flex;
      width: fit-content;
      margin-bottom: 7px;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      color: white;
      background: #3f6f62;
    }}
    .timeline-source.habitual {{ background: #b85d24; }}
    .timeline-source.combined {{ background: #76569a; }}
    .timeline-source.world_model {{ background: #386f8c; }}
    .timeline-source.positioning {{ background: #687078; }}
    .timeline-summary {{
      display: block;
      font-size: 14px;
      line-height: 1.5;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .timeline button.active {{
      background: #d6672d;
      border-color: #b84f1d;
      color: white;
    }}
    .timeline button.active .timeline-proposal {{ color: rgba(255,255,255,0.82); }}
    .changes {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 13px;
      line-height: 1.42;
    }}
    .pill {{
      display: inline-flex;
      width: fit-content;
      max-width: 100%;
      border: 1px solid #d9c5af;
      background: #fff5e6;
      border-radius: 999px;
      padding: 3px 8px;
      color: #58402c;
      font-size: 12px;
    }}
    .empty {{ color: var(--muted); font-size: 13px; }}
    .state-action-page {{
      min-height: calc(100vh - 57px);
      padding: 24px;
      background: #f8f3ec;
    }}
    .state-action-shell {{
      width: min(1500px, 100%);
      margin: 0 auto;
    }}
    .state-action-header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--line);
    }}
    .state-action-header p {{
      margin: 5px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .state-action-list {{
      display: grid;
      gap: 0;
      margin-top: 16px;
      border-top: 1px solid var(--line);
    }}
    .state-action-row {{
      display: grid;
      grid-template-columns: 120px minmax(320px, 0.9fr) minmax(420px, 1.35fr);
      min-width: 0;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.46);
    }}
    .state-action-row:nth-child(even) {{ background: rgba(255, 250, 243, 0.82); }}
    .trace-cell {{
      min-width: 0;
      padding: 14px 16px;
      border-right: 1px solid var(--line);
    }}
    .trace-cell:last-child {{ border-right: 0; }}
    .trace-turn {{
      font-size: 15px;
      font-weight: 750;
      font-variant-numeric: tabular-nums;
    }}
    .trace-meta, .trace-result {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
    .trace-label {{
      margin-bottom: 7px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 750;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .trace-state {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 13px;
      line-height: 1.5;
    }}
    .trace-action {{
      font-size: 15px;
      line-height: 1.5;
      font-weight: 650;
      overflow-wrap: anywhere;
    }}
    .trace-source {{ margin-bottom: 8px; }}
    .final-settlement {{
      display: grid;
      grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr);
      gap: 24px;
      margin-top: 24px;
      padding: 20px 0;
      border-top: 2px solid #8d725b;
    }}
    .final-state-text, .final-reward-text {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 13px;
      line-height: 1.55;
    }}
    .final-reward-value {{
      margin: 4px 0 8px;
      font-size: 28px;
      font-weight: 780;
      font-variant-numeric: tabular-nums;
    }}
    .world-errors-page {{
      min-height: calc(100vh - 57px);
      padding: 24px;
      background: #f8f3ec;
    }}
    .world-errors-shell {{
      width: min(1500px, 100%);
      margin: 0 auto;
    }}
    .failure-summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(150px, 1fr));
      gap: 1px;
      margin-top: 16px;
      border: 1px solid var(--line);
      background: var(--line);
    }}
    .failure-metric {{
      min-width: 0;
      padding: 14px 16px;
      background: #fffaf3;
    }}
    .failure-metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 22px;
      font-variant-numeric: tabular-nums;
    }}
    .run-error {{
      margin-top: 16px;
      padding: 12px 14px;
      border-left: 4px solid #c84b3c;
      background: #fff0ed;
      color: #6f241d;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 13px;
      line-height: 1.5;
    }}
    .failure-list {{
      display: grid;
      gap: 14px;
      margin-top: 20px;
    }}
    .failure-item {{
      display: grid;
      grid-template-columns: 170px minmax(300px, 0.8fr) minmax(420px, 1.2fr);
      border: 1px solid #d7b9ad;
      background: rgba(255, 255, 255, 0.58);
    }}
    .failure-item .trace-cell {{ border-color: #dfc7bd; }}
    .failure-kind {{
      color: #9a342b;
      font-weight: 750;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .failure-text {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 13px;
      line-height: 1.5;
    }}
    .failure-details {{ margin-top: 10px; }}
    .failure-details summary {{
      cursor: pointer;
      color: var(--blue);
      font-size: 12px;
      font-weight: 650;
    }}
    .failure-details pre {{
      max-height: 420px;
      overflow: auto;
      margin: 8px 0 0;
      padding: 10px;
      border: 1px solid #dacabb;
      background: #fffaf3;
      font-size: 11px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .failure-empty {{
      margin-top: 20px;
      padding: 26px 0;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 14px;
    }}
    .area-label {{ font-size: 12px; fill: #394044; font-weight: 700; }}
    .element-label {{ font-size: 9px; fill: #1f2528; pointer-events: none; }}
    .element {{ transition: fill 180ms ease, stroke 180ms ease, stroke-width 180ms ease, opacity 180ms ease; }}
    .element.target {{ stroke: var(--target); stroke-width: 5; }}
    .element.changed {{ stroke: var(--change); stroke-width: 5; }}
    .path-line {{ fill: none; stroke: var(--accent); stroke-width: 5; stroke-linecap: round; stroke-linejoin: round; opacity: 0.92; }}
    .path-ghost {{ fill: none; stroke: #7c8890; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; opacity: 0.22; }}
    .agent {{
      filter: drop-shadow(0 4px 8px rgba(0,0,0,0.22));
      transition: transform 420ms ease;
    }}
    .agent-dot {{ fill: #1f8f63; stroke: #093d2a; stroke-width: 2; }}
    .agent-arrow {{ fill: #093d2a; }}
    .view-sector {{
      fill: rgba(255, 196, 72, 0.18);
      stroke: rgba(190, 126, 20, 0.58);
      stroke-width: 1.5;
      stroke-dasharray: 5 4;
      pointer-events: none;
    }}
    .speech {{
      position: absolute;
      left: 18px;
      bottom: 18px;
      max-width: min(560px, calc(100% - 36px));
      background: rgba(255, 250, 243, 0.94);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 14px;
      line-height: 1.45;
      box-shadow: 0 10px 26px rgba(62, 43, 24, 0.14);
    }}
    @media (max-width: 1460px) {{
      .app {{ grid-template-columns: 1fr; }}
      .resizer {{ display: none; }}
      .stage {{
        position: relative;
        top: auto;
      }}
      .side {{
        border-left: 0;
        border-top: 1px solid var(--line);
        grid-template-columns: minmax(300px, 1fr) 8px minmax(420px, 1.2fr);
        min-height: 0;
      }}
    }}
    @media (max-width: 860px) {{
      .side {{
        grid-template-columns: 1fr;
        max-height: none;
        overflow: visible;
      }}
      .side-split-resizer {{ display: none; }}
      .side-header, .timeline-column {{ grid-column: auto; }}
      .detail-column, .timeline-column {{ overflow: visible; }}
      .timeline-column {{ min-width: 0; }}
      svg {{ min-height: 420px; }}
      .state-action-page {{ padding: 16px; }}
      .state-action-header {{ align-items: start; flex-direction: column; }}
      .state-action-row {{ grid-template-columns: 1fr; }}
      .trace-cell {{ border-right: 0; border-bottom: 1px dashed #dacabb; }}
      .trace-cell:last-child {{ border-bottom: 0; }}
      .final-settlement {{ grid-template-columns: 1fr; }}
      .world-errors-page {{ padding: 16px; }}
      .failure-summary, .failure-item {{ grid-template-columns: 1fr; }}
      .failure-item .trace-cell {{ border-right: 0; border-bottom: 1px dashed #dfc7bd; }}
      .failure-item .trace-cell:last-child {{ border-bottom: 0; }}
    }}
  </style>
</head>
<body>
  <nav class="page-tabs" aria-label="Replay views">
    <button class="page-tab active" id="replayTab" type="button">Replay</button>
    <button class="page-tab" id="stateActionTab" type="button">State / Action / Reward</button>
    <button class="page-tab" id="worldErrorsTab" type="button">World Errors</button>
  </nav>
  <div class="page-view" id="replayPage">
  <div class="app">
    <main class="stage">
      <div class="map-wrap">
        <svg id="scene"></svg>
        <div class="speech" id="speech"></div>
      </div>
      <div class="controls">
        <button id="prevBtn" title="Previous step">◀</button>
        <button id="playBtn" title="Play or pause">▶</button>
        <input id="stepRange" type="range" min="0" max="0" value="0" />
        <select id="speedSelect" title="Playback speed">
          <option value="1800">0.5x</option>
          <option value="1000" selected>1x</option>
          <option value="520">2x</option>
        </select>
      </div>
    </main>
    <div class="resizer" id="layoutResizer" title="Drag to resize replay panel"></div>
    <aside class="side">
      <div class="side-header">
        <h1>Intent Action Replay</h1>
        <div class="step-title">
          <span id="stepLabel"></span>
          <span id="statusLabel"></span>
        </div>
      </div>
      <div class="detail-column">
        <section class="card">
          <h2>Intent</h2>
          <div class="text" id="intentText"></div>
        </section>
        <section class="card">
          <h2>Cumulative Reward</h2>
          <div class="big" id="cumulativeRewardValue"></div>
          <div class="text" id="cumulativeRewardText"></div>
        </section>
        <section class="card">
          <h2>State</h2>
          <div class="state-gauges" id="stateGauges"></div>
          <div class="text" id="stateText"></div>
        </section>
        <section class="card">
          <h2>Does</h2>
          <div class="big" id="proposalText"></div>
          <div class="pill" id="actionType"></div>
        </section>
        <section class="card">
          <h2>World Feedback</h2>
          <div class="text" id="feedbackText"></div>
        </section>
        <section class="card">
          <h2>Agent State</h2>
          <div class="text" id="actorStateText"></div>
        </section>
        <section class="card">
          <h2>Estimated Duration</h2>
          <div class="text" id="durationText"></div>
        </section>
        <section class="card">
          <h2>World Changes</h2>
          <div class="changes" id="changesText"></div>
        </section>
        <section class="card">
          <h2>Temporary Elements</h2>
          <div class="changes" id="temporaryElementsText"></div>
        </section>
      </div>
      <div class="side-split-resizer" id="sideSplitResizer" title="Drag to resize timeline"></div>
      <section class="card timeline-column">
        <h2 id="timelineTitle">Timeline</h2>
        <div class="timeline" id="timeline"></div>
      </section>
    </aside>
  </div>
  </div>
  <section class="page-view state-action-page" id="stateActionPage" hidden>
    <div class="state-action-shell">
      <header class="state-action-header">
        <div>
          <h1>State / Action Trace</h1>
          <p>Each row shows the state available before an action and the result returned by World.</p>
        </div>
        <div class="pill" id="traceTurnCount"></div>
      </header>
      <div class="state-action-list" id="stateActionList"></div>
      <footer class="final-settlement">
        <section>
          <h2>Final State</h2>
          <div class="final-state-text" id="finalStateText"></div>
        </section>
        <section>
          <h2>Final Cumulative Reward</h2>
          <div class="final-reward-value" id="finalRewardValue"></div>
          <div class="final-reward-text" id="finalRewardText"></div>
        </section>
      </footer>
    </div>
  </section>
  <section class="page-view world-errors-page" id="worldErrorsPage" hidden>
    <div class="world-errors-shell">
      <header class="state-action-header">
        <div>
          <h1>Contextual World Errors</h1>
          <p>Failures recorded during this simulation run only.</p>
        </div>
        <div class="pill" id="worldFailureCount"></div>
      </header>
      <div class="failure-summary">
        <div class="failure-metric">
          <div class="trace-label">Action turns</div>
          <strong id="failureActionTurns">0</strong>
        </div>
        <div class="failure-metric">
          <div class="trace-label">World failures</div>
          <strong id="failureTotal">0</strong>
        </div>
        <div class="failure-metric">
          <div class="trace-label">Failed modules</div>
          <strong id="failureModuleCount">0</strong>
        </div>
      </div>
      <div class="run-error" id="runErrorText" hidden></div>
      <div class="failure-list" id="worldFailureList"></div>
      <div class="failure-empty" id="worldFailureEmpty" hidden>
        No Contextual World pipeline errors were recorded in this run.
      </div>
    </div>
  </section>
  <script id="replay-data" type="application/json">{data_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("replay-data").textContent);
    const svg = document.getElementById("scene");
    const speech = document.getElementById("speech");
    const stepRange = document.getElementById("stepRange");
    const playBtn = document.getElementById("playBtn");
    const prevBtn = document.getElementById("prevBtn");
    const speedSelect = document.getElementById("speedSelect");
    const timeline = document.getElementById("timeline");
    const cumulativeRewardValue = document.getElementById("cumulativeRewardValue");
    const cumulativeRewardText = document.getElementById("cumulativeRewardText");
    const replayTab = document.getElementById("replayTab");
    const stateActionTab = document.getElementById("stateActionTab");
    const worldErrorsTab = document.getElementById("worldErrorsTab");
    const replayPage = document.getElementById("replayPage");
    const stateActionPage = document.getElementById("stateActionPage");
    const worldErrorsPage = document.getElementById("worldErrorsPage");

    function rewardLines(reward) {{
      const lines = [];
      if (Number.isFinite(Number(reward.total_intent_satisfaction))) {{
        lines.push(`total satisfaction: ${{Number(reward.total_intent_satisfaction).toFixed(2)}} across ${{reward.intent_count || 0}} intents`);
      }} else if (Number.isFinite(Number(reward.intent_satisfaction))) {{
        lines.push(`satisfaction: ${{Number(reward.intent_satisfaction).toFixed(2)}} / 10`);
      }}
      const steps = Number(reward.execution_steps);
      if (Number.isFinite(steps)) lines.push(`execution steps: ${{steps}}`);
      if (reward.reason) lines.push(`reason: ${{reward.reason}}`);
      for (const item of reward.intent_results || []) {{
        lines.push(`${{item.intent_text}}: ${{Number(item.cumulative_reward).toFixed(4)}}`);
      }}
      return lines;
    }}

    function renderCumulativeReward() {{
      const reward = payload.cumulative_reward || {{}};
      if (reward.error) {{
        cumulativeRewardValue.textContent = "Unavailable";
        cumulativeRewardText.textContent = reward.error;
        document.getElementById("finalRewardValue").textContent = "Unavailable";
        document.getElementById("finalRewardText").textContent = reward.error;
        return;
      }}
      const value = Number(reward.cumulative_reward);
      const valueText = Number.isFinite(value) ? value.toFixed(4) : "-";
      const lines = rewardLines(reward);
      cumulativeRewardValue.textContent = valueText;
      cumulativeRewardText.textContent = lines.join("\\n") || "No settlement result.";
      document.getElementById("finalRewardValue").textContent = valueText;
      document.getElementById("finalRewardText").textContent = lines.join("\\n") || "No settlement result.";
    }}

    function setPage(pageName) {{
      const showReplay = pageName === "replay";
      const showStateAction = pageName === "state-action";
      const showWorldErrors = pageName === "world-errors";
      replayPage.hidden = !showReplay;
      stateActionPage.hidden = !showStateAction;
      worldErrorsPage.hidden = !showWorldErrors;
      replayTab.classList.toggle("active", showReplay);
      stateActionTab.classList.toggle("active", showStateAction);
      worldErrorsTab.classList.toggle("active", showWorldErrors);
    }}
    replayTab.addEventListener("click", () => setPage("replay"));
    stateActionTab.addEventListener("click", () => setPage("state-action"));
    worldErrorsTab.addEventListener("click", () => setPage("world-errors"));
    const app = document.querySelector(".app");
    const side = document.querySelector(".side");
    const layoutResizer = document.getElementById("layoutResizer");
    const sideSplitResizer = document.getElementById("sideSplitResizer");
    const scale = payload.scale;
    const padding = payload.padding;
    const width = payload.home.width * scale + padding * 2;
    const height = payload.home.height * scale + padding * 2;
    let current = 0;
    let playing = false;
    let timer = null;
    const elementNodes = new Map();

    svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
    stepRange.max = Math.max(0, payload.steps.length - 1);

    const toPx = value => value * scale + padding;
    const ns = "http://www.w3.org/2000/svg";
    function el(name, attrs = {{}}, text = "") {{
      const node = document.createElementNS(ns, name);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
      if (text) node.textContent = text;
      return node;
    }}
    function portalColor(kind) {{
      if (kind === "glass_door") return "#2267b5";
      if (kind === "door") return "#b94435";
      return "#74532f";
    }}
    function pointString(points) {{
      return points.map(point => `${{toPx(point[0])}},${{toPx(point[1])}}`).join(" ");
    }}
    function viewSectorPath(fieldOfViewDegrees) {{
      const fov = Math.max(0.1, Math.min(360, Number(fieldOfViewDegrees) || 160));
      const radius = Math.hypot(payload.home.width, payload.home.height) * scale;
      if (fov >= 359.9) {{
        return `M ${{radius}} 0 A ${{radius}} ${{radius}} 0 1 1 ${{-radius}} 0 A ${{radius}} ${{radius}} 0 1 1 ${{radius}} 0 Z`;
      }}
      const half = fov / 2 * Math.PI / 180;
      const start = [radius * Math.cos(-half), radius * Math.sin(-half)];
      const end = [radius * Math.cos(half), radius * Math.sin(half)];
      const largeArc = fov > 180 ? 1 : 0;
      return `M 0 0 L ${{start[0]}} ${{start[1]}} A ${{radius}} ${{radius}} 0 ${{largeArc}} 1 ${{end[0]}} ${{end[1]}} Z`;
    }}
    function drawScene() {{
      svg.innerHTML = "";
      elementNodes.clear();
      svg.appendChild(el("rect", {{ x: 0, y: 0, width, height, fill: "#fbf7f0", stroke: "#ddd0c2" }}));
      for (const area of payload.home.areas) {{
        const [x1, y1, x2, y2] = area.bounds;
        svg.appendChild(el("rect", {{
          x: toPx(x1), y: toPx(y1), width: (x2 - x1) * scale, height: (y2 - y1) * scale,
          fill: area.color, stroke: "#5e6264", "stroke-width": 2, opacity: 0.74
        }}));
        svg.appendChild(el("text", {{ x: toPx(x1) + 8, y: toPx(y1) + 18, class: "area-label" }}, area.name));
        for (const item of area.elements) {{
          const [cx, cy] = item.center;
          const [w, h] = item.size;
          const rect = el("rect", {{
            id: `element-${{item.id}}`, class: "element", x: toPx(cx - w / 2), y: toPx(cy - h / 2),
            width: w * scale, height: h * scale, rx: 2,
            fill: item.blocking ? "#896954" : "#94b977", stroke: item.blocking ? "#372217" : "#315b28",
            "stroke-width": 1.4, opacity: 0.86
          }});
          elementNodes.set(item.id, rect);
          svg.appendChild(rect);
          if (w * scale > 34 && h * scale > 18) {{
            svg.appendChild(el("text", {{ x: toPx(cx - w / 2) + 3, y: toPx(cy - h / 2) + 13, class: "element-label" }}, item.name));
          }}
        }}
      }}
      for (const portal of payload.home.portals) {{
        const [[x1, y1], [x2, y2]] = portal.segment;
        svg.appendChild(el("line", {{
          x1: toPx(x1), y1: toPx(y1), x2: toPx(x2), y2: toPx(y2),
          stroke: portalColor(portal.kind), "stroke-width": portal.kind === "opening" ? 4 : 6,
          "stroke-linecap": "round", "stroke-dasharray": portal.kind === "opening" ? "8 5" : ""
        }}));
      }}
      const path = [payload.initial_position, ...payload.steps.map(step => step.to_position)];
      svg.appendChild(el("polyline", {{ id: "ghostPath", class: "path-ghost", points: pointString(path) }}));
      svg.appendChild(el("polyline", {{ id: "activePath", class: "path-line", points: "" }}));
      const group = el("g", {{ id: "agent", class: "agent" }});
      group.appendChild(el("path", {{ id: "viewSector", class: "view-sector", d: viewSectorPath(payload.agent?.field_of_view_degrees) }}));
      group.appendChild(el("circle", {{ cx: 0, cy: 0, r: 10, class: "agent-dot" }}));
      group.appendChild(el("path", {{ d: "M 13 0 L 2 -5 L 2 5 Z", class: "agent-arrow" }}));
      const agentName = payload.agent?.name || payload.steps[0]?.agent_name || "Agent";
      group.appendChild(el("text", {{ x: 14, y: -12, "font-size": 12, fill: "#093d2a", "font-weight": 700 }}, agentName));
      svg.appendChild(group);
    }}
    function setAgent(step) {{
      const agent = document.getElementById("agent");
      const [x, y] = step.to_position;
      const [fromX, fromY] = step.from_position;
      const movementAngle = Math.atan2(y - fromY, x - fromX) * 180 / Math.PI;
      const state = step.actor_state || {{}};
      const facing = Number.isFinite(Number(state.facing)) ? Number(state.facing) : movementAngle;
      const fieldOfView = state.field_of_view_degrees ?? payload.agent?.field_of_view_degrees ?? 160;
      document.getElementById("viewSector").setAttribute("d", viewSectorPath(fieldOfView));
      agent.setAttribute("transform", `translate(${{toPx(x)}} ${{toPx(y)}}) rotate(${{Number.isFinite(facing) ? facing : 0}})`);
    }}
    function renderTimeline() {{
      timeline.innerHTML = "";
      document.getElementById("timelineTitle").textContent = `Timeline (${{payload.steps.length}} world steps)`;
      payload.steps.forEach((step, index) => {{
        const button = document.createElement("button");
        const source = document.createElement("span");
        source.className = `timeline-source ${{step.control_source || "goal_directed"}}`;
        source.textContent = step.control_label || "Goal-Directed";
        const proposal = document.createElement("span");
        proposal.className = "timeline-proposal";
        proposal.textContent = step.proposal || "(no action proposal)";
        const summary = document.createElement("span");
        summary.className = "timeline-summary";
        summary.textContent = `${{step.turn_index || index + 1}}. ${{step.world_summary || step.feedback || "(no world summary)"}}`;
        button.appendChild(source);
        button.appendChild(proposal);
        button.appendChild(summary);
        button.addEventListener("click", () => setStep(index));
        timeline.appendChild(button);
      }});
    }}
    function formatState(step) {{
      const desire = step?.desire_state || payload.desire?.current || {{}};
      if (!Object.keys(desire).length) return "(no desire state)";
      const goals = desire.work_goal || [];
      const lines = [];
      if (desire.mental) lines.push(`mental: ${{desire.mental}}`);
      if (step?.self_belief) lines.push(`self belief: ${{step.self_belief}}`);
      if (goals.length) {{
        lines.push("work_goal:");
        for (const goal of goals) {{
          lines.push(`- ${{goal.completed ? "done" : "open"}}: ${{goal.text || ""}}`);
        }}
      }}
      const update = step?.desire_update || {{}};
      if (Object.keys(update).length) {{
        lines.push(`update: ${{update.intent_status || ""}} ${{update.intent_text || ""}}`);
        if (update.benefit) lines.push(`benefit: ${{update.benefit}}`);
        if (update.cost) lines.push(`cost: ${{update.cost}}`);
        if (update.physiological_reason) lines.push(`physiological: ${{update.physiological_reason}}`);
        if (update.internal_state_reason) lines.push(`internal: ${{update.internal_state_reason}}`);
        if (update.mental_reason) lines.push(`mental update: ${{update.mental_reason}}`);
      }}
      return lines.join("\\n");
    }}
    function renderStateGauges(step) {{
      const root = document.getElementById("stateGauges");
      root.innerHTML = "";
      const desire = step?.desire_state || payload.desire?.current || {{}};
      const legacy = desire.self_care || null;
      const physiological = desire.physiological_state || legacy || {{}};
      const internal = desire.internal_state || (legacy ? {{ fatigue: legacy.fatigue }} : {{}});
      const scale = legacy && !desire.physiological_state ? 2 : 1;
      const values = [
        ["hunger", physiological.hunger, "physiological"],
        ["thirst", physiological.thirst, "physiological"],
        ["hygiene", physiological.hygiene, "physiological"],
        ["stress", internal.stress, "internal"],
        ["tension", internal.tension, "internal"],
        ["fatigue", internal.fatigue, "internal"],
      ];
      for (const [label, rawValue, group] of values) {{
        const missing = rawValue === undefined || rawValue === null;
        const value = missing ? 0 : Math.max(0, Math.min(10, (Number(rawValue) || 0) * scale));
        const row = document.createElement("div");
        row.className = `state-gauge ${{group}}`;
        const name = document.createElement("span");
        name.textContent = label;
        const track = document.createElement("div");
        track.className = "state-gauge-track";
        const fill = document.createElement("div");
        fill.className = "state-gauge-fill";
        fill.classList.add(missing ? "unknown" : value <= 3 ? "low" : value <= 6 ? "medium" : "high");
        fill.style.width = `${{value * 10}}%`;
        track.appendChild(fill);
        const valueText = document.createElement("span");
        valueText.className = "state-gauge-value";
        valueText.textContent = missing ? "?/10" : `${{value}}/10`;
        row.append(name, track, valueText);
        root.appendChild(row);
      }}
    }}
    function formatActorState(step) {{
      const state = step.actor_state || {{}};
      const update = step.actor_state_update || {{}};
      const lines = [];
      if (Object.keys(state).length) {{
        lines.push(`posture: ${{state.posture || ""}}`);
        lines.push(`body_surface: ${{state.body_surface || ""}}`);
        lines.push(`worn_items: ${{(state.worn_items || []).join(", ") || "none"}}`);
        lines.push(`gaze: ${{state.gaze_target || ""}}`);
        lines.push(`method: ${{state.interaction_method || ""}}`);
        if ((state.interaction_elements || []).length) {{
          lines.push(`elements: ${{state.interaction_elements.map(item => item.name || item.id || "").filter(Boolean).join(", ")}}`);
        }}
        if (state.text_to_motion_description) {{
          lines.push("");
          lines.push(`motion: ${{state.text_to_motion_description}}`);
        }}
      }} else {{
        lines.push("(no actor state)");
      }}
      if (Object.keys(update).length) {{
        lines.push("");
        lines.push("update:");
        lines.push(JSON.stringify(update, null, 2));
      }}
      return lines.join("\\n");
    }}
    function areaNameForPosition(position) {{
      if (!Array.isArray(position) || position.length < 2) return "unknown area";
      const [x, y] = position;
      for (const area of payload.home.areas || []) {{
        const [x1, y1, x2, y2] = area.bounds;
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) return area.name;
      }}
      return "unknown area";
    }}
    function formatTraceState(step, position) {{
      const state = step?.actor_state || {{}};
      const desire = step?.desire_state || {{}};
      const physiological = desire.physiological_state || desire.self_care || {{}};
      const internal = desire.internal_state || {{ fatigue: desire.self_care?.fatigue }};
      const point = position || step?.to_position || payload.initial_position;
      const lines = [
        `location: ${{areaNameForPosition(point)}} (${{Number(point?.[0] || 0).toFixed(2)}}, ${{Number(point?.[1] || 0).toFixed(2)}})`,
      ];
      if (state.posture) lines.push(`posture: ${{state.posture}}`);
      if (state.gaze_target) lines.push(`gaze: ${{state.gaze_target}}`);
      if ((state.worn_items || []).length) lines.push(`worn: ${{state.worn_items.join(", ")}}`);
      if ((state.interaction_elements || []).length) {{
        const names = state.interaction_elements.map(item => item.name || item.id || item).filter(Boolean);
        if (names.length) lines.push(`interacting: ${{names.join(", ")}}`);
      }}
      const numeric = [
        ["hunger", physiological.hunger],
        ["thirst", physiological.thirst],
        ["hygiene", physiological.hygiene],
        ["stress", internal.stress],
        ["tension", internal.tension],
        ["fatigue", internal.fatigue],
      ].filter(([, value]) => value !== undefined && value !== null);
      if (numeric.length) lines.push(numeric.map(([name, value]) => `${{name}} ${{value}}/10`).join(" · "));
      if (desire.mental) lines.push(`mental: ${{desire.mental}}`);
      if (step?.self_belief) lines.push(`self belief: ${{step.self_belief}}`);
      return lines.join("\\n");
    }}
    function renderStateActionTrace() {{
      const list = document.getElementById("stateActionList");
      list.innerHTML = "";
      document.getElementById("traceTurnCount").textContent = `${{payload.steps.length}} action turns`;
      payload.steps.forEach((step, index) => {{
        const previous = index === 0
          ? {{
              to_position: payload.initial_position,
              desire_state: payload.desire?.initial || {{}},
              actor_state: {{}},
              self_belief: "",
            }}
          : payload.steps[index - 1];
        const row = document.createElement("article");
        row.className = "state-action-row";

        const turnCell = document.createElement("div");
        turnCell.className = "trace-cell";
        const turn = document.createElement("div");
        turn.className = "trace-turn";
        turn.textContent = `Turn ${{step.turn_index || index + 1}}`;
        const meta = document.createElement("div");
        meta.className = "trace-meta";
        meta.textContent = [step.time_text, step.intent_status].filter(Boolean).join(" · ");
        turnCell.append(turn, meta);

        const stateCell = document.createElement("div");
        stateCell.className = "trace-cell";
        const stateLabel = document.createElement("div");
        stateLabel.className = "trace-label";
        stateLabel.textContent = "State before action";
        const stateText = document.createElement("div");
        stateText.className = "trace-state";
        stateText.textContent = formatTraceState(previous, previous.to_position);
        stateCell.append(stateLabel, stateText);

        const actionCell = document.createElement("div");
        actionCell.className = "trace-cell";
        const source = document.createElement("span");
        source.className = `timeline-source trace-source ${{step.control_source || "goal_directed"}}`;
        source.textContent = step.control_label || "Goal-Directed";
        const action = document.createElement("div");
        action.className = "trace-action";
        action.textContent = step.proposal || step.action_text || "(no action)";
        const result = document.createElement("div");
        result.className = "trace-result";
        result.textContent = `World result: ${{step.world_summary || step.feedback || "(none)"}}`;
        actionCell.append(source, action, result);

        row.append(turnCell, stateCell, actionCell);
        list.appendChild(row);
      }});
      const finalStep = payload.steps[payload.steps.length - 1];
      const finalState = finalStep || {{
        to_position: payload.initial_position,
        desire_state: payload.desire?.current || payload.desire?.initial || {{}},
        actor_state: {{}},
      }};
      document.getElementById("finalStateText").textContent = formatTraceState(finalState, finalState.to_position);
    }}
    function renderWorldErrors() {{
      const report = payload.world_failure_report || {{}};
      const failures = Array.isArray(report.failures) ? report.failures : [];
      const failureCount = Number.isFinite(Number(report.failure_count))
        ? Number(report.failure_count)
        : failures.length;
      const actionTurns = Number.isFinite(Number(report.total_action_turns))
        ? Number(report.total_action_turns)
        : payload.steps.length;
      const moduleCounts = report.failure_counts_by_module || {{}};
      document.getElementById("worldFailureCount").textContent = `${{failureCount}} failures`;
      document.getElementById("failureActionTurns").textContent = String(actionTurns);
      document.getElementById("failureTotal").textContent = String(failureCount);
      document.getElementById("failureModuleCount").textContent = String(Object.keys(moduleCounts).length);

      const runError = document.getElementById("runErrorText");
      const runErrorText = String(report.run_error || "").trim();
      runError.hidden = !runErrorText;
      runError.textContent = runErrorText ? `Run error:\n${{runErrorText}}` : "";

      const list = document.getElementById("worldFailureList");
      const empty = document.getElementById("worldFailureEmpty");
      list.innerHTML = "";
      empty.hidden = failures.length > 0 || Boolean(runErrorText);
      failures.forEach((failure, index) => {{
        const item = document.createElement("article");
        item.className = "failure-item";

        const metaCell = document.createElement("div");
        metaCell.className = "trace-cell";
        const title = document.createElement("div");
        title.className = "trace-turn";
        title.textContent = `Failure ${{failure.failure_index || index + 1}}`;
        const kind = document.createElement("div");
        kind.className = "failure-kind";
        kind.textContent = failure.failed_module || failure.failure_kind || "unknown module";
        const meta = document.createElement("div");
        meta.className = "trace-meta";
        meta.textContent = [
          failure.turn_index ? `turn ${{failure.turn_index}}` : "",
          failure.world_route || "",
          failure.execution_kind || "",
        ].filter(Boolean).join(" · ");
        metaCell.append(title, kind, meta);

        const actionCell = document.createElement("div");
        actionCell.className = "trace-cell";
        const actionLabel = document.createElement("div");
        actionLabel.className = "trace-label";
        actionLabel.textContent = "Action and Intent";
        const actionText = document.createElement("div");
        actionText.className = "failure-text";
        const intent = failure.intent || {{}};
        actionText.textContent = [
          failure.action_proposal || "(no action proposal)",
          intent.text ? `Intent: ${{intent.text}}` : "",
        ].filter(Boolean).join("\\n\\n");
        actionCell.append(actionLabel, actionText);

        const errorCell = document.createElement("div");
        errorCell.className = "trace-cell";
        const errorLabel = document.createElement("div");
        errorLabel.className = "trace-label";
        errorLabel.textContent = "Error and World Feedback";
        const errorText = document.createElement("div");
        errorText.className = "failure-text";
        errorText.textContent = [
          failure.error ? `Error: ${{failure.error}}` : "",
          failure.world_feedback ? `Feedback: ${{failure.world_feedback}}` : "",
        ].filter(Boolean).join("\\n\\n") || "No diagnostic text.";
        const details = document.createElement("details");
        details.className = "failure-details";
        const summary = document.createElement("summary");
        summary.textContent = "Pipeline diagnostics";
        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify({{
          world_judgments: failure.world_judgments || {{}},
          contextual_world_trace: failure.contextual_world_trace || [],
          graph_transition_report: failure.graph_transition_report || {{}},
          actor_state: failure.actor_state || {{}},
          final_position: failure.final_position || [],
        }}, null, 2);
        details.append(summary, pre);
        errorCell.append(errorLabel, errorText, details);

        item.append(metaCell, actionCell, errorCell);
        list.appendChild(item);
      }});
    }}
    function setStep(index) {{
      if (!payload.steps.length) return;
      current = Math.max(0, Math.min(index, payload.steps.length - 1));
      const step = payload.steps[current];
      stepRange.value = current;
      const timeLabel = step.time_text ? ` / ${{step.time_text}}` : "";
      const stageLabel = step.counts_as_intent_action
        ? `Intent Action ${{step.action_index}}`
        : `Positioning for Intent Action ${{step.action_index}}`;
      document.getElementById("stepLabel").textContent = `World Step ${{step.turn_index || current + 1}} / Engine Step ${{step.step_id}} / ${{stageLabel}}${{timeLabel}}`;
      document.getElementById("statusLabel").textContent = step.intent_status || "active";
      document.getElementById("intentText").textContent = step.intent_text || "";
      renderStateGauges(step);
      document.getElementById("stateText").textContent = formatState(step);
      document.getElementById("proposalText").textContent = step.proposal || "(no action proposal)";
      document.getElementById("actionType").textContent = `${{step.control_label || "Goal-Directed"}} · ${{step.action_type || "unknown"}}`;
      document.getElementById("feedbackText").textContent = step.feedback || step.action_text || "";
      document.getElementById("durationText").textContent = step.estimated_duration || "(not estimated)";
      document.getElementById("actorStateText").textContent = formatActorState(step);
      speech.textContent = step.feedback || step.proposal || "";
      for (const node of elementNodes.values()) node.setAttribute("class", "element");
      for (const id of step.target_ids || []) {{
        const node = elementNodes.get(id);
        if (node) node.setAttribute("class", "element target");
      }}
      for (const change of step.changes || []) {{
        const node = elementNodes.get(change.element_id);
        if (node) node.setAttribute("class", "element changed");
      }}
      const path = [payload.initial_position, ...payload.steps.slice(0, current + 1).map(item => item.to_position)];
      document.getElementById("activePath").setAttribute("points", pointString(path));
      setAgent(step);
      const changesText = document.getElementById("changesText");
      changesText.innerHTML = "";
      if (!step.changes.length) {{
        changesText.appendChild(Object.assign(document.createElement("div"), {{ className: "empty", textContent: "none" }}));
      }} else {{
        for (const change of step.changes) {{
          const row = document.createElement("div");
          row.textContent = `${{change.element_name}}.${{change.field}}: ${{change.old}} -> ${{change.new}}`;
          changesText.appendChild(row);
        }}
      }}
      const temporaryElementsText = document.getElementById("temporaryElementsText");
      temporaryElementsText.innerHTML = "";
      if (!(step.temporary_element_changes || []).length) {{
        temporaryElementsText.appendChild(Object.assign(document.createElement("div"), {{ className: "empty", textContent: "none" }}));
      }} else {{
        for (const change of step.temporary_element_changes) {{
          const row = document.createElement("div");
          row.textContent = change.change_type === "created"
            ? `+ ${{change.temporary_element_id}} ${{change.name}} [${{change.new_status}}]`
            : `~ ${{change.temporary_element_id}} ${{change.name}}: ${{change.old_status}} -> ${{change.new_status}}`;
          temporaryElementsText.appendChild(row);
        }}
      }}
      [...timeline.children].forEach((button, idx) => button.classList.toggle("active", idx === current));
    }}
    function play() {{
      if (timer) clearInterval(timer);
      playing = true;
      playBtn.textContent = "Ⅱ";
      timer = setInterval(() => {{
        if (current >= payload.steps.length - 1) {{
          pause();
          return;
        }}
        setStep(current + 1);
      }}, Number(speedSelect.value));
    }}
    function pause() {{
      playing = false;
      playBtn.textContent = "▶";
      if (timer) clearInterval(timer);
      timer = null;
    }}
    playBtn.addEventListener("click", () => playing ? pause() : play());
    prevBtn.addEventListener("click", () => setStep(current - 1));
    stepRange.addEventListener("input", event => setStep(Number(event.target.value)));
    speedSelect.addEventListener("change", () => {{ if (playing) play(); }});
    function setSideWidth(widthPx) {{
      const min = 860;
      const max = Math.max(min, Math.min(1120, window.innerWidth - 560));
      const width = Math.max(min, Math.min(max, widthPx));
      app.style.gridTemplateColumns = `minmax(560px, 1fr) 8px ${{width}}px`;
      localStorage.setItem("intentReplaySideWidth", String(width));
    }}
    const savedSideWidth = Number(localStorage.getItem("intentReplaySideWidth"));
    if (Number.isFinite(savedSideWidth) && savedSideWidth > 0 && window.innerWidth > 1460) {{
      setSideWidth(savedSideWidth);
    }}
    function setTimelineWidth(widthPx) {{
      if (window.innerWidth <= 860) return;
      const sideRect = side.getBoundingClientRect();
      const minTimeline = 460;
      const minDetails = 320;
      const maxTimeline = Math.max(minTimeline, sideRect.width - minDetails - 76);
      const width = Math.max(minTimeline, Math.min(maxTimeline, widthPx));
      side.style.gridTemplateColumns = `minmax(${{minDetails}}px, 1fr) 8px ${{width}}px`;
      localStorage.setItem("intentReplayTimelineWidth", String(width));
    }}
    const savedTimelineWidth = Number(localStorage.getItem("intentReplayTimelineWidth"));
    if (Number.isFinite(savedTimelineWidth) && savedTimelineWidth > 0 && window.innerWidth > 860) {{
      setTimelineWidth(savedTimelineWidth);
    }}
    layoutResizer.addEventListener("pointerdown", event => {{
      if (window.innerWidth <= 1460) return;
      event.preventDefault();
      layoutResizer.classList.add("dragging");
      layoutResizer.setPointerCapture(event.pointerId);
    }});
    layoutResizer.addEventListener("pointermove", event => {{
      if (!layoutResizer.classList.contains("dragging")) return;
      setSideWidth(window.innerWidth - event.clientX);
    }});
    function stopResize(event) {{
      if (!layoutResizer.classList.contains("dragging")) return;
      layoutResizer.classList.remove("dragging");
      try {{ layoutResizer.releasePointerCapture(event.pointerId); }} catch (_) {{}}
    }}
    layoutResizer.addEventListener("pointerup", stopResize);
    layoutResizer.addEventListener("pointercancel", stopResize);
    sideSplitResizer.addEventListener("pointerdown", event => {{
      if (window.innerWidth <= 860) return;
      event.preventDefault();
      sideSplitResizer.classList.add("dragging");
      sideSplitResizer.setPointerCapture(event.pointerId);
    }});
    sideSplitResizer.addEventListener("pointermove", event => {{
      if (!sideSplitResizer.classList.contains("dragging")) return;
      const sideRect = side.getBoundingClientRect();
      setTimelineWidth(sideRect.right - event.clientX - 18);
    }});
    function stopTimelineResize(event) {{
      if (!sideSplitResizer.classList.contains("dragging")) return;
      sideSplitResizer.classList.remove("dragging");
      try {{ sideSplitResizer.releasePointerCapture(event.pointerId); }} catch (_) {{}}
    }}
    sideSplitResizer.addEventListener("pointerup", stopTimelineResize);
    sideSplitResizer.addEventListener("pointercancel", stopTimelineResize);
    renderCumulativeReward();
    drawScene();
    renderTimeline();
    renderStateActionTrace();
    renderWorldErrors();
    if (payload.steps.length) setStep(0);
  </script>
</body>
</html>
"""


def write_intent_run_html(
    home: Home,
    *,
    initial_position: tuple[float, float],
    results: list[EngineStepResult],
    output_path: str | Path = OUTPUT_HTML,
    desire_state: dict | None = None,
    initial_desire_state: dict | None = None,
    desire_updates: list[dict] | None = None,
    cumulative_reward: dict | None = None,
    world_failure_report: dict | None = None,
) -> Path:
    payload = build_payload(
        home,
        initial_position=initial_position,
        results=results,
        desire_state=desire_state,
        initial_desire_state=initial_desire_state,
        desire_updates=desire_updates,
        cumulative_reward=cumulative_reward,
        world_failure_report=world_failure_report,
    )
    path = Path(output_path)
    path.write_text(build_html(payload), encoding="utf-8")
    return path

from __future__ import annotations

from html import escape
from pathlib import Path

from engine.interaction_engine import EngineStepResult
from contextual_world.world_old.spatial_rules.physics_rules import get_physics_profile
from contextual_world.structure.scene_schema import Home
from contextual_world.structure.space_layout import PORTAL_DEFINITIONS


OUTPUT_SVG = "latest_main_run.svg"
SCALE = 90
PADDING = 30

AREA_COLORS = {
    "kitchen": "#f6d8ae",
    "dining_room": "#f3e6b3",
    "bedroom": "#d6e8c8",
    "entryway": "#e3d5ca",
    "living_room": "#f5deb3",
    "bathroom": "#cfe8f7",
    "main_balcony": "#d8ead3",
    "utility_balcony": "#d9ead3",
}


def to_px(value: float) -> float:
    return value * SCALE + PADDING


def rect_svg(x: float, y: float, w: float, h: float, fill: str, stroke: str, stroke_width: int = 2, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}"/>'
    )


def text_svg(x: float, y: float, text: str, size: int = 12, fill: str = "#222") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-family="Helvetica, Arial, sans-serif">{escape(text)}</text>'
    )


def line_svg(x1: float, y1: float, x2: float, y2: float, stroke: str, stroke_width: int = 2, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" stroke-linecap="round"{dash_attr}/>'
    )


def circle_svg(cx: float, cy: float, r: float, fill: str, stroke: str, stroke_width: int = 2) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def polyline_svg(points: list[tuple[float, float]], stroke: str, stroke_width: int = 4, opacity: float = 0.95) -> str:
    pts = " ".join(f"{to_px(x):.1f},{to_px(y):.1f}" for x, y in points)
    return (
        f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"/>'
    )


def build_run_svg(home: Home, *, initial_position: tuple[float, float], results: list[EngineStepResult]) -> str:
    width = to_px(9.4) + PADDING
    height = to_px(10.0) + PADDING
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        rect_svg(0, 0, width, height, "#faf7f2", "#ddd", 1),
    ]

    for area in home.areas:
        x_min, y_min, x_max, y_max = area.bounds
        fill = AREA_COLORS.get(area.node_id, "#eee")
        parts.append(
            rect_svg(
                to_px(x_min),
                to_px(y_min),
                (x_max - x_min) * SCALE,
                (y_max - y_min) * SCALE,
                fill,
                "#666",
                2,
                0.68,
            )
        )
        parts.append(text_svg(to_px(x_min) + 8, to_px(y_min) + 18, area.node_id, 13, "#333"))

        for element in area.elements:
            center_x, center_y = element.center
            size_x, size_y = element.size
            x = to_px(center_x - size_x / 2)
            y = to_px(center_y - size_y / 2)
            w = size_x * SCALE
            h = size_y * SCALE
            profile = get_physics_profile(element)
            fill_color = "#8d6e63" if profile.is_blocking else "#aed581"
            stroke = "#3e2723" if profile.is_blocking else "#33691e"
            parts.append(rect_svg(x, y, w, h, fill_color, stroke, 1, 0.8))

    for portal in PORTAL_DEFINITIONS:
        (x1, y1), (x2, y2) = portal["segment"]
        kind = portal["kind"]
        stroke = "#2c7be5" if kind == "glass_door" else "#d64545" if kind == "door" else "#7a5c2e"
        width = 6 if kind != "opening" else 4
        dash = "8 5" if kind == "opening" else None
        parts.append(line_svg(to_px(x1), to_px(y1), to_px(x2), to_px(y2), stroke, width, dash))

    path_points = [initial_position, *[result.final_position for result in results]]
    if len(path_points) >= 2:
        parts.append(polyline_svg(path_points, "#ff7a00", 5, 0.9))

    start_x, start_y = initial_position
    parts.append(circle_svg(to_px(start_x), to_px(start_y), 8, "#1f9d55", "#145a32", 2))
    parts.append(text_svg(to_px(start_x) + 10, to_px(start_y) - 10, "start", 12, "#145a32"))

    for result in results:
        x, y = result.final_position
        parts.append(circle_svg(to_px(x), to_px(y), 7, "#fff4d6", "#c96a00", 2))
        parts.append(text_svg(to_px(x) + 8, to_px(y) - 8, str(result.step_id), 12, "#8a4b08"))

    legend_x = width - 240
    legend_y = 28
    parts.append(rect_svg(legend_x, legend_y, 210, 78, "#fffaf4", "#d7c7b3", 1, 0.95))
    parts.append(text_svg(legend_x + 12, legend_y + 22, "Main Run Movement", 14, "#3a2a1a"))
    parts.append(line_svg(legend_x + 14, legend_y + 40, legend_x + 70, legend_y + 40, "#ff7a00", 5))
    parts.append(text_svg(legend_x + 80, legend_y + 44, "movement path", 12, "#5a4630"))
    parts.append(circle_svg(legend_x + 22, legend_y + 62, 7, "#1f9d55", "#145a32", 2))
    parts.append(text_svg(legend_x + 38, legend_y + 66, "start / step markers", 12, "#5a4630"))

    parts.append("</svg>")
    return "\n".join(parts)


def write_run_svg(
    home: Home,
    *,
    initial_position: tuple[float, float],
    results: list[EngineStepResult],
    output_path: str | Path = OUTPUT_SVG,
) -> Path:
    path = Path(output_path)
    path.write_text(build_run_svg(home, initial_position=initial_position, results=results), encoding="utf-8")
    return path

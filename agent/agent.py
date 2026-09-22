from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import atan2, degrees, hypot, inf

from agent.belief.short_time_memory import ShortTermMemory
from agent.belief.self_state import SelfState
from agent.belief.long_term_memory import LongTermMemory
from agent.belief.time_belief import TimeBelief
from agent.belief.spatial_memory.spatial_belief import SpatialBelief
from agent.desire import DesireState
from agent.habitual_controller.cue_extraction import ContextCueBuffer, CueExtractionResult
from datetime import datetime
from agent.intent.state import IntentState
from agent.perceive import (
    PerceiveResult,
    initialize_spatial_belief,
    perceive_agent,
    perceive_narrate_output as do_perceive_narrate_output,
)
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine
from contextual_world.world_old.spatial_rules.physics_types import MoveResult


DEFAULT_AGENT_WORN_ITEMS = ["jacket"]


@dataclass
class RoutePlan:
    success: bool
    reason: str
    start_point: tuple[float, float] | None
    goal_point: tuple[float, float] | None
    raw_path: list[tuple[float, float]]
    smooth_path: list[tuple[float, float]]
    from_element_id: str | None = None
    to_element_id: str | None = None


@dataclass
class Agent:
    node_id: str
    name: str
    center: tuple[float, float]
    personality: str = ""
    degree_of_model_based_control: float = 0.5
    size: tuple[float, float] = (0.35, 0.35)
    facing: float = 90.0
    field_of_view_degrees: float = 160.0
    step_size: float = 0.25
    posture: str = "standing"
    interaction_elements: list[dict] = field(default_factory=list)
    interaction_method: str = ""
    gaze_target: str = ""
    worn_items: list[str] = field(default_factory=lambda: list(DEFAULT_AGENT_WORN_ITEMS))
    body_surface: str = "dry_clean"
    text_to_motion_description: str = ""
    last_world_feedback_summary: str = ""
    current_area_id: str | None = None
    belief: SpatialBelief | None = None
    short_time_memory: ShortTermMemory = field(default_factory=lambda: ShortTermMemory(intent_text=""))
    self_state: SelfState = field(default_factory=SelfState)
    time_belief: TimeBelief = field(default_factory=TimeBelief)
    desire_state: DesireState = field(default_factory=DesireState)
    base_desire_state: DesireState | None = None
    base_world_agent_state: dict | None = None
    long_term_memory: LongTermMemory = field(default_factory=lambda: LongTermMemory(agent_name=""))
    perception_step: int = 0
    cue_extraction_buffer: ContextCueBuffer = field(default_factory=ContextCueBuffer)
    latest_cue_extraction: CueExtractionResult | None = None
    habitual_last_executed_at: dict[str, datetime] = field(default_factory=dict)
    habitual_last_considered_at: dict[str, datetime] = field(default_factory=dict)
    visited_area_ids: set[str] = field(default_factory=set)
    active_intent: IntentState | None = None

    def __post_init__(self) -> None:
        self.degree_of_model_based_control = min(
            1.0,
            max(0.0, float(self.degree_of_model_based_control)),
        )
        if not self.long_term_memory.agent_name:
            self.long_term_memory.agent_name = self.name
        default_desire = DesireState()
        if self.desire_state.to_dict() == default_desire.to_dict():
            self.desire_state = DesireState.for_agent(self.name)
        if self.base_desire_state is None:
            self.base_desire_state = deepcopy(self.desire_state)
        if self.base_world_agent_state is None:
            self.base_world_agent_state = self.world_agent_state()

    def normalize_angle(self, angle: float) -> float:
        return angle % 360.0

    def update_facing(self, dx: float, dy: float) -> None:
        if dx == 0 and dy == 0:
            return
        self.facing = self.normalize_angle(degrees(atan2(dy, dx)))

    def can_move_to(
        self,
        engine: PhysicsEngine,
        target_center: tuple[float, float],
    ) -> MoveResult:
        return engine.can_place_box(new_center=target_center, size=self.size)

    def move_by(
        self,
        engine: PhysicsEngine,
        dx: float,
        dy: float,
    ) -> MoveResult:
        target_center = (self.center[0] + dx, self.center[1] + dy)
        result = self.can_move_to(engine, target_center)
        if result.success:
            self.update_facing(dx, dy)
            self.center = target_center
        return result

    def move_to(
        self,
        engine: PhysicsEngine,
        target_center: tuple[float, float],
    ) -> MoveResult:
        result = self.can_move_to(engine, target_center)
        if result.success:
            self.update_facing(target_center[0] - self.center[0], target_center[1] - self.center[1])
            self.center = target_center
        return result

    def _snap(self, point: tuple[float, float], resolution: float) -> tuple[float, float]:
        x, y = point
        return (round(x / resolution) * resolution, round(y / resolution) * resolution)

    def _distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return hypot(a[0] - b[0], a[1] - b[1])

    def _neighbors(self, point: tuple[float, float], resolution: float) -> list[tuple[tuple[float, float], float]]:
        diagonals = resolution
        return [
            ((round(point[0] + resolution, 4), round(point[1], 4)), resolution),
            ((round(point[0] - resolution, 4), round(point[1], 4)), resolution),
            ((round(point[0], 4), round(point[1] + resolution, 4)), resolution),
            ((round(point[0], 4), round(point[1] - resolution, 4)), resolution),
            ((round(point[0] + resolution, 4), round(point[1] + resolution, 4)), diagonals * 1.4142),
            ((round(point[0] + resolution, 4), round(point[1] - resolution, 4)), diagonals * 1.4142),
            ((round(point[0] - resolution, 4), round(point[1] + resolution, 4)), diagonals * 1.4142),
            ((round(point[0] - resolution, 4), round(point[1] - resolution, 4)), diagonals * 1.4142),
        ]

    def _a_star(
        self,
        engine: PhysicsEngine,
        start: tuple[float, float],
        goal: tuple[float, float],
        *,
        resolution: float = 0.25,
        max_expansions: int = 12000,
    ) -> list[tuple[float, float]]:
        start = self._snap(start, resolution)
        goal = self._snap(goal, resolution)
        if start == goal:
            return [start]
        if not engine.can_place_box(new_center=goal, size=self.size).success:
            return []

        open_heap: list[tuple[float, int, tuple[float, float]]] = []
        heappush(open_heap, (0.0, 0, start))
        came_from: dict[tuple[float, float], tuple[float, float] | None] = {start: None}
        g_score: dict[tuple[float, float], float] = {start: 0.0}
        counter = 0
        expansions = 0

        while open_heap and expansions < max_expansions:
            _, _, current = heappop(open_heap)
            expansions += 1
            if current == goal:
                path = [current]
                while came_from[current] is not None:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor, step_cost in self._neighbors(current, resolution):
                check = engine.can_place_box(new_center=neighbor, size=self.size)
                if not check.success:
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g >= g_score.get(neighbor, inf):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                counter += 1
                priority = tentative_g + self._distance(neighbor, goal)
                heappush(open_heap, (priority, counter, neighbor))

        return []

    def _segment_clear(
        self,
        engine: PhysicsEngine,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        sample_step: float = 0.12,
    ) -> bool:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = self._distance(start, end)
        if distance == 0:
            return True
        steps = max(1, int(distance / sample_step))
        for index in range(1, steps + 1):
            t = index / steps
            point = (start[0] + dx * t, start[1] + dy * t)
            if not engine.can_place_box(new_center=point, size=self.size).success:
                return False
        return True

    def smooth_path(
        self,
        engine: PhysicsEngine,
        path: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if len(path) <= 2:
            return path[:]
        smoothed = [path[0]]
        anchor_index = 0
        while anchor_index < len(path) - 1:
            next_index = len(path) - 1
            while next_index > anchor_index + 1:
                if self._segment_clear(engine, path[anchor_index], path[next_index]):
                    break
                next_index -= 1
            smoothed.append(path[next_index])
            anchor_index = next_index
        return smoothed

    def find_path(
        self,
        engine: PhysicsEngine,
        target_center: tuple[float, float],
        *,
        resolution: float = 0.25,
        max_expansions: int = 12000,
        smooth: bool = True,
    ) -> list[tuple[float, float]]:
        raw = self._a_star(
            engine,
            self.center,
            target_center,
            resolution=resolution,
            max_expansions=max_expansions,
        )
        if not raw:
            return []
        return self.smooth_path(engine, raw) if smooth else raw

    def initialize_belief(self, engine: PhysicsEngine) -> None:
        self.belief = initialize_spatial_belief(engine)
        area = engine.get_area_for_point(*self.center)
        self.current_area_id = area.node_id if area is not None else None

    def world_agent_state(self) -> dict:
        return {
            "facing": self.facing,
            "field_of_view_degrees": self.field_of_view_degrees,
            "posture": self.posture,
            "interaction_elements": deepcopy(self.interaction_elements),
            "interaction_method": self.interaction_method,
            "gaze_target": self.gaze_target,
            "worn_items": list(self.worn_items),
            "body_surface": self.body_surface,
            "text_to_motion_description": self.text_to_motion_description,
        }

    def apply_world_agent_state(self, state: dict | None, *, update_base: bool = False) -> None:
        if not isinstance(state, dict):
            return
        if "facing" in state and state["facing"] is not None:
            self.facing = self.normalize_angle(float(state["facing"]))
        if "field_of_view_degrees" in state and state["field_of_view_degrees"] is not None:
            self.field_of_view_degrees = min(360.0, max(0.1, float(state["field_of_view_degrees"])))
        if "posture" in state and str(state["posture"]).strip():
            self.posture = str(state["posture"]).strip()
        if "interaction_elements" in state and isinstance(state["interaction_elements"], list):
            self.interaction_elements = deepcopy(state["interaction_elements"])
        if "interaction_method" in state:
            self.interaction_method = str(state["interaction_method"] or "")
        if "gaze_target" in state:
            self.gaze_target = str(state["gaze_target"] or "")
        if "worn_items" in state and isinstance(state["worn_items"], list):
            self.worn_items = [str(item).strip() for item in state["worn_items"] if str(item).strip()]
        if "body_surface" in state and str(state["body_surface"]).strip():
            self.body_surface = str(state["body_surface"]).strip()
        if "text_to_motion_description" in state:
            self.text_to_motion_description = str(state["text_to_motion_description"] or "")
        if update_base:
            self.base_world_agent_state = self.world_agent_state()

    def reset_for_new_game(self, engine: PhysicsEngine) -> None:
        self.belief = None
        self.current_area_id = None
        self.perception_step = 0
        self.cue_extraction_buffer.clear()
        self.latest_cue_extraction = None
        self.habitual_last_executed_at.clear()
        self.habitual_last_considered_at.clear()
        self.posture = "standing"
        self.interaction_elements = []
        self.interaction_method = ""
        self.gaze_target = ""
        self.worn_items = list(DEFAULT_AGENT_WORN_ITEMS)
        self.body_surface = "dry_clean"
        self.text_to_motion_description = ""
        self.apply_world_agent_state(self.base_world_agent_state)
        self.last_world_feedback_summary = ""
        self.visited_area_ids.clear()
        self.active_intent = None
        self.short_time_memory = ShortTermMemory(intent_text="")
        self.self_state = SelfState()
        self.time_belief.reset()
        self.desire_state = deepcopy(self.base_desire_state) if self.base_desire_state is not None else DesireState.for_agent(self.name)
        self.long_term_memory.agent_name = self.name
        self.initialize_belief(engine)

    def perceive(self, engine: PhysicsEngine) -> PerceiveResult:
        if self.belief is None:
            self.initialize_belief(engine)
        return perceive_agent(self, engine)

    def perceive_narrate_output(
        self,
        engine: PhysicsEngine,
        *,
        use_llm: bool = False,
        provider_name: str = "ollama",
        model: str | None = None,
    ) -> str:
        return do_perceive_narrate_output(
            self,
            engine,
            use_llm=use_llm,
            provider_name=provider_name,
            model=model,
        )

    def follow_path(
        self,
        engine: PhysicsEngine,
        path: list[tuple[float, float]],
    ) -> bool:
        for point in path[1:]:
            result = self.move_to(engine, point)
            if not result.success:
                return False
        return True

    def element_perimeter_points(
        self,
        engine: PhysicsEngine,
        element_id: str,
        *,
        clearance: float = 0.20,
        resolution: float = 0.25,
    ) -> list[tuple[float, float]]:
        element = engine.get_element(element_id)
        if element is None:
            return []
        bounds = engine.get_element_bounds(element)
        half_w = self.size[0] / 2 + clearance
        half_h = self.size[1] / 2 + clearance
        cx, cy = bounds.center
        raw_points = [
            (cx, bounds.y_min - half_h),
            (cx, bounds.y_max + half_h),
            (bounds.x_min - half_w, cy),
            (bounds.x_max + half_w, cy),
            (bounds.x_min - half_w, bounds.y_min - half_h),
            (bounds.x_max + half_w, bounds.y_min - half_h),
            (bounds.x_min - half_w, bounds.y_max + half_h),
            (bounds.x_max + half_w, bounds.y_max + half_h),
        ]
        unique_points: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        for point in raw_points:
            snapped = self._snap(point, resolution)
            if snapped in seen:
                continue
            seen.add(snapped)
            if engine.can_place_box(new_center=snapped, size=self.size).success:
                unique_points.append(snapped)
        return unique_points

    def plan_between_elements(
        self,
        engine: PhysicsEngine,
        from_element_id: str,
        to_element_id: str,
        *,
        resolution: float = 0.25,
        clearance: float = 0.20,
        max_expansions: int = 12000,
    ) -> RoutePlan:
        start_candidates = self.element_perimeter_points(
            engine,
            from_element_id,
            clearance=clearance,
            resolution=resolution,
        )
        goal_candidates = self.element_perimeter_points(
            engine,
            to_element_id,
            clearance=clearance,
            resolution=resolution,
        )
        if not start_candidates:
            return RoutePlan(False, "no_start_candidates", None, None, [], [], from_element_id, to_element_id)
        if not goal_candidates:
            return RoutePlan(False, "no_goal_candidates", None, None, [], [], from_element_id, to_element_id)

        best_raw: list[tuple[float, float]] = []
        best_smooth: list[tuple[float, float]] = []
        best_start = None
        best_goal = None

        for start in start_candidates:
            for goal in goal_candidates:
                raw_path = self._a_star(
                    engine,
                    start,
                    goal,
                    resolution=resolution,
                    max_expansions=max_expansions,
                )
                if not raw_path:
                    continue
                smooth_path = self.smooth_path(engine, raw_path)
                if not best_smooth or self.path_length(smooth_path) < self.path_length(best_smooth):
                    best_raw = raw_path
                    best_smooth = smooth_path
                    best_start = start
                    best_goal = goal

        if not best_smooth:
            return RoutePlan(False, "no_route_found", None, None, [], [], from_element_id, to_element_id)

        return RoutePlan(
            True,
            "ok",
            best_start,
            best_goal,
            best_raw,
            best_smooth,
            from_element_id,
            to_element_id,
        )

    def apply_route_plan(self, engine: PhysicsEngine, plan: RoutePlan) -> bool:
        if not plan.success or not plan.smooth_path:
            return False
        self.center = plan.smooth_path[0]
        return self.follow_path(engine, plan.smooth_path)

    def path_length(self, path: list[tuple[float, float]]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for index in range(1, len(path)):
            total += self._distance(path[index - 1], path[index])
        return total

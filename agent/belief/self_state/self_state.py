from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from agent.belief.self_state.belief_state_update import update_belief_state


@dataclass
class ElementReference:
    element_id: str
    element_name: str
    area_id: str = ""
    area_name: str = ""

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "element_name": self.element_name,
            "area_id": self.area_id,
            "area_name": self.area_name,
        }


@dataclass
class MovementPoint:
    position: tuple[float, float]
    area_id: str = ""
    area_name: str = ""
    anchor_element_id: str = ""
    anchor_element_name: str = ""
    anchor_area_id: str = ""
    anchor_area_name: str = ""

    def to_dict(self) -> dict:
        return {
            "position": [round(self.position[0], 3), round(self.position[1], 3)],
            "area_id": self.area_id,
            "area_name": self.area_name,
            "anchor_element_id": self.anchor_element_id,
            "anchor_element_name": self.anchor_element_name,
            "anchor_area_id": self.anchor_area_id,
            "anchor_area_name": self.anchor_area_name,
        }


@dataclass
class MovementTrace:
    from_point: MovementPoint
    to_point: MovementPoint
    summary: str = ""

    @property
    def moved(self) -> bool:
        return hypot(
            self.to_point.position[0] - self.from_point.position[0],
            self.to_point.position[1] - self.from_point.position[1],
        ) > 0.05

    def to_dict(self) -> dict:
        return {
            "from": self.from_point.to_dict(),
            "to": self.to_point.to_dict(),
            "moved": self.moved,
            "summary": self.summary,
        }


@dataclass
class SelfStateUpdateInput:
    step_id: int
    intent_text: str
    action_proposal_text: str
    interacted_elements: list[ElementReference]
    environment_feedback: str
    movement: MovementTrace
    world_changes: list[dict] = field(default_factory=list)


@dataclass
class SelfStateActionRecord:
    step_id: int
    intent_text: str
    action_proposal_text: str
    environment_feedback: str
    movement: MovementTrace
    interacted_elements: list[ElementReference]
    inferred_markers: list[str]
    confidence: float
    conflicts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "intent_text": self.intent_text,
            "action_proposal_text": self.action_proposal_text,
            "environment_feedback": self.environment_feedback,
            "movement": self.movement.to_dict(),
            "interacted_elements": [element.to_dict() for element in self.interacted_elements],
            "inferred_markers": self.inferred_markers,
            "confidence": self.confidence,
            "conflicts": self.conflicts,
        }


@dataclass
class SelfState:
    """Post-action inferred belief about the agent's current activity state."""

    current_position: tuple[float, float] | None = None
    current_area_id: str = ""
    current_area_name: str = ""
    current_anchor_element_id: str = ""
    current_anchor_element_name: str = ""
    posture: str = "unknown"
    current_activity: str = "unknown"
    current_phase: str = "unknown"
    held_items: list[str] = field(default_factory=list)
    worn_items: list[str] = field(default_factory=list)
    engaged_elements: list[ElementReference] = field(default_factory=list)
    object_bindings: dict[str, str] = field(default_factory=dict)
    completed_markers: list[str] = field(default_factory=list)
    blocked_or_failed_markers: list[str] = field(default_factory=list)
    state_conflicts: list[str] = field(default_factory=list)
    self_belief: str = ""
    last_action: SelfStateActionRecord | None = None
    history: list[SelfStateActionRecord] = field(default_factory=list)

    def update(self, update_input: SelfStateUpdateInput) -> SelfStateActionRecord:
        movement = update_input.movement
        self.current_position = movement.to_point.position
        self.current_area_id = movement.to_point.area_id
        self.current_area_name = movement.to_point.area_name
        self.current_anchor_element_id = movement.to_point.anchor_element_id
        self.current_anchor_element_name = movement.to_point.anchor_element_name
        self.engaged_elements = update_input.interacted_elements[:]
        if self.current_area_name:
            self.object_bindings["current_area"] = self.current_area_name

        feedback = update_input.environment_feedback.strip()
        inferred_markers = self._infer_completed_markers(feedback)
        failed_markers = self._infer_failed_markers(feedback)
        conflicts = self._infer_conflicts(update_input)

        self._append_unique(self.completed_markers, inferred_markers)
        self._append_unique(self.blocked_or_failed_markers, failed_markers)
        self._append_unique(self.state_conflicts, conflicts)
        self._apply_marker_effects(inferred_markers, feedback)

        confidence = self._confidence(
            inferred_markers=inferred_markers,
            failed_markers=failed_markers,
            conflicts=conflicts,
            feedback=feedback,
        )
        record = SelfStateActionRecord(
            step_id=update_input.step_id,
            intent_text=update_input.intent_text,
            action_proposal_text=update_input.action_proposal_text,
            environment_feedback=feedback,
            movement=movement,
            interacted_elements=update_input.interacted_elements[:],
            inferred_markers=[*inferred_markers, *failed_markers],
            confidence=confidence,
            conflicts=conflicts,
        )
        self.last_action = record
        self.history.append(record)
        if len(self.history) > 30:
            self.history = self.history[-30:]
        return record

    def to_dict(self) -> dict:
        return {
            "action_proposal": (
                self.last_action.action_proposal_text
                if self.last_action is not None
                else ""
            ),
            "environment_feedback": (
                self.last_action.environment_feedback
                if self.last_action is not None
                else ""
            ),
            "movement": self._movement_dict(),
            "interacted_elements": [
                {
                    "id": element.element_id,
                    "name": element.element_name,
                }
                for element in self.engaged_elements
            ],
            "self_belief": self.self_belief,
        }

    def update_self_belief(
        self,
        *,
        agent_name: str,
        provider_name: str = "ollama",
        model: str | None = None,
    ) -> str:
        self.self_belief = update_belief_state(
            agent_name=agent_name,
            state_fields=self.to_dict_without_belief(),
            provider_name=provider_name,
            model=model,
        )
        return self.self_belief

    def to_trace_dict(self) -> dict:
        data = self.to_dict()
        data["inferred_state_trace"] = {
            "location": self._location_dict(),
            "posture": self.posture,
            "current_activity": self.current_activity,
            "current_phase": self.current_phase,
            "possessions": {
                "held": self.held_items,
                "worn": self.worn_items,
            },
            "object_bindings": dict(self.object_bindings),
            "completed_markers": self.completed_markers,
            "blocked_or_failed_markers": self.blocked_or_failed_markers,
            "uncertainties": self.state_conflicts,
            "last_action": self.last_action.to_dict() if self.last_action is not None else None,
        }
        return data

    def to_dict_without_belief(self) -> dict:
        data = self.to_dict()
        data.pop("self_belief", None)
        return data

    def _location_dict(self) -> dict:
        return {
            "position": (
                [round(self.current_position[0], 3), round(self.current_position[1], 3)]
                if self.current_position is not None
                else None
            ),
            "area": self.current_area_name or self.current_area_id,
            "anchor": self.current_anchor_element_name or self.current_anchor_element_id,
        }

    def _last_observation_dict(self) -> dict | None:
        if self.last_action is None:
            return None
        movement = self.last_action.movement
        return {
            "proposal": self.last_action.action_proposal_text,
            "feedback": self.last_action.environment_feedback,
            "movement": {
                "from": self._movement_label(movement.from_point),
                "to": self._movement_label(movement.to_point),
                "moved": movement.moved,
            },
            "markers": self.last_action.inferred_markers,
            "confidence": self.last_action.confidence,
        }

    def _movement_dict(self) -> dict:
        if self.last_action is None:
            return {
                "from": "",
                "to": self._location_label(),
                "moved": False,
            }
        movement = self.last_action.movement
        return {
            "from": self._movement_label(movement.from_point),
            "to": self._movement_label(movement.to_point),
            "moved": movement.moved,
        }

    def _location_label(self) -> str:
        area = self.current_area_name or self.current_area_id or "unknown_area"
        anchor = self.current_anchor_element_name or self.current_anchor_element_id or "no_anchor"
        if self.current_position is None:
            return f"[{area} / {anchor}]"
        position = f"({round(self.current_position[0], 3)}, {round(self.current_position[1], 3)})"
        return f"{position} [{area} / {anchor}]"

    def _movement_label(self, point: MovementPoint) -> str:
        area = point.area_name or point.area_id or "unknown_area"
        anchor = point.anchor_element_name or point.anchor_element_id or "no_anchor"
        position = f"({round(point.position[0], 3)}, {round(point.position[1], 3)})"
        return f"{position} [{area} / {anchor}]"

    def _apply_marker_effects(self, markers: list[str], feedback: str) -> None:
        if "seated_at_table" in markers or "Sit" in feedback:
            self.posture = "seated"
        elif self.posture == "unknown":
            self.posture = "standing"

        for marker in markers:
            if marker == "arrived_area":
                self.current_activity = "navigation"
                self.current_phase = "arrived_at_area"
                if self.current_area_name:
                    self.object_bindings["current_area"] = self.current_area_name
            elif marker == "fridge_opened":
                self.current_activity = "meal_preparation"
                self.current_phase = "checking_food_source"
                self.object_bindings["food_source"] = "fridge"
            elif marker == "food_acquired":
                self.current_activity = "meal_preparation"
                self.current_phase = "food_acquired"
                self._add_held_item("food")
                self.object_bindings["meal_item"] = "food_from_fridge"
                self.object_bindings["meal_location"] = "held_by_agent"
            elif marker == "food_in_microwave":
                self.current_activity = "meal_preparation"
                self.current_phase = "food_in_microwave"
                self._remove_held_item("food")
                self._remove_held_item("heated_food")
                self.object_bindings["meal_location"] = "microwave"
            elif marker == "microwave_time_set":
                self.current_activity = "meal_preparation"
                self.current_phase = "microwave_time_set"
            elif marker == "microwave_started":
                self.current_activity = "meal_preparation"
                self.current_phase = "heating_in_progress"
                self.object_bindings["meal_location"] = "microwave"
            elif marker == "heating_waiting":
                self.current_activity = "meal_preparation"
                self.current_phase = "waiting_for_heating"
            elif marker == "heated_food_taken_out":
                self.current_activity = "meal_preparation"
                self.current_phase = "heated_food_in_hand"
                self._remove_held_item("food")
                self._add_held_item("heated_food")
                self.object_bindings["meal_location"] = "held_by_agent"
            elif marker == "food_on_table":
                self.current_activity = "meal_preparation"
                self.current_phase = "food_on_table"
                self._remove_held_item("food")
                self._remove_held_item("heated_food")
                self.object_bindings["meal_location"] = "dining_table"
            elif marker == "seated_at_table":
                self.current_activity = "meal_preparation"
                self.current_phase = "seated_for_meal"
                self.object_bindings["body_location"] = "dining_table"
            elif marker == "tableware_acquired":
                self.current_activity = "meal_preparation"
                self.current_phase = "tableware_acquired"
                self._add_held_item("tableware")
                self.object_bindings["tableware_location"] = "held_by_agent"
            elif marker == "tableware_on_table":
                self.current_activity = "meal_preparation"
                self.current_phase = "tableware_on_table"
                self._remove_held_item("tableware")
                self.object_bindings["tableware_location"] = "dining_table"
            elif marker == "started_eating":
                self.current_activity = "eating"
                self.current_phase = "eating"
            elif marker == "outerwear_worn":
                self._add_worn_item("outerwear")

    def _infer_completed_markers(self, feedback: str) -> list[str]:
        rules = [
            ("arrived_area", ["walked to"]),
            ("fridge_opened", ["opened the refrigerator", "open the refrigerator"]),
            ("food_acquired", ["took out from the refrigerator", "took from the refrigerator", "take out food", "took out food"]),
            ("food_in_microwave", ["put into the microwave", "put into the microwave", "food put into the microwave", "Put food in the microwave"]),
            ("microwave_time_set", ["Set the microwave", "Time setting", "Setting button", "Heating time"]),
            ("microwave_started", ["Start button", "Press the microwave's start button", "Start the microwave"]),
            ("heating_waiting", ["Wait for the microwave", "Observe the remaining heating time", "Observe the heating", "Timer"]),
            ("heated_food_taken_out", ["Take out the heated food", "Take it out of the microwave", "Take it out of the open microwave"]),
            ("food_on_table", ["Brought to the dining table", "Bring to the dining table", "Place on the dining table", "Put on the dining table"]),
            ("seated_at_table", ["Sat down", "Sat at the dining table", "Sit at the dining table", "Sit beside the dining table"]),
            ("tableware_acquired", ["Pick up a set of cutlery", "Pick up the cutlery", "Took a set of cutlery"]),
            ("tableware_on_table", ["Cutlery placed on the dining table"]),
            ("started_eating", ["Start enjoying", "Start eating", "Put into mouth", "Start eating"]),
            ("outerwear_worn", ["Put on coat", "Wore it", "Worn on body"]),
        ]
        markers = []
        for marker, phrases in rules:
            if any(phrase in feedback for phrase in phrases):
                markers.append(marker)
        return markers

    def _infer_failed_markers(self, feedback: str) -> list[str]:
        if any(word in feedback for word in ["Cannot find", "No new empty spots", "Not clearly understood", "Did not reach", "Failed", "Cannot", "Unable to", "No way forward"]):
            return ["action_blocked_or_failed"]
        return []

    def _infer_conflicts(self, update_input: SelfStateUpdateInput) -> list[str]:
        conflicts = []
        movement = update_input.movement
        if (
            movement.to_point.area_id
            and movement.to_point.anchor_area_id
            and movement.to_point.anchor_area_id != movement.to_point.area_id
        ):
            conflicts.append(
                "movement anchor area differs from coordinate area: "
                f"{movement.to_point.anchor_element_name}({movement.to_point.anchor_area_name or movement.to_point.anchor_area_id}) vs "
                f"{movement.to_point.area_name or movement.to_point.area_id}"
            )
        if movement.to_point.area_id:
            for element in update_input.interacted_elements:
                if not element.area_id or element.area_id == movement.to_point.area_id:
                    continue
                if not movement.moved:
                    conflicts.append(
                        "interaction target area differs from agent coordinate area: "
                        f"{element.element_name}({element.area_name or element.area_id}) vs "
                        f"{movement.to_point.area_name or movement.to_point.area_id}"
                    )
        return conflicts

    def _confidence(
        self,
        *,
        inferred_markers: list[str],
        failed_markers: list[str],
        conflicts: list[str],
        feedback: str,
    ) -> float:
        if not feedback:
            return 0.2
        score = 0.65
        if inferred_markers:
            score += 0.2
        if failed_markers:
            score -= 0.1
        if conflicts:
            score -= 0.2
        return max(0.1, min(0.95, round(score, 2)))

    def _add_held_item(self, item: str) -> None:
        if item not in self.held_items:
            self.held_items.append(item)

    def _remove_held_item(self, item: str) -> None:
        self.held_items = [held for held in self.held_items if held != item]

    def _add_worn_item(self, item: str) -> None:
        if item not in self.worn_items:
            self.worn_items.append(item)

    def _append_unique(self, target: list[str], items: list[str]) -> None:
        for item in items:
            if item and item not in target:
                target.append(item)

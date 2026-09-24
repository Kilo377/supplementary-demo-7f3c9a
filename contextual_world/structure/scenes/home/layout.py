AREA_DEFINITIONS = [
    ("kitchen", "kitchen", (0.20, 0.35, 4.10, 3.60)),
    ("dining_room", "dining room", (4.10, 0.35, 6.50, 3.75)),
    ("bedroom", "bedroom", (6.55, 0.25, 9.15, 4.35)),
    ("entryway", "entrance (including outdoor platform)", (0.20, 3.95, 2.25, 9.75)),
    ("living_room", "living room (including central transition area)", (2.25, 3.60, 7.05, 8.55)),
    ("bathroom", "Bathroom", (7.05, 4.35, 9.15, 7.25)),
    ("main_balcony", "Main Balcony", (2.35, 8.55, 7.05, 9.75)),
    ("utility_balcony", "Laundry Balcony / Utility Balcony", (7.05, 7.25, 9.15, 9.75)),
]


HOME_DEFAULT_AGENT_START = (3.10, 5.10)  # living_room


RENDERING = {
    "area_colors": {
        "kitchen": "#f1d2a8",
        "dining_room": "#e6d994",
        "bedroom": "#bdd7b4",
        "entryway": "#d8c8bd",
        "living_room": "#e6c89b",
        "bathroom": "#afd6e6",
        "main_balcony": "#bcdab7",
        "utility_balcony": "#b7d8bf",
    }
}


# Doors and primary passages are defined as line segments.
# segment = ((x1, y1), (x2, y2))
# kind:
# - door: physical door
# - opening: open passage
# - glass_door: glass door
PORTAL_DEFINITIONS = [
    {
        "id": "front_door_portal",
        "name": "Entrance Door",
        "kind": "door",
        "areas": ("entryway", "outside"),
        "segment": ((0.20, 7.45), (0.20, 8.45)),
    },
    {
        "id": "entry_living_opening",
        "name": "Hallway to Living Room Passage",
        "kind": "opening",
        "areas": ("entryway", "living_room"),
        "segment": ((2.25, 5.10), (2.25, 8.10)),
    },
    {
        "id": "kitchen_dining_opening",
        "name": "Kitchen to Dining Room Passage",
        "kind": "opening",
        "areas": ("kitchen", "dining_room"),
        "segment": ((4.10, 1.60), (4.10, 3.10)),
    },
    {
        "id": "dining_living_opening",
        "name": "Dining Room to Living Room Passage",
        "kind": "opening",
        "areas": ("dining_room", "living_room"),
        "segment": ((4.10, 3.75), (6.40, 3.75)),
    },
    {
        "id": "bedroom_door_portal",
        "name": "Bedroom Door",
        "kind": "door",
        "areas": ("living_room", "bedroom"),
        "segment": ((6.78, 3.68), (6.78, 4.28)),
    },
    {
        "id": "bathroom_door_portal",
        "name": "Bathroom Door",
        "kind": "door",
        "areas": ("living_room", "bathroom"),
        "segment": ((7.05, 5.00), (7.05, 5.95)),
    },
    {
        "id": "bathroom_utility_opening",
        "name": "Bathroom to Utility Balcony Passage",
        "kind": "opening",
        "areas": ("bathroom", "utility_balcony"),
        "segment": ((7.45, 7.25), (8.75, 7.25)),
    },
    {
        "id": "living_main_balcony_glass_door",
        "name": "Living Room to Main Balcony Glass Door",
        "kind": "glass_door",
        "areas": ("living_room", "main_balcony"),
        "segment": ((3.40, 8.55), (5.40, 8.55)),
    },
]

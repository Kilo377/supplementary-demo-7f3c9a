AREA_DEFINITIONS = [
    ("storage_archive_area", "Document storage area", (0.20, 0.35, 3.35, 2.55)),
    ("main_workspace", "Main office area", (3.35, 0.35, 7.25, 4.65)),
    ("print_supply_area", "Printing and office supplies area", (7.25, 0.35, 9.45, 4.65)),
    ("lounge_area", "Rest and reception area", (0.20, 2.55, 3.35, 6.20)),
    ("meeting_area", "Small meeting area", (3.35, 4.65, 7.25, 7.15)),
    ("entryway", "Office entrance / entryway", (0.20, 6.20, 2.35, 9.75)),
    ("pantry_area", "Pantry area", (3.35, 7.15, 7.25, 9.75)),
    ("bathroom", "Bathroom", (7.25, 6.55, 9.45, 9.75)),
]


PORTAL_DEFINITIONS = [
    {
        "id": "office_front_door_portal",
        "name": "Office Entrance Door",
        "kind": "door",
        "areas": ("entryway", "outside"),
        "segment": ((0.75, 9.75), (1.85, 9.75)),
    },
    {
        "id": "entry_lounge_opening",
        "name": "Path from entrance to reception area",
        "kind": "opening",
        "areas": ("entryway", "lounge_area"),
        "segment": ((1.05, 6.20), (2.15, 6.20)),
    },
    {
        "id": "lounge_storage_opening",
        "name": "Path from reception area to document storage area",
        "kind": "opening",
        "areas": ("lounge_area", "storage_archive_area"),
        "segment": ((1.00, 2.55), (2.60, 2.55)),
    },
    {
        "id": "lounge_workspace_opening",
        "name": "Path from reception area to main office area",
        "kind": "opening",
        "areas": ("lounge_area", "main_workspace"),
        "segment": ((3.35, 2.95), (3.35, 4.15)),
    },
    {
        "id": "workspace_print_opening",
        "name": "Passage from the main office area to the printing area",
        "kind": "opening",
        "areas": ("main_workspace", "print_supply_area"),
        "segment": ((7.25, 1.45), (7.25, 3.75)),
    },
    {
        "id": "workspace_meeting_opening",
        "name": "Passage from the main office area to the meeting area",
        "kind": "opening",
        "areas": ("main_workspace", "meeting_area"),
        "segment": ((4.40, 4.65), (6.45, 4.65)),
    },
    {
        "id": "meeting_pantry_opening",
        "name": "Passage from the meeting area to the tea and snack area",
        "kind": "opening",
        "areas": ("meeting_area", "pantry_area"),
        "segment": ((4.10, 7.15), (6.80, 7.15)),
    },
    {
        "id": "pantry_bathroom_opening",
        "name": "Passage from the tea and snack area to the restroom",
        "kind": "door",
        "areas": ("pantry_area", "bathroom"),
        "segment": ((7.25, 7.35), (7.25, 8.65)),
    },
]


OFFICE_DEFAULT_AGENT_START = (1.30, 8.35)  # entryway


RENDERING = {
    "area_colors": {
        "storage_archive_area": "#d6b98f",
        "main_workspace": "#d8a96f",
        "print_supply_area": "#c6b7a4",
        "lounge_area": "#a8bf8f",
        "meeting_area": "#dbc17b",
        "entryway": "#cfc0b1",
        "pantry_area": "#d3a36d",
        "bathroom": "#8fc4d8",
    }
}

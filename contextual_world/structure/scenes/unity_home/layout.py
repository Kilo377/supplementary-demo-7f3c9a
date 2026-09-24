AREA_DEFINITIONS = [
    ("bedroom", "Bedroom / Computer Area", (0.25, 4.70, 5.60, 12.55)),
    ("bathroom", "Bathroom", (5.60, 0.25, 12.20, 5.75)),
    ("living_room", "Living room", (5.60, 5.75, 12.20, 14.56)),
    ("kitchen", "Kitchen / Dining Area", (12.20, 0.25, 18.22, 7.70)),
]


PORTAL_DEFINITIONS = [
    {
        "id": "bedroom_living_portal",
        "name": "Bedroom to Living Room Door",
        "kind": "door",
        "areas": ("bedroom", "living_room"),
        "segment": ((5.60, 6.35), (5.60, 7.50)),
    },
    {
        "id": "bathroom_living_portal",
        "name": "Bathroom to Living Room Door",
        "kind": "door",
        "areas": ("bathroom", "living_room"),
        "segment": ((9.05, 5.75), (10.30, 5.75)),
    },
    {
        "id": "kitchen_living_portal",
        "name": "Kitchen to Living Room Door",
        "kind": "door",
        "areas": ("kitchen", "living_room"),
        "segment": ((12.20, 5.95), (12.20, 7.10)),
    },
]


UNITY_HOME_DEFAULT_AGENT_START = (8.20, 8.20)  # living_room


RENDERING = {
    "area_colors": {
        "bedroom": "#b7a58b",
        "bathroom": "#8fb9c9",
        "living_room": "#b9c49b",
        "kitchen": "#d0a56f",
    }
}

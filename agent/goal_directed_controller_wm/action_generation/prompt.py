from __future__ import annotations

from agent.intent.state import IntentState
from agent.intuition import build_intuition_prompt
from agent.working_memory import WorkingMemoryFrame
from contextual_world.world_old.spatial_rules.physics_engine import PhysicsEngine


def build_action_generation_prompt(
    *,
    agent_name: str,
    intent: IntentState,
    working_memory: WorkingMemoryFrame,
    engine: PhysicsEngine,
    action_sample_window: int = 3,
) -> str:
    return build_intuition_prompt(
        agent_name=agent_name,
        intent=intent,
        working_memory=working_memory,
        engine=engine,
        action_sample_window=action_sample_window,
        include_think=False,
    )

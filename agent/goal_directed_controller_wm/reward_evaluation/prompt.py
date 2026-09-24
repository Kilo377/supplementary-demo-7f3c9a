from __future__ import annotations


def build_reward_evaluation_prompt(
    *,
    agent_name: str,
    intent_text: str,
    current_state_text: str,
    action_chains_text: str,
    personality_text: str = "",
) -> str:
    name = str(agent_name or "Agent").strip()
    sections = [
        f"You are {name}.",
        f"Your current intent:\n{intent_text.strip()}",
    ]
    _append_section(sections, "You consider yourself to be the kind of person who:", personality_text)
    _append_section(sections, "Your current state is:", current_state_text)
    sections.append(
        "Below are the action chains derived from forward simulation of several initial actions starting from the same current state:\n\n"
        + action_chains_text.strip()
    )
    sections.append(
        """
Please compare these action chains at once and judge the overall value of each initial action for {name}'s current intent.

The items being scored are the initial actions of each chain. The subsequent actions and predicted states are the potential consequences of that initial action.

Combine goal satisfaction, goal progress, final state, physical and psychological changes, time cost, effort, failure risk, uncertainty, information value, and alignment with {name}'s actual preferences into a single comprehensive score from 0 to 10. Do not output any sub-item scores.

Scoring anchors:
- 0: Clearly failed, severely deviated from the intent, or produced significant negative outcomes.
- 2: Barely progressed, with obvious costs, risks, or uncertainties.
- 5: Some progress made, but still far from the goal, or with high cost and uncertainty.
- 8: Goal basically satisfied, process reasonable, cost acceptable.
- 10: Fully satisfied the goal, with ideal results, process, and character state.

Requirements:
- All actions must use the same scale and be scored in a single comparison.
- Scores represent absolute value, not ranking. Even if an action is the best among candidates, if it is inherently poor, it should receive a low score.
- Do not force wide gaps between scores; multiple actions can receive the same score.
- Do not give higher scores simply because an action chain is longer, has more text, or is described in more detail.
- If the simulation ends due to reaching the iteration limit, score based on the progress already achieved; do not assume subsequent success.
- If the simulation relies on unconfirmed objects or states, factor this uncertainty into the total score.
- Each action_id must be output exactly once; do not add new actions.

Return only JSON, no explanation.

JSON schema:
{{
  "evaluations": [
    {{
      "action_id": "Action chain ID",
      "reward": 0,
      "reason": "Brief reason for this comprehensive score"
    }}
  ]
}}
""".strip()
    )
    return "\n\n".join(section for section in sections if section.strip()).strip()


def _append_section(sections: list[str], heading: str, content: str) -> None:
    cleaned = str(content or "").strip()
    if cleaned:
        sections.append(f"{heading}\n\n{cleaned}")

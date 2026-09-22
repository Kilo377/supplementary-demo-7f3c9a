import atexit
import json
import random
from pathlib import Path
from collections import Counter


def summarize_actions(actions):
    counts = Counter(item["control"] for item in actions)
    total = sum(counts[key] for key in ("habitual", "goal_directed", "combined"))
    return {
        "counts": dict(counts),
        "denominator": total,
        "definition": "non-positioning executed action attempts; failures included; combined separate",
        "habitual_ratio": counts["habitual"] / total if total else None,
        "goal_directed_ratio": counts["goal_directed"] / total if total else None,
        "combined_ratio": counts["combined"] / total if total else None,
    }


def add_arguments(parser):
    parser.add_argument("--emprical-gate", choices=("on", "off"), default="on")
    parser.add_argument("--emprical-gate-seed", type=int, default=None)
    parser.add_argument("--emprical-gate-freeze-state", action="store_true",
                        help="Keep the three empirical internal-state inputs fixed before every perception.")
    for factor in ("stress", "depletion", "cognitive-load"):
        parser.add_argument(f"--emprical-gate-{factor}", type=int, choices=range(11),
                            help="Fixed gate-only input; omitted values follow Desire State.")
    parser.add_argument("--emprical-gate-log-out", default=None,
                        help="Optional JSON trace of retrieved-habit gate decisions.")


def configure_runtime(runtime, args):
    loop = runtime.loop
    loop.emprical_gate_enabled = args.emprical_gate == "on"
    loop.emprical_gate_rng = random.Random(args.emprical_gate_seed)
    loop.emprical_gate_overrides = {
        key: value for key in ("stress", "depletion", "cognitive_load")
        if (value := getattr(args, f"emprical_gate_{key}")) is not None
    }
    loop.emprical_gate_freeze_state = bool(args.emprical_gate_freeze_state)
    if loop.emprical_gate_freeze_state:
        if len(loop.emprical_gate_overrides) != 3:
            raise ValueError("Freezing requires explicit stress, depletion, and cognitive-load values.")
        loop._apply_emprical_gate_fixed_state(runtime.agent)
    print(f"Emprical gate: {args.emprical_gate}, fixed inputs={loop.emprical_gate_overrides}, "
          f"freeze={loop.emprical_gate_freeze_state}")
    if args.emprical_gate_log_out:
        destination = Path(args.emprical_gate_log_out)

        def export():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({
                "enabled": loop.emprical_gate_enabled,
                "seed": args.emprical_gate_seed,
                "fixed_inputs": loop.emprical_gate_overrides,
                "state_frozen": loop.emprical_gate_freeze_state,
                "scope": "decisions are conditional on retrieved habits; actions and behavior_summary cover the full run",
                "decisions": loop.emprical_gate_history,
                "behavior_summary": summarize_actions(loop.emprical_gate_action_history),
                "actions": loop.emprical_gate_action_history,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        atexit.register(export)

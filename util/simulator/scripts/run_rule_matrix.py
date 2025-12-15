from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from util.simulator.simulator import SimulationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every redeployment rule for a scenario template."
    )
    parser.add_argument(
        "--template",
        default="stress",
        help="Scenario template name (default: stress).",
    )
    parser.add_argument(
        "--history",
        type=Path,
        action="append",
        help="Optional incident history override (can repeat).",
    )
    parser.add_argument(
        "--rules",
        nargs="*",
        help="Optional subset of rule IDs; defaults to all catalog rules.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=500,
        help="Base RNG seed used when iterating over rules.",
    )
    parser.add_argument(
        "--prefer-cache",
        action="store_true",
        help="Skip database refresh and use cached assets only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_paths: Sequence[Path] | None = args.history
    runner = SimulationRunner(prefer_database=not args.prefer_cache)
    rows = runner.evaluate_rules(
        args.template,
        rule_ids=tuple(args.rules) if args.rules else None,
        seed=args.seed,
        history_paths=history_paths,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()


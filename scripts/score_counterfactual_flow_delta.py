#!/usr/bin/env python3
"""Score candidate-blind same-source flow differences for exploratory progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iac_new.flow_structure_scoring import score_counterfactual_structure_pairs


def _read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--descriptor", default="median_flow_magnitude_px")
    parser.add_argument("--orientation", type=int, default=1)
    parser.add_argument("--action-reference", choices=["endpoint_column", "trajectory_path_length"], default="trajectory_path_length")
    parser.add_argument("--action-column", type=int, default=2)
    parser.add_argument("--minimum-common-intervals", type=int, default=2)
    parser.add_argument("--minimum-action-delta", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score_counterfactual_structure_pairs(
        _read_jsonl(args.measurement),
        _read_jsonl(args.manifest),
        descriptor=args.descriptor,
        orientation=args.orientation,
        action_reference=args.action_reference,
        action_column=args.action_column,
        minimum_common_intervals=args.minimum_common_intervals,
        minimum_action_delta=args.minimum_action_delta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()

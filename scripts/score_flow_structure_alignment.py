#!/usr/bin/env python3
"""Score decoder-free flow-structure alignment for paired WAM branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iac_new.flow_structure_scoring import score_flow_structure_pairs


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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    score = config["ordinal_alignment"]
    report = score_flow_structure_pairs(
        _read_jsonl(args.measurement),
        _read_jsonl(args.manifest),
        descriptor=str(score["primary_descriptor"]),
        orientation=int(score["orientation"]),
        action_column=int(score["action_column"]),
        action_reference=str(score.get("action_reference", "endpoint_column")),
        minimum_common_intervals=int(score["minimum_common_intervals"]),
        minimum_action_delta=float(score["minimum_action_delta"]),
        bootstrap_draws=int(score.get("bootstrap_draws", 2000)),
        bootstrap_seed=int(score.get("bootstrap_seed", 6101)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()

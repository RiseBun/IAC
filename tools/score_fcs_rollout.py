#!/usr/bin/env python3
"""Score explicit independent simulator outcomes for the FCS cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.fcs import score_fcs_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL independent-rollout rows")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=1)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = score_fcs_rollout(rows, minimum_rows=args.minimum_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows_detail"}, indent=2))
    if report["status"] == "unavailable":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

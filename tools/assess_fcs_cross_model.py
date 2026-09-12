#!/usr/bin/env python3
"""Fail-closed cross-model assessment for independently scored FCS reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.fcs import assess_cross_model_fcs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="one FCS JSON report per model")
    parser.add_argument("--minimum-models", type=int, default=2)
    parser.add_argument("--minimum-scored-rows", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    result = assess_cross_model_fcs(
        reports,
        minimum_models=args.minimum_models,
        minimum_scored_rows=args.minimum_scored_rows,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["claim_enabled"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate a JSON evidence table against one metric contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.metric_evidence_contract import validate_metric_evidence_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", required=True, choices=("MAS", "RCS", "GS", "FCS"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    rows = value if isinstance(value, list) else list(value.get("rows") or [])
    report = validate_metric_evidence_table(rows, args.metric)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

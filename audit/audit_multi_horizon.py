#!/usr/bin/env python3
"""Audit horizon-specific manifests and emit a reproducible protocol summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _pathfix  # noqa: F401

from multi_horizon_protocol import (  # noqa: E402
    HORIZONS,
    get_horizon,
    summarize_jsonl,
    write_protocol_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output protocol manifest JSON")
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        metavar="HORIZON=PATH",
        help="Optional JSONL manifest to audit, e.g. 4s=/path/rows.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_protocol_manifest(args.out)
    report = {"protocol_manifest": str(Path(args.out).resolve()), "audits": {}}
    for item in args.manifest:
        if "=" not in item:
            raise SystemExit(f"Invalid --manifest {item!r}; expected HORIZON=PATH")
        horizon, raw_path = item.split("=", 1)
        spec = get_horizon(horizon)
        result = summarize_jsonl(raw_path, spec)
        report["audits"][spec.name] = result
    report_path = Path(args.out).with_name(Path(args.out).stem + "_audit.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

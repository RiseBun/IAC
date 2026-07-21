#!/usr/bin/env python3
"""Validate that an IAC-PathBench summary follows the frozen v2 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_CATEGORIES = [
    "hit",
    "ambiguous_accept",
    "evidence_supported_miss",
    "likely_model_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", help="wam_iac_summary.json files")
    parser.add_argument(
        "--expect-protocol-name",
        default="IAC-PathBench v2 ambiguity-aware",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional machine-readable validation report.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_summary(path: Path, expected_name: str) -> Dict[str, Any]:
    summary = _load_json(path)
    v2 = summary.get("iac_pathbench_v2", {})
    protocol = v2.get("protocol", {})
    diagnostics = v2.get("diagnostic_metrics", {})
    primary = v2.get("primary_scientific_metrics", {})
    secondary = v2.get("secondary_ranking_metrics", {})

    categories = list(protocol.get("formal_categories") or [])
    category_set = set(categories)
    hit_ratio = secondary.get("hard_top1")
    raw_miss_fraction = diagnostics.get("raw_miss_fraction")

    issues: List[str] = []
    if protocol.get("name") != expected_name:
        issues.append(f"protocol.name={protocol.get('name')!r}")
    if category_set != set(EXPECTED_CATEGORIES):
        issues.append(f"formal_categories={categories!r}")
    if protocol.get("hard_top1_is_secondary") is not True:
        issues.append("hard_top1_is_secondary is not true")
    if "raw_miss_fraction" not in diagnostics:
        issues.append("missing diagnostic raw_miss_fraction")
    if "likely_model_error_fraction" not in diagnostics:
        issues.append("missing diagnostic likely_model_error_fraction")
    if "ambiguity_supported_miss_fraction" not in diagnostics:
        issues.append("missing diagnostic ambiguity_supported_miss_fraction")
    if raw_miss_fraction is not None and hit_ratio is not None:
        expected_raw_miss = 1.0 - float(hit_ratio)
        if abs(float(raw_miss_fraction) - expected_raw_miss) > 1e-6:
            issues.append(
                f"raw_miss_fraction={raw_miss_fraction!r} != 1-hard_top1={expected_raw_miss!r}"
            )
    for key in (
        "exact_path_win_fraction",
        "exact_path_delta",
        "path_minus_sky_delta",
        "ambiguity_adjusted_top1",
    ):
        if key not in primary:
            issues.append(f"missing primary metric {key}")
    for key in ("hard_top1", "mrr", "ndcg@3", "ndcg@5"):
        if key not in secondary:
            issues.append(f"missing secondary metric {key}")

    return {
        "summary": str(path),
        "ok": not issues,
        "issues": issues,
        "protocol": protocol,
        "primary": primary,
        "secondary": secondary,
        "diagnostic": diagnostics,
    }


def main() -> None:
    args = parse_args()
    reports = [_check_summary(Path(path), args.expect_protocol_name) for path in args.summaries]
    for report in reports:
        status = "OK" if report["ok"] else "FAIL"
        print(f"[{status}] {report['summary']}")
        for issue in report["issues"]:
            print(f"  - {issue}")
    payload = {"expected_protocol_name": args.expect_protocol_name, "reports": reports}
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(not report["ok"] for report in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

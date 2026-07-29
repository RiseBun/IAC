#!/usr/bin/env python3
"""Compare normal and controlled ordered-motion score JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ordered_motion_common import (  # noqa: E402
    DEFAULT_ACCEPTABLE_SOURCES,
    DEFAULT_HARD_SOURCES,
    load_rows,
    ranking_metrics,
    split_csv,
    write_json,
)


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--scores must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise ValueError("--scores must use non-empty NAME=PATH")
    return name.strip(), Path(raw_path)


def compare(args: argparse.Namespace) -> Dict[str, Any]:
    named = [_parse_named_path(value) for value in args.scores]
    if not named or named[0][0] != args.reference:
        raise ValueError(
            f"first --scores entry must be the reference {args.reference}=PATH"
        )
    rows_by_name = {name: load_rows(path) for name, path in named}
    reference_rows = rows_by_name[args.reference]
    reference_by_id = {
        str(row.get("sample_id", "")): row
        for row in reference_rows
    }
    reference_ids = list(reference_by_id)
    score_by_name: Dict[str, List[float]] = {}
    metrics: Dict[str, Any] = {}
    for name, _ in named:
        current_by_id = {
            str(row.get("sample_id", "")): row
            for row in rows_by_name[name]
        }
        missing = [
            sample_id
            for sample_id in reference_ids
            if sample_id not in current_by_id
        ]
        if missing:
            raise ValueError(
                f"{name} misses {len(missing)} reference sample ids"
            )
        scores = [
            float(current_by_id[sample_id][args.score_key])
            for sample_id in reference_ids
        ]
        score_by_name[name] = scores
        metrics[name] = ranking_metrics(
            [reference_by_id[sample_id] for sample_id in reference_ids],
            scores,
            acceptable_sources=split_csv(args.acceptable_sources),
            hard_sources=split_csv(args.hard_sources),
        )

    reference_scores = score_by_name[args.reference]
    delta: Dict[str, Any] = {}
    for name, values in score_by_name.items():
        if name == args.reference:
            continue
        absolute = [
            abs(left - right)
            for left, right in zip(reference_scores, values)
        ]
        delta[name] = {
            "mean_absolute_score_delta": (
                sum(absolute) / len(absolute) if absolute else None
            ),
            "max_absolute_score_delta": max(absolute, default=0.0),
            "fraction_changed_gt_1e_6": (
                sum(value > 1e-6 for value in absolute)
                / max(len(absolute), 1)
            ),
        }
    summary = {
        "kind": "ordered_motion_score_comparison",
        "reference": args.reference,
        "score_key": args.score_key,
        "rows": len(reference_ids),
        "metrics": metrics,
        "score_delta_from_reference": delta,
        "source_labels_used_for_report_metrics_only": True,
    }
    write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        action="append",
        required=True,
        help="Repeat NAME=PATH; put the reference first.",
    )
    parser.add_argument("--reference", default="normal")
    parser.add_argument("--score-key", default="ordered_motion_rank_score")
    parser.add_argument("--output-summary", required=True)
    parser.add_argument(
        "--acceptable-sources",
        default=DEFAULT_ACCEPTABLE_SOURCES,
    )
    parser.add_argument("--hard-sources", default=DEFAULT_HARD_SOURCES)
    return parser.parse_args()


def main() -> None:
    compare(parse_args())


if __name__ == "__main__":
    main()

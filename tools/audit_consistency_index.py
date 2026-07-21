#!/usr/bin/env python3
"""Audit IAC JSONL indexes before training or reporting benchmark results.

The core IAC invariant is temporal: for a positive sample, `future_images`
should be GT frames after the current history window, not a replay of history.
This script makes that invariant visible and optionally fails when an index
violates configured thresholds.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


_DIFFICULTY_BY_SOURCE = {
    "gt_pos": 0,
    "image_swap": 1,
    "perturb_lateral": 1,
    "perturb_heading": 2,
    "traj_swap": 3,
    "time_shift_future": 3,
    "reverse_traj": 3,
    "perturb_speed": 4,
    "perturb_lateral+heading": 2,
    "perturb_lateral+speed": 2,
    "perturb_heading+speed": 2,
    "perturb_lateral+heading+speed": 3,
    "all_perturb+reverse": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit IAC consistency JSONL indexes")
    parser.add_argument("indexes", nargs="+", help="Index JSONL files to audit")
    parser.add_argument("--image-root", default=None, help="Resolve relative image paths and check file existence")
    parser.add_argument("--max-rows", type=int, default=0, help="Rows to sample per index; 0 scans all rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=None, help="Optional path for machine-readable audit report")
    parser.add_argument("--group-key", default="group_id", help="Candidate group key.")
    parser.add_argument(
        "--required-sources",
        default="gt_pos,image_swap,time_shift_future,traj_swap,perturb_lateral,perturb_heading,perturb_speed",
        help="Comma-separated source types that should appear in a mature coverage split.",
    )
    parser.add_argument(
        "--fail-missing-required-source",
        action="store_true",
        help="Fail if any required source type has zero rows.",
    )
    parser.add_argument(
        "--fail-positive-exact-overlap",
        type=float,
        default=None,
        help="Fail if positive rows with history_images == future_images exceed this ratio",
    )
    parser.add_argument(
        "--fail-positive-any-overlap",
        type=float,
        default=None,
        help="Fail if positive rows sharing any history/future path exceed this ratio",
    )
    parser.add_argument(
        "--fail-missing-images",
        action="store_true",
        help="Fail if any referenced image is missing under --image-root",
    )
    return parser.parse_args()


def _iter_rows(path: Path, max_rows: int, seed: int) -> Iterable[Dict[str, Any]]:
    if max_rows <= 0:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    rng = random.Random(seed)
    reservoir: List[str] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if len(reservoir) < max_rows:
                reservoir.append(line)
            else:
                pick = rng.randrange(seen)
                if pick < max_rows:
                    reservoir[pick] = line
    for line in reservoir:
        yield json.loads(line)


def _summary_path(index_path: Path) -> Path | None:
    candidates = [
        index_path.parent / "consistency_index_summary.json",
        index_path.with_name("index_summary.json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _safe_ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _safe_mean(values: List[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _percentile(values: List[float], pct: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _difficulty_bucket(source: str) -> int:
    return int(_DIFFICULTY_BY_SOURCE.get(source, 1))


def _candidate_group_id(row: Dict[str, Any], group_key: str) -> str:
    group = row.get(group_key) or row.get("anchor_id") or row.get("group_id")
    if group is not None:
        return str(group)
    sample_id = row.get("sample_id")
    if sample_id is None:
        return "unknown"
    sample_id = str(sample_id)
    source = row.get("source_type") or row.get("sample_type") or row.get("action_type")
    if source is not None:
        suffix = f"__{source}"
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    if "__" in sample_id:
        return sample_id.rsplit("__", 1)[0]
    return sample_id


def _image_paths(row: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for key in ("history_images", "future_images"):
        value = row.get(key, [])
        if isinstance(value, list):
            paths.extend(str(item) for item in value if isinstance(item, str))
    return paths


def audit_index(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    by_source: Dict[str, Counter] = defaultdict(Counter)
    by_difficulty: Counter = Counter()
    group_sizes: Counter = Counter()
    label_counts: Counter = Counter()
    source_counts: Counter = Counter()
    missing_examples: List[str] = []
    exact_overlap = 0
    any_overlap = 0
    positive_exact_overlap = 0
    positive_any_overlap = 0
    positive_rows = 0
    rows = 0
    missing_images = 0
    checked_images = 0
    image_root = Path(args.image_root) if args.image_root else None
    group_rows: Dict[str, int] = defaultdict(int)
    required_sources = [item.strip() for item in str(args.required_sources).split(",") if item.strip()]

    for row in _iter_rows(path, args.max_rows, args.seed):
        rows += 1
        source = str(row.get("source_type", "unknown"))
        label = int(float(row.get("consistency_label", 0)) > 0.5)
        source_counts[source] += 1
        label_counts[str(label)] += 1
        by_difficulty[_difficulty_bucket(source if label == 0 else "gt_pos")] += 1
        gid = _candidate_group_id(row, args.group_key)
        group_rows[gid] += 1

        history = [str(item) for item in row.get("history_images", []) if isinstance(item, str)]
        future = [str(item) for item in row.get("future_images", []) if isinstance(item, str)]
        is_exact = bool(history) and history == future
        is_any = bool(set(history) & set(future))
        exact_overlap += int(is_exact)
        any_overlap += int(is_any)
        by_source[source]["rows"] += 1
        by_source[source]["exact_overlap"] += int(is_exact)
        by_source[source]["any_overlap"] += int(is_any)
        by_source[source][f"label_{label}"] += 1

        if label == 1:
            positive_rows += 1
            positive_exact_overlap += int(is_exact)
            positive_any_overlap += int(is_any)

        if image_root is not None:
            for raw in _image_paths(row):
                checked_images += 1
                resolved = Path(raw)
                if not resolved.is_absolute():
                    resolved = image_root / raw
                if not resolved.exists():
                    missing_images += 1
                    if len(missing_examples) < 10:
                        missing_examples.append(str(resolved))

    group_size_values = list(group_rows.values())

    summary_file = _summary_path(path)
    index_summary: Dict[str, Any] = {}
    if summary_file is not None:
        try:
            index_summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            index_summary = {"summary_parse_error": str(exc)}

    per_source = {}
    for source, counter in sorted(by_source.items()):
        n = counter["rows"]
        per_source[source] = {
            "rows": n,
            "labels": {
                "positive": counter["label_1"],
                "negative": counter["label_0"],
            },
            "exact_overlap_ratio": _safe_ratio(counter["exact_overlap"], n),
            "any_overlap_ratio": _safe_ratio(counter["any_overlap"], n),
        }

    required_source_report = {
        source: int(source_counts.get(source, 0))
        for source in required_sources
    }
    missing_required_sources = [source for source, count in required_source_report.items() if count <= 0]

    return {
        "index": str(path),
        "rows_scanned": rows,
        "sampled": args.max_rows > 0,
        "index_summary": {
            key: index_summary.get(key)
            for key in (
                "source",
                "future_image_policy",
                "num_train_rows",
                "num_val_rows",
                "num_train_anchors",
                "num_val_anchors",
            )
            if key in index_summary
        },
        "source_counts": dict(source_counts),
        "label_counts": dict(label_counts),
        "exact_overlap_ratio": _safe_ratio(exact_overlap, rows),
        "any_overlap_ratio": _safe_ratio(any_overlap, rows),
        "positive_rows": positive_rows,
        "positive_exact_overlap_ratio": _safe_ratio(positive_exact_overlap, positive_rows),
        "positive_any_overlap_ratio": _safe_ratio(positive_any_overlap, positive_rows),
        "per_source": per_source,
        "coverage": {
            "required_sources": required_sources,
            "required_source_counts": required_source_report,
            "missing_required_sources": missing_required_sources,
            "difficulty_counts": {str(key): int(value) for key, value in sorted(by_difficulty.items())},
            "group_count": len(group_rows),
            "group_size": {
                "mean": _safe_mean(group_size_values),
                "p50": _percentile(group_size_values, 50.0),
                "p90": _percentile(group_size_values, 90.0),
                "min": min(group_size_values) if group_size_values else None,
                "max": max(group_size_values) if group_size_values else None,
            },
            "source_coverage_ratio": _safe_ratio(
                sum(1 for count in required_source_report.values() if count > 0),
                len(required_sources) if required_sources else 1,
            ),
        },
        "image_check": {
            "enabled": image_root is not None,
            "checked_images": checked_images,
            "missing_images": missing_images,
            "missing_examples": missing_examples,
        },
    }


def _print_report(report: Dict[str, Any]) -> None:
    print(f"\nIndex: {report['index']}")
    print("=" * 72)
    summary = report["index_summary"]
    if summary:
        print("metadata:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    print(f"rows_scanned: {report['rows_scanned']}")
    print(f"labels: {report['label_counts']}")
    print(f"sources: {report['source_counts']}")
    print(f"exact history/future overlap: {report['exact_overlap_ratio']:.4f}")
    print(f"any history/future overlap:   {report['any_overlap_ratio']:.4f}")
    print(
        "positive overlap: "
        f"exact={report['positive_exact_overlap_ratio']:.4f} "
        f"any={report['positive_any_overlap_ratio']:.4f} "
        f"n={report['positive_rows']}"
    )
    print("per_source:")
    for source, item in report["per_source"].items():
        print(
            f"  {source}: rows={item['rows']} "
            f"pos={item['labels']['positive']} neg={item['labels']['negative']} "
            f"exact={item['exact_overlap_ratio']:.4f} "
            f"any={item['any_overlap_ratio']:.4f}"
        )
    coverage = report["coverage"]
    print("coverage:")
    print(f"  group_count: {coverage['group_count']}")
    print(f"  group_size: {coverage['group_size']}")
    print(f"  difficulty_counts: {coverage['difficulty_counts']}")
    print(f"  required_sources: {coverage['required_source_counts']}")
    if coverage["missing_required_sources"]:
        print(f"  missing_required_sources: {coverage['missing_required_sources']}")
    image_check = report["image_check"]
    if image_check["enabled"]:
        print(
            "image_check: "
            f"checked={image_check['checked_images']} "
            f"missing={image_check['missing_images']}"
        )
        for example in image_check["missing_examples"]:
            print(f"  missing: {example}")


def main() -> None:
    args = parse_args()
    reports = [audit_index(Path(index), args) for index in args.indexes]
    for report in reports:
        _print_report(report)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = False
    for report in reports:
        if (
            args.fail_positive_exact_overlap is not None
            and report["positive_exact_overlap_ratio"] > args.fail_positive_exact_overlap
        ):
            failed = True
        if (
            args.fail_positive_any_overlap is not None
            and report["positive_any_overlap_ratio"] > args.fail_positive_any_overlap
        ):
            failed = True
        if args.fail_missing_required_source and report["coverage"]["missing_required_sources"]:
            failed = True
        if args.fail_missing_images and report["image_check"]["missing_images"] > 0:
            failed = True
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

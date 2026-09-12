#!/usr/bin/env python3
"""Recompute the structural Grounding Score from user-supplied references.

The reference stream may be private logged future data.  This command makes
the protocol independently executable without bundling any benchmark imagery.
It never substitutes missing intervals with zeros.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.visual_consistency import score_structural_grounding


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _expand(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        profile = row.get("flow_structure")
        if isinstance(profile, dict) and isinstance(profile.get("rows"), list):
            for interval in profile["rows"]:
                expanded.append({
                    **interval,
                    "source_key": row.get("source_key") or row.get("sample_id"),
                    "stratum": row.get("stratum"),
                })
        else:
            expanded.append(dict(row))
    return expanded


def _bootstrap_median(values: np.ndarray, *, draws: int, seed: int) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    medians = [float(np.median(values[rng.integers(0, len(values), len(values))])) for _ in range(int(draws))]
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def score(
    generated: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    scales: dict[str, float],
    min_common_intervals: int = 3,
    min_descriptors_per_interval: int = 3,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 20260912,
) -> dict[str, Any]:
    generated = _expand(generated)
    reference = _expand(reference)
    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
        result: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            source = str(row.get("source_key") or "")
            interval = int(row.get("interval_index", 0))
            if not source:
                raise ValueError("every grounding row requires source_key")
            key = (source, interval)
            if key in result:
                raise ValueError(f"duplicate grounding row: {key}")
            result[key] = row
        return result
    generated_index = index(generated)
    reference_index = index(reference)
    sources = sorted({source for source, _ in generated_index} | {source for source, _ in reference_index})
    source_reports: list[dict[str, Any]] = []
    for source in sources:
        intervals = sorted({interval for item_source, interval in generated_index if item_source == source} | {interval for item_source, interval in reference_index if item_source == source})
        generated_rows: list[dict[str, Any]] = []
        reference_rows: list[dict[str, Any]] = []
        for interval in intervals:
            generated_rows.append(generated_index.get((source, interval), {"interval_index": interval, "input_available": False}))
            reference_rows.append(reference_index.get((source, interval), {"interval_index": interval, "input_available": False}))
        report = score_structural_grounding(
            generated_rows,
            reference_rows,
            descriptor_scales=scales,
            min_common_intervals=min_common_intervals,
            min_descriptors_per_interval=min_descriptors_per_interval,
        )
        source_reports.append({
            "source_key": source,
            "status": report["status"],
            "score": report["score"],
            "common_interval_count": report["common_interval_count"],
            "interval_count": report["interval_count"],
            "stratum": next((row.get("stratum") for row in generated_rows if row.get("stratum") is not None), "unknown"),
        })
    scored = [row for row in source_reports if row["status"] == "ok" and row["score"] is not None]
    values = np.asarray([row["score"] for row in scored], dtype=np.float64)
    return {
        "protocol": "iac-structural-grounding-score-v1",
        "metric_id": "GS",
        "status": "ok" if scored else "unavailable",
        "source_count": len(source_reports),
        "scored_source_count": len(scored),
        "source_coverage": len(scored) / len(source_reports) if source_reports else None,
        "score_median": float(np.median(values)) if len(values) else None,
        "score_mean": float(np.mean(values)) if len(values) else None,
        "score_bootstrap_ci95": _bootstrap_median(values, draws=bootstrap_draws, seed=bootstrap_seed),
        "descriptor_scales": {key: float(value) for key, value in scales.items()},
        "source_reports": source_reports,
        "claim": "external_reality_grounding_not_future_to_action_causality",
        "missing_value_policy": "unavailable_never_zero_fill",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True, help="JSON object of frozen descriptor scales")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    report = score(
        _read(args.generated),
        _read(args.reference),
        scales=json.loads(args.scales.read_text(encoding="utf-8")),
        bootstrap_draws=args.bootstrap_draws,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "source_count", "scored_source_count", "source_coverage", "score_median", "score_bootstrap_ci95")}, indent=2))


if __name__ == "__main__":
    main()

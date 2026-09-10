#!/usr/bin/env python3
"""Compare real/generated trajectory-flow energies after calibration correction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ENERGY_FIELDS = ("logged_gt_energy", "action_head_energy", "zero_flow_energy")


def _read(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("records") or [])
    return rows


def _intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        for interval in row["geometry_comparison"]["by_interval"]:
            result.append({
                "source_key": str(row["source_key"]),
                "branch_role": str(row.get("branch_role") or "real"),
                "stratum": str(row.get("stratum") or "unknown"),
                **interval,
            })
    return result


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "q25": float(np.quantile(data, 0.25)),
        "q75": float(np.quantile(data, 0.75)),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row["logged_gt_energy"] is not None]
    result: dict[str, Any] = {
        "intervals": len(rows),
        "available_intervals": len(available),
        "availability": len(available) / len(rows) if rows else None,
    }
    for field in ENERGY_FIELDS:
        result[field] = _quantiles([
            float(row[field]) for row in available if row.get(field) is not None
        ])
    result.update({
        "action_minus_gt_energy": _quantiles([
            float(row["action_head_energy"] - row["logged_gt_energy"])
            for row in available
            if row.get("action_head_energy") is not None
        ]),
        "action_better_than_gt": int(sum(bool(row["action_better"]) for row in available)),
        "action_better_than_gt_fraction": (
            float(np.mean([row["action_better"] for row in available]))
            if available else None
        ),
        "gt_better_than_zero": int(sum(bool(row["gt_better_than_zero"]) for row in available)),
        "gt_better_than_zero_fraction": (
            float(np.mean([row["gt_better_than_zero"] for row in available]))
            if available else None
        ),
        "action_better_than_zero": int(sum(bool(row["action_better_than_zero"]) for row in available)),
        "action_better_than_zero_fraction": (
            float(np.mean([row["action_better_than_zero"] for row in available]))
            if available else None
        ),
    })
    return result


def _cluster_bootstrap_mean(
    rows: list[dict[str, Any]],
    field: str,
    *,
    draws: int = 4000,
) -> list[float] | None:
    clusters: dict[str, list[float]] = {}
    for row in rows:
        clusters.setdefault(row["source_key"], []).append(float(row[field]))
    if not clusters:
        return None
    values = np.asarray([np.mean(items) for items in clusters.values()], dtype=np.float64)
    rng = np.random.default_rng(20260910)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.mean(values[samples], axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _paired(
    real: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    *,
    include_strata: bool = True,
) -> dict[str, Any]:
    real_index = {
        (row["source_key"], int(row["interval_index"])): row
        for row in real
        if row["logged_gt_energy"] is not None
    }
    pairs = []
    for row in generated:
        key = (row["source_key"], int(row["interval_index"]))
        reference = real_index.get(key)
        if reference is None or row["logged_gt_energy"] is None:
            continue
        paired = {
            "source_key": row["source_key"],
            "branch_role": row["branch_role"],
            "stratum": row["stratum"],
            "interval_index": int(row["interval_index"]),
        }
        for field in ENERGY_FIELDS:
            paired[f"{field}_generated_minus_real"] = float(row[field] - reference[field])
        pairs.append(paired)
    output: dict[str, Any] = {
        "pairs": len(pairs),
        "source_clusters": len({row["source_key"] for row in pairs}),
    }
    for field in ENERGY_FIELDS:
        delta_field = f"{field}_generated_minus_real"
        values = [float(row[delta_field]) for row in pairs]
        output[delta_field] = {
            **(_quantiles(values) or {}),
            "source_cluster_bootstrap_mean_ci95": _cluster_bootstrap_mean(pairs, delta_field),
            "generated_worse_count": int(sum(value > 0.0 for value in values)),
            "generated_worse_fraction": float(np.mean(np.asarray(values) > 0.0)) if values else None,
        }
    output["by_stratum"] = {
        stratum: _paired(
            [row for row in real if row["stratum"] == stratum],
            [row for row in generated if row["stratum"] == stratum],
            include_strata=False,
        )
        for stratum in sorted({row["stratum"] for row in pairs})
    } if pairs and include_strata else {}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, action="append", required=True)
    parser.add_argument("--generated", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    real = _intervals(_read(args.real))
    generated = _intervals(_read(args.generated))
    report = {
        "protocol": "intrinsics-corrected-real-generated-energy-audit-v1",
        "energy_denominator": "per-interval common GT/action projection support",
        "real": _summary(real),
        "generated": _summary(generated),
        "paired_generated_minus_real": _paired(real, generated),
        "stratum_counts": dict(Counter(row["stratum"] for row in generated)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

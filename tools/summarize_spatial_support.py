#!/usr/bin/env python3
"""Aggregate sampled image-row distributions and GT projection rates."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EDGES = np.asarray([0.50, 0.60, 0.70, 0.80, 0.90, 1.001], dtype=np.float64)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quantiles = []
    bins = [{"v_min": float(lo), "v_max": float(min(hi, 1.0)), "sampled_pixels": 0,
             "source_interval_points": 0, "projected_interval_points": 0}
            for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    for row in rows:
        spatial = row.get("spatial_support") or {}
        q = spatial.get("sampled_v_quantiles")
        if q:
            quantiles.append(q)
        for target, source in zip(bins, spatial.get("v_bins") or []):
            for field in ("sampled_pixels", "source_interval_points", "projected_interval_points"):
                target[field] += int(source.get(field) or 0)
    for item in bins:
        source = item["source_interval_points"]
        item["projection_rate"] = item["projected_interval_points"] / source if source else None
    out = {"rows": len(rows), "v_bins": bins}
    if quantiles:
        out["sampled_v_quantiles_mean"] = {
            name: float(np.mean([float(q[name]) for q in quantiles]))
            for name in ("q05", "q25", "median", "q75", "q95")
        }
        out["sampled_v_quantiles_median"] = {
            name: float(np.median([float(q[name]) for q in quantiles]))
            for name in ("q05", "q25", "median", "q75", "q95")
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("records", []))
    output = {"protocol": "spatial-support-summary-v1", "aggregate": aggregate(rows)}
    by_stratum = defaultdict(list)
    for row in rows:
        by_stratum[row.get("stratum")].append(row)
    output["by_stratum"] = {stratum: aggregate(items) for stratum, items in sorted(by_stratum.items(), key=lambda item: str(item[0]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

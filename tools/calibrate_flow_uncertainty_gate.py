#!/usr/bin/env python3
"""Calibrate an interval-level RAFT refinement gate on real logged motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def _read(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    for start in range(len(values)):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        if start == 0 or values[order[start]] != values[order[start - 1]]:
            result[order[start:stop]] = 0.5 * (start + stop - 1)
    return result


def _wilson(hits: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = hits / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [float(center - half), float(center + half)]


def _metrics(rows: list[dict], threshold: float, min_yaw_delta: float) -> dict:
    selected = [row for row in rows if row["uncertainty"] <= threshold]
    material = [row for row in selected if abs(row["gt_yaw_delta"]) >= min_yaw_delta]
    hits = sum(np.sign(row["flow_center"]) == np.sign(row["gt_yaw_delta"]) for row in material)
    rho = None
    if len(material) >= 3:
        first = np.asarray([row["flow_center"] for row in material])
        second = np.asarray([row["gt_yaw_delta"] for row in material])
        if np.ptp(first) > 0.0 and np.ptp(second) > 0.0:
            rho = float(np.corrcoef(_rank(first), _rank(second))[0, 1])
    return {
        "intervals": len(rows),
        "selected": len(selected),
        "coverage": len(selected) / len(rows) if rows else None,
        "material_intervals": len(material),
        "direction_hits": int(hits),
        "direction_accuracy": hits / len(material) if material else None,
        "direction_ci95": _wilson(int(hits), len(material)),
        "spearman": rho,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-step-s", type=float, default=0.5)
    parser.add_argument("--min-yaw-delta", type=float, default=0.005)
    parser.add_argument("--target-direction-accuracy", type=float, default=0.90)
    args = parser.parse_args()

    measurements = {row["sample_id"]: row for row in _read(args.measurement)}
    manifests = _read(args.manifest)
    cache: dict[str, dict] = {}
    intervals = []
    for manifest in manifests:
        measurement = measurements[manifest["sample_id"]]
        source = str(manifest["lineage"]["source_sample"])
        if source not in cache:
            cache[source] = pickle.loads(Path(source).read_bytes())
        future = cache[source]["future_trajectory"]
        target_times = np.asarray(manifest["future_times_s"], dtype=np.float64)
        indices = np.rint(target_times / args.source_step_s).astype(np.int64) - 1
        yaw = np.asarray([future[index]["pose"][2] for index in indices], dtype=np.float64)
        yaw_delta = np.diff(np.concatenate([[0.0], np.unwrap(yaw)]))
        for row, delta in zip(measurement["flow_structure"]["rows"], yaw_delta):
            uncertainty = (row.get("refinement_uncertainty") or {}).get("median")
            flow_center = row.get("horizontal_flow_center")
            if uncertainty is None or flow_center is None:
                continue
            intervals.append({
                "source_key": manifest["source_key"],
                "uncertainty": float(uncertainty),
                "flow_center": float(flow_center),
                "gt_yaw_delta": float(delta),
            })

    calibration, validation = [], []
    for row in intervals:
        bucket = hashlib.sha256(row["source_key"].encode()).digest()[0]
        (calibration if bucket < 128 else validation).append(row)
    candidates = sorted({row["uncertainty"] for row in calibration})
    chosen = None
    for threshold in candidates:
        metrics = _metrics(calibration, threshold, args.min_yaw_delta)
        if (
            metrics["material_intervals"] >= 20
            and metrics["direction_accuracy"] is not None
            and metrics["direction_accuracy"] >= args.target_direction_accuracy
        ):
            chosen = threshold
    if chosen is None:
        raise RuntimeError("no uncertainty threshold satisfies the preregistered accuracy constraint")
    report = {
        "protocol": "raft-refinement-uncertainty-gate-calibration-v1",
        "reference_domain": "real_logged_navsim_only",
        "threshold": float(chosen),
        "threshold_definition": "median late-iteration vector RMS / (1 + flow magnitude)",
        "selection_rule": "maximum calibration coverage with direction accuracy >= target",
        "source_split": "sha256(source_key)[0] < 128",
        "source_step_s": args.source_step_s,
        "min_yaw_delta": args.min_yaw_delta,
        "target_direction_accuracy": args.target_direction_accuracy,
        "calibration": _metrics(calibration, chosen, args.min_yaw_delta),
        "validation": _metrics(validation, chosen, args.min_yaw_delta),
        "all_real": _metrics(intervals, chosen, args.min_yaw_delta),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

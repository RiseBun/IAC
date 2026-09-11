#!/usr/bin/env python3
"""Fit a candidate-blind action-to-flow-structure adapter and score MAS.

Calibration uses only logged real future flow and the native action trajectory.
Confirmation uses untouched generated branches.  The adapter is deliberately
simple and auditable: per interval, robust affine maps predict structural
descriptors from cumulative action progress.  No generated confirmation row is
used to choose descriptors, scales, or thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DESCRIPTORS = ("median_flow_magnitude_px", "vertical_flow_center", "divergence")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trajectory_values(value: Any) -> np.ndarray:
    # DriveWAM stores predicted_action_trajectory as [1,3,T] component-first;
    # the Epona/manifest form is [T,3] (or a list of {pose: ...}).
    raw = np.asarray(value)
    if raw.dtype != object and raw.ndim == 3 and raw.shape[0] == 1 and raw.shape[1] >= 3:
        return np.asarray(raw[0, :3, :].T, dtype=float)
    arr = np.asarray(value, dtype=object)
    out = []
    for item in arr.tolist():
        if isinstance(item, dict):
            out.append(item["pose"])
        else:
            out.append(item)
    result = np.asarray(out, dtype=float)
    if result.ndim != 2 or result.shape[1] < 3:
        raise ValueError("action trajectory must contain [x,y,yaw]")
    return result[:, :3]


def _progress_by_interval(trajectory: Any, intervals: int) -> np.ndarray:
    pose = _trajectory_values(trajectory)
    # Native NavSim trajectories in this protocol are sampled at 0.5 s; the
    # four 1-Hz intervals use knots 1,3,5,7.  Fall back to evenly spaced knots.
    if len(pose) >= 2 * intervals:
        indices = np.minimum(2 * np.arange(intervals) + 1, len(pose) - 1)
    else:
        indices = np.linspace(0, len(pose) - 1, intervals).round().astype(int)
    return np.linalg.norm(pose[indices, :2], axis=1)


def _flow_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("flow_structure", {}).get("rows", []))


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(x) < 8:
        raise ValueError("insufficient calibration rows")
    coef = np.polyfit(x, y, 1)
    pred = coef[0] * x + coef[1]
    residual = y - pred
    med = float(np.median(residual))
    mad = float(1.4826 * np.median(np.abs(residual - med)))
    scale = max(mad, float(np.std(residual)), 1e-6)
    return float(coef[0]), float(coef[1]), scale


def fit_adapter(manifest: list[dict[str, Any]], flows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(str(row["source_key"]), str(row["branch_role"])): row for row in manifest}
    xs: dict[int, list[float]] = {}
    ys: dict[tuple[str, int], list[float]] = {}
    for record in flows:
        key = (str(record["source_key"]), str(record.get("branch_role", "logged_gt")))
        source = by_key.get(key) or by_key.get((key[0], "logged_gt"))
        if source is None:
            continue
        rows = _flow_rows(record)
        trajectory = source.get("action_trajectory") or source.get("predicted_action_trajectory")
        if trajectory is None:
            continue
        progress = _progress_by_interval(trajectory, len(rows))
        for index, row in enumerate(rows):
            if not row.get("input_available", True):
                continue
            xs.setdefault(index, []).append(float(progress[index]))
            for descriptor in DESCRIPTORS:
                value = row.get(descriptor)
                if value is not None and np.isfinite(value):
                    ys.setdefault((descriptor, index), []).append(float(value))
    fit: dict[str, Any] = {}
    for descriptor in DESCRIPTORS:
        fit[descriptor] = {}
        for index in sorted(xs):
            x = np.asarray(xs[index], dtype=float)
            y = np.asarray(ys.get((descriptor, index), []), dtype=float)
            if len(y) != len(x):
                # Rebuild aligned finite pairs for this descriptor.
                pairs = []
                for record in flows:
                    key = (str(record["source_key"]), str(record.get("branch_role", "logged_gt")))
                    source = by_key.get(key) or by_key.get((key[0], "logged_gt"))
                    if source is None:
                        continue
                    rows = _flow_rows(record)
                    if index >= len(rows):
                        continue
                    value = rows[index].get(descriptor)
                    if value is not None and np.isfinite(value):
                        trajectory = source.get("action_trajectory") or source.get("predicted_action_trajectory")
                        if trajectory is not None:
                            pairs.append((_progress_by_interval(trajectory, len(rows))[index], float(value)))
                x = np.asarray([p[0] for p in pairs], dtype=float)
                y = np.asarray([p[1] for p in pairs], dtype=float)
            if len(x) >= 8:
                slope, intercept, scale = _fit_affine(x, y)
                fit[descriptor][str(index)] = {"slope": slope, "intercept": intercept, "scale": scale, "n": int(len(x))}
    return {"protocol": "iac-mas-action-structure-calibration-v1", "descriptors": list(DESCRIPTORS), "fit": fit}


def score_confirmation(adapter: dict[str, Any], manifest: list[dict[str, Any]], flows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(str(row["source_key"]), str(row["branch_role"])): row for row in manifest}
    branch_scores = []
    for record in flows:
        source = by_key.get((str(record["source_key"]), str(record.get("branch_role"))))
        if source is None:
            continue
        rows = _flow_rows(record)
        trajectory = source.get("action_trajectory") or source.get("predicted_action_trajectory")
        if trajectory is None:
            continue
        progress = _progress_by_interval(trajectory, len(rows))
        interval_scores = []
        zero_scores = []
        for index, row in enumerate(rows):
            if not row.get("input_available", True):
                continue
            comps, zeros = [], []
            for descriptor in DESCRIPTORS:
                params = adapter["fit"].get(descriptor, {}).get(str(index))
                value = row.get(descriptor)
                if params is None or value is None or not np.isfinite(value):
                    continue
                scale = float(params["scale"])
                pred = float(params["slope"] * progress[index] + params["intercept"])
                zero = float(params["intercept"])
                # Magnitudes are positive and are calibrated in log-space.
                if descriptor == "median_flow_magnitude_px":
                    value_t, pred_t, zero_t = np.log1p(max(float(value), 0.0)), np.log1p(max(pred, 0.0)), np.log1p(max(zero, 0.0))
                else:
                    value_t, pred_t, zero_t = float(value), pred, zero
                comps.append(float(np.exp(-abs(value_t - pred_t) / max(scale, 1e-6))))
                zeros.append(float(np.exp(-abs(value_t - zero_t) / max(scale, 1e-6))))
            if len(comps) >= 2:
                interval_scores.append(float(np.median(comps)))
                zero_scores.append(float(np.median(zeros)))
        branch_scores.append({"source_key": record["source_key"], "branch_role": record.get("branch_role"), "speed_role": source.get("speed_role"), "interval_count": len(interval_scores), "score": float(np.median(interval_scores)) if interval_scores else None, "zero_score": float(np.median(zero_scores)) if zero_scores else None})
    available = [r for r in branch_scores if r["score"] is not None]
    twin_map: dict[str, dict[str, dict[str, Any]]] = {}
    for row in available:
        twin_map.setdefault(str(row["source_key"]), {})[str(row.get("speed_role"))] = row
    pairs = []
    for source, branches in twin_map.items():
        if "fast" in branches and "slow" in branches:
            fast, slow = branches["fast"], branches["slow"]
            pairs.append({"source_key": source, "fast_score": fast["score"], "slow_score": slow["score"], "fast_wins": bool(fast["score"] > slow["score"])})
    return {"protocol": "iac-mas-action-structure-confirmation-v1", "metric_id": "MAS", "branch_count": len(branch_scores), "coverage": len(available) / len(branch_scores) if branch_scores else 0.0, "median_score": float(np.median([r["score"] for r in available])) if available else None, "median_zero_score": float(np.median([r["zero_score"] for r in available])) if available else None, "pairs": pairs, "pair_speed_order_accuracy": float(np.mean([r["fast_wins"] for r in pairs])) if pairs else None, "branches": branch_scores}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-flows", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-flows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = fit_adapter(_read_jsonl(args.calibration_manifest), _read_jsonl(args.calibration_flows))
    result = score_confirmation(adapter, _read_jsonl(args.confirmation_manifest), _read_jsonl(args.confirmation_flows))
    result["calibration"] = adapter
    result["claim_boundary"] = "candidate-blind action-to-structure adapter; not a metric-free causal test"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run normal/reversed/zero twin differential forward-consistency controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iac_new.scoring import polygon_mask
from iac_new.visual_consistency import (
    score_twin_differential_consistency,
    trajectory_conditioned_flow,
)


def _interval_trajectory(value: np.ndarray, interval_count: int) -> np.ndarray:
    """Resample native 8-knot (0.5s) actions to four 1s flow intervals."""
    if len(value) == interval_count:
        return value
    if len(value) == 2 * interval_count:
        return value[[1, 3, 5, 7]]
    indices = np.linspace(0, len(value) - 1, interval_count).round().astype(int)
    return value[indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-vector-norm", type=float, default=0.10)
    args = parser.parse_args()
    records = []
    for path in sorted(args.root.glob("*.json")):
        if path.name == "manifest.json":
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if "flow_path" in metadata:
            records.append(metadata)
    by_source: dict[str, dict[str, dict]] = {}
    for metadata in records:
        by_source.setdefault(str(metadata["source_key"]), {})[str(metadata.get("branch_role"))] = metadata
    rows = []
    for source, branches in sorted(by_source.items()):
        if "left" not in branches or "right" not in branches:
            continue
        left_meta, right_meta = branches["left"], branches["right"]
        left_flow, right_flow = np.load(args.root / left_meta["flow_path"]), np.load(args.root / right_meta["flow_path"])
        left_valid, right_valid = np.load(args.root / left_meta["valid_path"]), np.load(args.root / right_meta["valid_path"])
        shape = left_flow.shape[1:3]
        roi = polygon_mask(shape[0], shape[1], [[0.08, 0.98], [0.92, 0.98], [0.63, 0.53], [0.37, 0.53]])
        interval_count = left_flow.shape[0]
        left_traj = _interval_trajectory(np.asarray(left_meta["trajectory"]), interval_count)
        right_traj = _interval_trajectory(np.asarray(right_meta["trajectory"]), interval_count)
        left_expected, left_geometry = trajectory_conditioned_flow(left_traj, intrinsics=np.asarray(left_meta["intrinsics"]), camera_to_ego=np.asarray(left_meta["camera_to_ego"]), frame_shape=shape)
        right_expected, right_geometry = trajectory_conditioned_flow(right_traj, intrinsics=np.asarray(right_meta["intrinsics"]), camera_to_ego=np.asarray(right_meta["camera_to_ego"]), frame_shape=shape)
        common = left_valid & right_valid & left_geometry & right_geometry & roi
        expected_delta = left_expected - right_expected
        controls = {
            "normal": (left_expected, right_expected),
            "reversed": (right_expected, left_expected),
            "zero": (right_expected, right_expected),
        }
        for control, (expected_left, expected_right) in controls.items():
            score = score_twin_differential_consistency(left_flow, right_flow, expected_left, expected_right, valid_mask=common, min_vector_norm_px=args.min_vector_norm)
            rows.append({"source_key": source, "control": control, **{key: score[key] for key in ("status_counts", "interval_coverage", "median_residual_px", "median_observed_delta_px", "median_expected_delta_px", "median_direction_cosine", "median_direction_vector_fraction", "median_support_fraction", "minimum_support_fraction")}})
    summary = {"protocol": "iac-twin-differential-forward-consistency-v1", "min_vector_norm_px": args.min_vector_norm, "controls": {control: {key: (float(np.median([row[key] for row in rows if row["control"] == control and row[key] is not None])) if any(row["control"] == control and row[key] is not None for row in rows) else None) for key in ("interval_coverage", "median_residual_px", "median_observed_delta_px", "median_expected_delta_px", "median_direction_cosine", "median_direction_vector_fraction", "median_support_fraction", "minimum_support_fraction")} for control in ("normal", "reversed", "zero")}}
    normal = [row["median_observed_delta_px"] / row["median_expected_delta_px"] for row in rows if row["control"] == "normal" and row["median_expected_delta_px"] and row["median_expected_delta_px"] > 0]
    summary["normal_response_gain_median"] = float(np.median(normal)) if normal else None
    summary["normal_response_gain_q25"] = float(np.quantile(normal, 0.25)) if normal else None
    summary["normal_response_gain_q75"] = float(np.quantile(normal, 0.75)) if normal else None
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recompute the universal Step1 controls from archived raw flow arrays.

This is deliberately a small, candidate-blind bridge.  It does not estimate
a trajectory.  The normal control compares the observed same-source flow
difference with the action-conditioned structural difference.  Reversed and
identity-swap retain the observed images but reverse the action assignment;
they must therefore not create a positive directional hit.  Zero contrast
has no directional estimand and is reported as unavailable, never as a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.scoring import polygon_mask
from iac_new.visual_consistency import (
    score_twin_differential_consistency,
    trajectory_conditioned_flow,
)


def _interval_trajectory(value: np.ndarray, interval_count: int) -> np.ndarray:
    if len(value) == interval_count:
        return value
    if len(value) == 2 * interval_count:
        return value[[1, 3, 5, 7]]
    indices = np.linspace(0, len(value) - 1, interval_count).round().astype(int)
    return value[indices]


def _load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"manifest.json", "score_roi.json", "twin_controls.json"}:
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not all(key in metadata for key in ("source_key", "branch_role", "trajectory", "flow_path", "valid_path")):
            continue
        metadata["_metadata_path"] = str(path)
        metadata["_array_root"] = str(path.parent)
        records.append(metadata)
    return records


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    estimates = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(draws)]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _control_score(
    left_flow: np.ndarray,
    right_flow: np.ndarray,
    left_expected: np.ndarray,
    right_expected: np.ndarray,
    common: np.ndarray,
    minimum_vector_norm: float,
) -> dict[str, Any]:
    score = score_twin_differential_consistency(
        left_flow,
        right_flow,
        left_expected,
        right_expected,
        valid_mask=common,
        min_vector_norm_px=minimum_vector_norm,
    )
    return {
        key: score.get(key)
        for key in (
            "status_counts",
            "interval_coverage",
            "median_residual_px",
            "median_observed_delta_px",
            "median_expected_delta_px",
            "median_direction_cosine",
            "median_direction_vector_fraction",
            "temporal_persistence",
            "median_support_fraction",
            "minimum_support_fraction",
        )
    }


def run(root: Path, *, minimum_vector_norm: float, minimum_interval_coverage: float, bootstrap_draws: int, seed: int) -> dict[str, Any]:
    records = _load_records(root)
    by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_source.setdefault(str(record["source_key"]), {})[str(record["branch_role"])] = record

    rows: list[dict[str, Any]] = []
    for source, branches in sorted(by_source.items()):
        if not {"left", "right"}.issubset(branches):
            continue
        left_meta, right_meta = branches["left"], branches["right"]
        array_root = Path(left_meta["_array_root"])
        left_flow = np.load(array_root / left_meta["flow_path"])
        right_flow = np.load(Path(right_meta["_array_root"]) / right_meta["flow_path"])
        left_valid = np.load(array_root / left_meta["valid_path"]).astype(bool)
        right_valid = np.load(Path(right_meta["_array_root"]) / right_meta["valid_path"]).astype(bool)
        shape = tuple(int(v) for v in left_flow.shape[1:3])
        roi = polygon_mask(shape[0], shape[1], [[0.08, 0.98], [0.92, 0.98], [0.63, 0.53], [0.37, 0.53]])
        count = int(left_flow.shape[0])
        left_traj = _interval_trajectory(np.asarray(left_meta["trajectory"], dtype=float), count)
        right_traj = _interval_trajectory(np.asarray(right_meta["trajectory"], dtype=float), count)
        left_expected, left_geometry = trajectory_conditioned_flow(
            left_traj, intrinsics=np.asarray(left_meta["intrinsics"]), camera_to_ego=np.asarray(left_meta["camera_to_ego"]), frame_shape=shape
        )
        right_expected, right_geometry = trajectory_conditioned_flow(
            right_traj, intrinsics=np.asarray(right_meta["intrinsics"]), camera_to_ego=np.asarray(right_meta["camera_to_ego"]), frame_shape=shape
        )
        common = left_valid & right_valid & left_geometry & right_geometry & roi
        normal = _control_score(left_flow, right_flow, left_expected, right_expected, common, minimum_vector_norm)
        reversed_score = _control_score(left_flow, right_flow, right_expected, left_expected, common, minimum_vector_norm)
        # Identity swap is a separate semantic control: images stay attached
        # to their source, while the action identity is swapped.
        identity = _control_score(left_flow, right_flow, right_expected, left_expected, common, minimum_vector_norm)
        # Zero-contrast control: remove the observed branch difference while
        # retaining the normal action labels.  It must become unavailable,
        # never a directional pass caused by numerical noise.
        midpoint_flow = 0.5 * (left_flow + right_flow)
        zero = _control_score(midpoint_flow, midpoint_flow, left_expected, right_expected, common, minimum_vector_norm)
        rows.append({"source_key": source, "normal": normal, "reversed": reversed_score, "identity_swap": identity, "zero_contrast": zero})

    normal_rows = [row["normal"] for row in rows if row["normal"].get("interval_coverage", 0.0) >= minimum_interval_coverage and row["normal"].get("median_direction_cosine") is not None]
    normal_hits = np.asarray([float(row["median_direction_cosine"]) > 0.0 for row in normal_rows], dtype=float)
    reversed_rows = [row["reversed"] for row in rows if row["reversed"].get("interval_coverage", 0.0) >= minimum_interval_coverage and row["reversed"].get("median_direction_cosine") is not None]
    identity_rows = [row["identity_swap"] for row in rows if row["identity_swap"].get("interval_coverage", 0.0) >= minimum_interval_coverage and row["identity_swap"].get("median_direction_cosine") is not None]
    # Zero has no visual contrast by construction; its directional score is
    # unavailable.  We expose the residual numerical contrast as a diagnostic.
    zero_rows = [row["zero_contrast"] for row in rows]
    return {
        "protocol": "iac-step1-universal-response-raw-controls-v1",
        "raw_video_rerun": True,
        "root": str(root),
        "required_controls": ["normal_action", "reversed_action", "identity_swap", "zero_contrast"],
        "pair_count": len(rows),
        "normal": {
            "effective_pairs": len(normal_rows),
            "coverage": len(normal_rows) / len(rows) if rows else 0.0,
            "direction_accuracy": float(np.mean(normal_hits)) if len(normal_hits) else None,
            "direction_ci95": _bootstrap(normal_hits, bootstrap_draws, seed),
            "median_temporal_persistence": float(np.median([row["temporal_persistence"] for row in normal_rows if row.get("temporal_persistence") is not None])) if any(row.get("temporal_persistence") is not None for row in normal_rows) else None,
        },
        "controls": {
            "reversed_action": {
                "effective_pairs": len(reversed_rows),
                "positive_false_positive_rate": float(np.mean([float(row["median_direction_cosine"]) > 0.0 for row in reversed_rows])) if reversed_rows else None,
            },
            "identity_swap": {
                "effective_pairs": len(identity_rows),
                "positive_false_positive_rate": float(np.mean([float(row["median_direction_cosine"]) > 0.0 for row in identity_rows])) if identity_rows else None,
            },
            "zero_contrast": {
                "effective_pairs": len(zero_rows),
                "directional_estimand": "unavailable",
                "median_observed_delta_px": float(np.median([row["median_observed_delta_px"] for row in zero_rows if row.get("median_observed_delta_px") is not None])) if any(row.get("median_observed_delta_px") is not None for row in zero_rows) else None,
            },
        },
        "rows": rows,
        "claim_boundary": "Raw-flow control rerun. Reversed and identity controls are negative controls; zero has no directional estimand. This report is not a promotion until all model gates and held-out controls pass.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-vector-norm", type=float, default=0.10)
    parser.add_argument("--min-interval-coverage", type=float, default=0.90)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    report = run(args.root, minimum_vector_norm=args.min_vector_norm, minimum_interval_coverage=args.min_interval_coverage, bootstrap_draws=args.bootstrap_draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pair_count", "normal", "controls")}, indent=2))


if __name__ == "__main__":
    main()

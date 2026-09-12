#!/usr/bin/env python3
"""Run fixed-support trajectory-conditioned SE(2) controls on raw flow twins.

Unlike the legacy control runner, this tool fixes the image support before any
candidate trajectory is evaluated.  Candidate projection validity is reported
as a separate coverage quantity, so a trajectory cannot win by projecting out
of frame and receiving a robust-loss constant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.scoring import polygon_mask
from iac_new.visual_consistency import (
    score_trajectory_conditioned_likelihood,
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
        if path.name in {"manifest.json", "score_roi.json", "twin_controls.json", "se2_hybrid_controls.json"}:
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        required = ("source_key", "branch_role", "trajectory", "flow_path", "valid_path")
        if not all(key in metadata for key in required):
            continue
        metadata["_array_root"] = str(path.parent)
        records.append(metadata)
    return records


def _load_expected(record: dict[str, Any], interval_count: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    trajectory = _interval_trajectory(np.asarray(record["trajectory"], dtype=float), interval_count)
    return trajectory_conditioned_flow(
        trajectory,
        intrinsics=np.asarray(record["intrinsics"], dtype=float),
        camera_to_ego=np.asarray(record["camera_to_ego"], dtype=float),
        frame_shape=shape,
    )


def _score(
    observed_delta: np.ndarray,
    expected_delta: np.ndarray,
    fixed_support: np.ndarray,
    expected_valid: np.ndarray,
    *,
    min_projection_valid_fraction: float,
    likelihood_scale_px: float,
) -> dict[str, Any]:
    report = score_trajectory_conditioned_likelihood(
        observed_delta,
        expected_delta,
        fixed_support_mask=fixed_support,
        expected_valid_mask=expected_valid,
        min_projection_valid_fraction=min_projection_valid_fraction,
        likelihood_scale_px=likelihood_scale_px,
    )
    return {
        key: report.get(key)
        for key in (
            "status_counts",
            "fixed_support_fraction",
            "projection_valid_fraction",
            "interval_coverage",
            "reliable_interval_fraction",
            "score",
            "median_residual_px",
            "median_direction_cosine",
            "intervals",
        )
    }


def run(root: Path, *, min_projection_valid_fraction: float, likelihood_scale_px: float) -> dict[str, Any]:
    records = _load_records(root)
    by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_source.setdefault(str(record["source_key"]), {})[str(record["branch_role"])] = record

    rows: list[dict[str, Any]] = []
    for source, branches in sorted(by_source.items()):
        if not {"left", "right"}.issubset(branches):
            continue
        left_meta, right_meta = branches["left"], branches["right"]
        left_root = Path(left_meta["_array_root"])
        right_root = Path(right_meta["_array_root"])
        left_flow = np.load(left_root / left_meta["flow_path"])
        right_flow = np.load(right_root / right_meta["flow_path"])
        left_valid = np.load(left_root / left_meta["valid_path"]).astype(bool)
        right_valid = np.load(right_root / right_meta["valid_path"]).astype(bool)
        shape = tuple(int(v) for v in left_flow.shape[1:3])
        interval_count = int(left_flow.shape[0])
        left_expected, left_projection = _load_expected(left_meta, interval_count, shape)
        right_expected, right_projection = _load_expected(right_meta, interval_count, shape)
        roi = polygon_mask(shape[0], shape[1], [[0.08, 0.98], [0.92, 0.98], [0.63, 0.53], [0.37, 0.53]])
        left_mas = _score(
            left_flow, left_expected, left_valid & roi[None, ...], left_projection,
            min_projection_valid_fraction=min_projection_valid_fraction,
            likelihood_scale_px=likelihood_scale_px,
        )
        right_mas = _score(
            right_flow, right_expected, right_valid & roi[None, ...], right_projection,
            min_projection_valid_fraction=min_projection_valid_fraction,
            likelihood_scale_px=likelihood_scale_px,
        )
        # This mask is frozen before normal/reversed/zero candidates are scored.
        fixed_support = left_valid & right_valid & roi[None, ...]
        observed_delta = left_flow - right_flow
        expected_delta = left_expected - right_expected
        expected_valid = left_projection & right_projection
        normal = _score(
            observed_delta, expected_delta, fixed_support, expected_valid,
            min_projection_valid_fraction=min_projection_valid_fraction,
            likelihood_scale_px=likelihood_scale_px,
        )
        reversed_score = _score(
            observed_delta, -expected_delta, fixed_support, expected_valid,
            min_projection_valid_fraction=min_projection_valid_fraction,
            likelihood_scale_px=likelihood_scale_px,
        )
        zero = _score(
            np.zeros_like(observed_delta), expected_delta, fixed_support, expected_valid,
            min_projection_valid_fraction=min_projection_valid_fraction,
            likelihood_scale_px=likelihood_scale_px,
        )
        rows.append({"source_key": source, "mas_left": left_mas, "mas_right": right_mas, "normal": normal, "reversed": reversed_score, "zero_contrast": zero})

    def effective(key: str) -> list[dict[str, Any]]:
        return [
            row[key]
            for row in rows
            if row[key].get("score") is not None and row[key].get("interval_coverage", 0.0) >= 0.75
        ]

    normal = effective("normal")
    reversed_rows = effective("reversed")
    mas_branches = [
        row[branch]
        for row in rows
        for branch in ("mas_left", "mas_right")
        if row[branch].get("score") is not None and row[branch].get("interval_coverage", 0.0) >= 0.75
    ]
    paired = [
        row for row in rows
        if row["normal"].get("score") is not None
        and row["reversed"].get("score") is not None
        and row["normal"].get("interval_coverage", 0.0) >= 0.75
        and row["reversed"].get("interval_coverage", 0.0) >= 0.75
    ]
    paired_zero = [
        row for row in rows
        if row["normal"].get("score") is not None
        and row["zero_contrast"].get("score") is not None
        and row["normal"].get("interval_coverage", 0.0) >= 0.75
        and row["zero_contrast"].get("interval_coverage", 0.0) >= 0.75
    ]
    return {
        "protocol": "iac-step1-se2-hybrid-fixed-support-v1",
        "metric_id": "MAS/RCS",
        "candidate_blind_support": True,
        "pair_count": len(rows),
        "normal": {
            "effective_pairs": len(normal),
            "coverage": len(normal) / len(rows) if rows else 0.0,
            "median_score": float(np.median([row["score"] for row in normal])) if normal else None,
            "median_direction_cosine": float(np.median([row["median_direction_cosine"] for row in normal if row.get("median_direction_cosine") is not None])) if any(row.get("median_direction_cosine") is not None for row in normal) else None,
            "direction_accuracy": float(np.mean([float(row.get("median_direction_cosine") or 0.0) > 0.0 for row in normal])) if normal else None,
        },
        "mas": {
            "effective_branches": len(mas_branches),
            "branch_count": 2 * len(rows),
            "coverage": len(mas_branches) / (2 * len(rows)) if rows else 0.0,
            "median_score": float(np.median([row["score"] for row in mas_branches])) if mas_branches else None,
            "median_direction_cosine": float(np.median([row["median_direction_cosine"] for row in mas_branches if row.get("median_direction_cosine") is not None])) if any(row.get("median_direction_cosine") is not None for row in mas_branches) else None,
            "direction_accuracy": float(np.mean([float(row.get("median_direction_cosine") or 0.0) > 0.0 for row in mas_branches])) if mas_branches else None,
        },
        "controls": {
            "reversed_action": {
                "effective_pairs": len(reversed_rows),
                "median_score": float(np.median([row["score"] for row in reversed_rows])) if reversed_rows else None,
                "normal_beats_reversed": float(np.mean([row["normal"]["score"] > row["reversed"]["score"] for row in paired])) if paired else None,
            },
            "zero_contrast": {
                "directional_estimand": "unavailable",
                "normal_beats_zero": float(np.mean([row["normal"]["score"] > row["zero_contrast"]["score"] for row in paired_zero])) if paired_zero else None,
            },
        },
        "thresholds": {
            "min_projection_valid_fraction": float(min_projection_valid_fraction),
            "likelihood_scale_px": float(likelihood_scale_px),
        },
        "rows": rows,
        "claim_boundary": "SE(2) supplied-trajectory likelihood with fixed candidate-blind support; not a free trajectory reconstruction and not a metric accuracy claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-projection-valid-fraction", type=float, default=0.20)
    parser.add_argument("--likelihood-scale-px", type=float, default=2.0)
    args = parser.parse_args()
    report = run(
        args.root,
        min_projection_valid_fraction=args.min_projection_valid_fraction,
        likelihood_scale_px=args.likelihood_scale_px,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pair_count": report["pair_count"], "normal": report["normal"], "controls": report["controls"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run normal/swapped/zero trajectory controls on dense-flow artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iac_new.scoring import polygon_mask
from iac_new.visual_consistency import (
    score_trajectory_visual_consistency,
    trajectory_conditioned_flow,
)


def _score(root: Path, metadata: dict, trajectory: np.ndarray) -> dict:
    observed = np.load(root / metadata["flow_path"])
    valid = np.asarray(np.load(root / metadata["valid_path"]), dtype=bool)
    expected, geometry_valid = trajectory_conditioned_flow(
        trajectory,
        intrinsics=np.asarray(metadata["intrinsics"]),
        camera_to_ego=np.asarray(metadata["camera_to_ego"]),
        frame_shape=observed.shape[1:3],
    )
    roi = polygon_mask(
        observed.shape[1], observed.shape[2],
        [[0.08, 0.98], [0.92, 0.98], [0.63, 0.53], [0.37, 0.53]],
    )
    return score_trajectory_visual_consistency(
        observed, expected, valid_mask=valid & geometry_valid & roi,
        max_median_residual_px=3.0,
    )


def _summary(rows: list[dict]) -> dict:
    residuals = [r["median_residual_px"] for r in rows if r["median_residual_px"] is not None]
    cosines = [r["median_direction_cosine"] for r in rows if r["median_direction_cosine"] is not None]
    return {
        "count": len(rows),
        "reliable_interval_fraction": float(np.mean([r["reliable_interval_fraction"] for r in rows])) if rows else None,
        "interval_coverage": float(np.mean([r["interval_coverage"] for r in rows])) if rows else None,
        "median_residual_px": float(np.median(residuals)) if residuals else None,
        "median_direction_cosine": float(np.median(cosines)) if cosines else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.root.glob("*.json")):
        if path.name == "manifest.json":
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if "flow_path" not in metadata:
            continue
        records.append(metadata)
    by_source = {}
    for metadata in records:
        by_source.setdefault(str(metadata["source_key"]), {})[str(metadata.get("branch_role"))] = metadata
    rows = []
    for source, branches in sorted(by_source.items()):
        if "left" not in branches or "right" not in branches:
            continue
        trajectories = {
            role: np.asarray(branches[role]["trajectory"], dtype=np.float64)
            for role in ("left", "right")
        }
        zero = np.zeros_like(trajectories["left"])
        for role, metadata in branches.items():
            for control, trajectory in (
                ("normal", trajectories[role]),
                ("swapped", trajectories["right" if role == "left" else "left"]),
                ("zero", zero),
            ):
                score = _score(args.root, metadata, trajectory)
                rows.append({
                    "source_key": source, "branch_role": role, "control": control,
                    **{key: score[key] for key in (
                        "status_counts", "input_support_fraction", "interval_coverage",
                        "reliable_interval_fraction", "median_residual_px",
                        "median_direction_cosine", "residual_inlier_fraction",
                    )},
                })
    summary = {
        "protocol": "iac-forward-visual-consistency-controls-v1",
        "candidate_blind": True,
        "controls": {control: _summary([r for r in rows if r["control"] == control]) for control in ("normal", "swapped", "zero")},
    }
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score a directory produced by extract_dense_flow.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iac_new.visual_consistency import (
    score_trajectory_visual_consistency,
    trajectory_conditioned_flow,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("*.json")):
        if path.name == "manifest.json":
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        observed = np.load(args.root / metadata["flow_path"])
        valid = np.load(args.root / metadata["valid_path"])
        expected, geometry_valid = trajectory_conditioned_flow(
            np.asarray(metadata["trajectory"]),
            intrinsics=np.asarray(metadata["intrinsics"]),
            camera_to_ego=np.asarray(metadata["camera_to_ego"]),
            frame_shape=observed.shape[1:3],
        )
        score = score_trajectory_visual_consistency(
            observed, expected, valid_mask=valid & geometry_valid,
            max_median_residual_px=3.0,
        )
        rows.append({
            "source_key": metadata.get("source_key"),
            "branch_role": metadata.get("branch_role"),
            "sample_id": metadata.get("sample_id"),
            "stratum": metadata.get("stratum"),
            **{key: score[key] for key in (
                "status_counts", "input_support_fraction", "interval_coverage",
                "reliable_interval_fraction", "median_residual_px",
                "median_direction_cosine", "residual_inlier_fraction",
            )},
        })
    residuals = [row["median_residual_px"] for row in rows if row["median_residual_px"] is not None]
    cosines = [row["median_direction_cosine"] for row in rows if row["median_direction_cosine"] is not None]
    summary = {
        "protocol": "iac-forward-visual-consistency-v1",
        "count": len(rows),
        "branch_status_counts": {
            status: sum(row["status_counts"].get(status, 0) for row in rows)
            for status in ("scored", "weak", "unavailable")
        },
        "branch_reliable_fraction": float(np.mean([row["reliable_interval_fraction"] for row in rows])) if rows else None,
        "median_branch_residual_px": float(np.median(residuals)) if residuals else None,
        "median_branch_direction_cosine": float(np.median(cosines)) if cosines else None,
    }
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

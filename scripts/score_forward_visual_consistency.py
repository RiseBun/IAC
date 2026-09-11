#!/usr/bin/env python3
"""Score a supplied trajectory against dense observed optical flow."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-flow", type=Path, required=True, help=".npy [T,H,W,2]")
    parser.add_argument("--trajectory", type=Path, required=True, help=".npy [T,3]")
    parser.add_argument("--intrinsics", type=Path, required=True, help=".npy [3,3]")
    parser.add_argument("--camera-to-ego", type=Path, required=True, help=".npy [4,4]")
    parser.add_argument("--depth", type=Path, help="optional .npy [T,H,W]")
    parser.add_argument("--valid-mask", type=Path, help="optional .npy [T,H,W]")
    parser.add_argument("--model", choices=["ground_plane", "depth"], default="ground_plane")
    parser.add_argument("--min-valid-fraction", type=float, default=0.20)
    parser.add_argument("--max-median-residual-px", type=float, default=3.0)
    parser.add_argument("--min-direction-cosine", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observed = np.load(args.observed_flow)
    trajectory = np.load(args.trajectory)
    expected, geometry_valid = trajectory_conditioned_flow(
        trajectory,
        intrinsics=np.load(args.intrinsics),
        camera_to_ego=np.load(args.camera_to_ego),
        frame_shape=tuple(observed.shape[1:3]),
        depths_m=np.load(args.depth) if args.depth else None,
        model=args.model,
    )
    valid = geometry_valid
    if args.valid_mask:
        valid &= np.asarray(np.load(args.valid_mask), dtype=bool)
    report = score_trajectory_visual_consistency(
        observed,
        expected,
        valid_mask=valid,
        min_valid_fraction=args.min_valid_fraction,
        max_median_residual_px=args.max_median_residual_px,
        min_direction_cosine=args.min_direction_cosine,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "intervals"}, indent=2))


if __name__ == "__main__":
    main()

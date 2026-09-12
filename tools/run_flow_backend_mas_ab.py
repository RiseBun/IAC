#!/usr/bin/env python3
"""Compare flow backends under the frozen SE(2)-conditioned MAS scorer.

The tool intentionally evaluates the same generated image sequences with the
same geometry, support mask, and action trajectory.  Only the optical-flow
backend changes.  It is an estimator A/B, not a new metric definition.
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from iac_new.flow import RaftFlowExtractor
from iac_new.scoring import polygon_mask
from iac_new.sea_raft_flow import SeaRaftFlowExtractor
from iac_new.visual_consistency import (
    score_trajectory_conditioned_likelihood,
    trajectory_conditioned_flow,
)


def _trajectory(value: Any, interval_count: int) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2 or raw.shape[0] != 3:
        raise ValueError(f"action trajectory must be [3,N], got {raw.shape}")
    knots = raw.T
    if len(knots) == interval_count:
        return knots
    if len(knots) == 2 * interval_count:
        return knots[[1, 3, 5, 7]]
    indices = np.linspace(0, len(knots) - 1, interval_count).round().astype(int)
    return knots[indices]


def _extractor(name: str, args: argparse.Namespace):
    if name == "raft_large":
        return RaftFlowExtractor(
            model_size="large",
            device=args.device,
            updates=args.iters,
            batch_size=1,
            forward_backward=args.use_forward_backward,
            fb_abs_threshold_px=args.fb_abs_threshold_px,
            fb_relative_threshold=args.fb_relative_threshold,
            checkpoint=args.raft_checkpoint,
        )
    return SeaRaftFlowExtractor(
        checkpoint=args.sea_checkpoint,
        device=args.device,
        iters=args.iters,
        forward_backward=args.use_forward_backward,
        fb_abs_threshold_px=args.fb_abs_threshold_px,
        fb_relative_threshold=args.fb_relative_threshold,
    )


def _run_backend(name: str, manifest: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    extractor = _extractor(name, args)
    rows: list[dict[str, Any]] = []
    for entry in manifest:
        source = Path(entry["source_sample"])
        with source.open("rb") as handle:
            sample = pickle.load(handle)
        metadata = sample["metadata"]
        intrinsics = np.asarray(metadata["camera_intrinsic"], dtype=np.float64)
        camera_to_ego = np.asarray(metadata["camera_to_ego"], dtype=np.float64)
        # The manifest stores four future frames, while the action trajectory
        # is anchored at the current frame.  Reinsert that current frame from
        # the source pickle so flow intervals and trajectory knots have the
        # same origin as the formal protocol.
        with tempfile.TemporaryDirectory(prefix="iac_flow_ab_") as temporary:
            current_path = Path(temporary) / "current.png"
            current = np.asarray(sample["images"][-1])
            current = cv2.resize(current, (args.target_width, args.target_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(current_path), cv2.cvtColor(current, cv2.COLOR_RGB2BGR))
            paths = [str(current_path)] + [str(Path(path)) for path in entry["future_images"]]
            observation = extractor.observe(
                paths,
                intrinsics,
                np.zeros(5, dtype=np.float64),
                (args.target_width, args.target_height),
                inference_size=(args.inference_width, args.inference_height),
                intrinsics_source_size=(args.source_width, args.source_height),
                return_uncertainty=True,
            )
        flow = np.asarray(observation.forward, dtype=np.float64)
        height, width = flow.shape[1:3]
        roi = polygon_mask(height, width, [[0.08, 0.98], [0.92, 0.98], [0.63, 0.53], [0.37, 0.53]])
        valid = (
            np.asarray(observation.consistency_masks, dtype=bool)
            if observation.consistency_masks is not None
            else np.isfinite(flow).all(axis=-1)
        )
        trajectory = _trajectory(entry["predicted_action_trajectory"], flow.shape[0])
        expected, expected_valid = trajectory_conditioned_flow(
            trajectory,
            intrinsics=np.asarray(observation.intrinsics, dtype=np.float64),
            camera_to_ego=camera_to_ego,
            frame_shape=(height, width),
        )
        report = score_trajectory_conditioned_likelihood(
            flow,
            expected,
            fixed_support_mask=valid & roi[None, ...],
            expected_valid_mask=expected_valid,
            min_projection_valid_fraction=args.min_projection_valid_fraction,
            likelihood_scale_px=args.likelihood_scale_px,
        )
        uncertainty = observation.refinement_uncertainty
        rows.append({
            "sample_index": int(entry["sample_index"]),
            "source_key": str(metadata.get("source_key", entry["sample_index"])),
            "mas": {
                "score": report.get("score"),
                "coverage": report.get("interval_coverage"),
                "reliable_interval_fraction": report.get("reliable_interval_fraction"),
                "median_residual_px": report.get("median_residual_px"),
                "median_direction_cosine": report.get("median_direction_cosine"),
                "status_counts": report.get("status_counts"),
                "projection_valid_fraction": report.get("projection_valid_fraction"),
            },
            "flow": {
                "median_magnitude_px": float(np.nanmedian(np.linalg.norm(flow, axis=-1))),
                "median_uncertainty_px": float(np.nanmedian(uncertainty)) if uncertainty is not None else None,
            },
        })
    effective = [row for row in rows if row["mas"]["score"] is not None and float(row["mas"]["coverage"] or 0.0) >= args.minimum_interval_coverage]
    return {
        "backend": name,
        "sample_count": len(rows),
        "effective_samples": len(effective),
        "coverage": len(effective) / len(rows) if rows else 0.0,
        "median_score": float(np.median([row["mas"]["score"] for row in effective])) if effective else None,
        "median_direction_cosine": float(np.median([row["mas"]["median_direction_cosine"] for row in effective if row["mas"].get("median_direction_cosine") is not None])) if any(row["mas"].get("median_direction_cosine") is not None for row in effective) else None,
        "median_residual_px": float(np.median([row["mas"]["median_residual_px"] for row in effective])) if effective else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raft-checkpoint", required=True)
    parser.add_argument("--sea-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--target-width", type=int, default=448)
    parser.add_argument("--target-height", type=int, default=256)
    parser.add_argument("--inference-width", type=int, default=512)
    parser.add_argument("--inference-height", type=int, default=288)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--fb-abs-threshold-px", type=float, default=1.5)
    parser.add_argument("--fb-relative-threshold", type=float, default=0.05)
    parser.add_argument("--use-forward-backward", action="store_true")
    parser.add_argument("--min-projection-valid-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-interval-coverage", type=float, default=0.75)
    parser.add_argument("--likelihood-scale-px", type=float, default=2.0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = {
        "protocol": "iac-flow-backend-mas-ab-v1",
        "scorer": "iac-trajectory-conditioned-likelihood-v1",
        "manifest": str(args.manifest),
        "backends": [
            _run_backend("raft_large", manifest, args),
            _run_backend("sea_raft", manifest, args),
        ],
        "claim_boundary": "Estimator A/B only; same generated images, trajectory, support mask, geometry, and MAS scorer. No RCS or cross-model promotion claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"backends": [{k: value for k, value in item.items() if k != "rows"} for item in report["backends"]]}, indent=2))


if __name__ == "__main__":
    main()

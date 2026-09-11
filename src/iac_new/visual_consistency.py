"""Forward, trajectory-conditioned visual consistency scoring.

The scorer tests a supplied trajectory as a hypothesis.  It never optimizes a
free trajectory to explain the video, so a zero-motion solution cannot win
simply because out-of-frame projections are assigned a robust loss.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import (
    adjacent_camera_transforms,
    candidate_camera_poses,
    ground_plane_homography,
    homography_flow,
    rigid_flow_from_depth,
)


def trajectory_conditioned_flow(
    trajectory: np.ndarray,
    *,
    intrinsics: np.ndarray,
    camera_to_ego: np.ndarray,
    frame_shape: tuple[int, int],
    depths_m: np.ndarray | None = None,
    model: str = "ground_plane",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate expected dense image flow for a supplied ego trajectory.

    ``trajectory`` is ``[T,3]`` with ``(x_m, y_m, yaw_rad)`` at future knots;
    the returned flow has ``[T,H,W,2]`` intervals.  ``model=ground_plane`` is
    metric-free apart from the camera height encoded by ``camera_to_ego``;
    ``model=depth`` uses a per-interval depth map and is therefore an optional
    diagnostic adapter.
    """
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("trajectory must have shape [T,3]")
    height, width = (int(frame_shape[0]), int(frame_shape[1]))
    if min(height, width) <= 0:
        raise ValueError("frame_shape must be positive")
    if model not in {"ground_plane", "depth"}:
        raise ValueError("model must be ground_plane or depth")
    if model == "depth":
        if depths_m is None:
            raise ValueError("depth model requires depths_m")
        depth = np.asarray(depths_m, dtype=np.float64)
        if depth.shape != (len(values), height, width):
            raise ValueError("depths_m must have shape [T,H,W]")
    poses = candidate_camera_poses(values, np.asarray(camera_to_ego, dtype=np.float64))
    transforms = adjacent_camera_transforms(poses)
    K = np.asarray(intrinsics, dtype=np.float64)
    flows: list[np.ndarray] = []
    valid_maps: list[np.ndarray] = []
    for index, transform in enumerate(transforms):
        if model == "depth":
            flow, valid = rigid_flow_from_depth(depth[index], K, transform)
        else:
            homography = ground_plane_homography(K, transform, poses[index])
            flow, valid = homography_flow(homography, height, width)
        flows.append(flow)
        valid_maps.append(valid)
    return np.stack(flows), np.stack(valid_maps)


def _robust_median_scale(values: np.ndarray, floor: float = 1e-3) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float(floor)
    median = float(np.median(finite))
    mad = float(1.4826 * np.median(np.abs(finite - median)))
    return max(float(floor), mad)


def score_trajectory_visual_consistency(
    observed_flows: np.ndarray,
    expected_flows: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    min_valid_fraction: float = 0.20,
    max_median_residual_px: float = 3.0,
    min_direction_cosine: float = 0.0,
    residual_inlier_threshold_px: float = 2.0,
) -> dict[str, Any]:
    """Score a supplied trajectory against observed video flow.

    Invalid pixels are excluded from the evidence denominator.  An interval is
    not silently treated as zero motion: if its valid support is too small it
    becomes ``unavailable``.  Thresholds are protocol inputs and should be
    frozen on a real-video calibration split.
    """
    observed = np.asarray(observed_flows, dtype=np.float64)
    expected = np.asarray(expected_flows, dtype=np.float64)
    if observed.shape != expected.shape or observed.ndim != 4 or observed.shape[-1] != 2:
        raise ValueError("observed_flows and expected_flows must match [T,H,W,2]")
    intervals, height, width = observed.shape[:3]
    if valid_mask is None:
        mask = np.ones((intervals, height, width), dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != (intervals, height, width):
            raise ValueError("valid_mask must match [T,H,W]")
    rows: list[dict[str, Any]] = []
    for index in range(intervals):
        finite = mask[index] & np.isfinite(observed[index]).all(axis=-1) & np.isfinite(expected[index]).all(axis=-1)
        support = int(finite.sum())
        support_fraction = float(support / max(height * width, 1))
        row: dict[str, Any] = {
            "interval_index": index,
            "support_pixels": support,
            "support_fraction": support_fraction,
        }
        if support == 0 or support_fraction < float(min_valid_fraction):
            row.update({"status": "unavailable", "reason": "insufficient_valid_support"})
            rows.append(row)
            continue
        residual = observed[index] - expected[index]
        residual_norm = np.linalg.norm(residual, axis=-1)[finite]
        observed_norm = np.linalg.norm(observed[index], axis=-1)[finite]
        expected_norm = np.linalg.norm(expected[index], axis=-1)[finite]
        dot = np.sum(observed[index][finite] * expected[index][finite], axis=-1)
        cosine = dot / np.maximum(observed_norm * expected_norm, 1e-6)
        median_residual = float(np.median(residual_norm))
        robust_scale = _robust_median_scale(residual_norm)
        inlier = residual_norm <= float(residual_inlier_threshold_px)
        row.update({
            "status": "scored",
            "median_residual_px": median_residual,
            "robust_residual_scale_px": robust_scale,
            "residual_inlier_fraction": float(np.mean(inlier)),
            "direction_cosine": float(np.mean(cosine)),
            "observed_flow_median_px": float(np.median(observed_norm)),
            "expected_flow_median_px": float(np.median(expected_norm)),
        })
        if median_residual > float(max_median_residual_px):
            row["status"] = "weak"
            row["reason"] = "residual_above_threshold"
        elif float(np.mean(cosine)) < float(min_direction_cosine):
            row["status"] = "weak"
            row["reason"] = "direction_below_threshold"
        rows.append(row)
    scored = [row for row in rows if row["status"] == "scored"]
    usable = [row for row in rows if row["status"] in {"scored", "weak"}]
    return {
        "protocol": "iac-forward-visual-consistency-v1",
        "candidate_blind": True,
        "metric_reconstruction_used": False,
        "interval_count": intervals,
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("scored", "weak", "unavailable")
        },
        "input_support_fraction": float(np.mean([row["support_fraction"] for row in rows])) if rows else None,
        "interval_coverage": len(usable) / intervals if intervals else None,
        "reliable_interval_fraction": len(scored) / intervals if intervals else None,
        "median_residual_px": float(np.median([row["median_residual_px"] for row in usable])) if usable else None,
        "median_direction_cosine": float(np.median([row["direction_cosine"] for row in usable])) if usable else None,
        "residual_inlier_fraction": float(np.median([row["residual_inlier_fraction"] for row in usable])) if usable else None,
        "thresholds": {
            "min_valid_fraction": float(min_valid_fraction),
            "max_median_residual_px": float(max_median_residual_px),
            "min_direction_cosine": float(min_direction_cosine),
            "residual_inlier_threshold_px": float(residual_inlier_threshold_px),
        },
        "warning": "Forward visual consistency tests a supplied trajectory; unavailable intervals are not zero-filled.",
        "intervals": rows,
    }


def score_twin_differential_consistency(
    observed_left: np.ndarray,
    observed_right: np.ndarray,
    expected_left: np.ndarray,
    expected_right: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    min_valid_fraction: float = 0.05,
    min_vector_norm_px: float = 0.10,
) -> dict[str, Any]:
    """Compare a same-source branch difference against a supplied trajectory difference.

    Common-mode motion is removed before scoring.  The function is intentionally
    differential: it does not estimate either branch trajectory and it reports
    low-support intervals as unavailable.  ``expected_left - expected_right``
    is the normal control; callers can pass its negation for the reversed
    control and zeros for an identity/zero-difference control.
    """
    left = np.asarray(observed_left, dtype=np.float64)
    right = np.asarray(observed_right, dtype=np.float64)
    expected = np.asarray(expected_left, dtype=np.float64) - np.asarray(expected_right, dtype=np.float64)
    if left.shape != right.shape or left.shape != expected.shape or left.ndim != 4 or left.shape[-1] != 2:
        raise ValueError("twin flow arrays must match [T,H,W,2]")
    intervals, height, width = left.shape[:3]
    if valid_mask is None:
        mask = np.ones((intervals, height, width), dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != (intervals, height, width):
            raise ValueError("valid_mask must match [T,H,W]")
    observed = left - right
    rows: list[dict[str, Any]] = []
    for index in range(intervals):
        finite = mask[index] & np.isfinite(observed[index]).all(axis=-1) & np.isfinite(expected[index]).all(axis=-1)
        support = int(finite.sum())
        support_fraction = float(support / max(height * width, 1))
        row: dict[str, Any] = {"interval_index": index, "support_pixels": support, "support_fraction": support_fraction}
        if support == 0 or support_fraction < float(min_valid_fraction):
            row.update({"status": "unavailable", "reason": "insufficient_common_support"})
            rows.append(row)
            continue
        observed_values = observed[index][finite]
        expected_values = expected[index][finite]
        residual_norm = np.linalg.norm(observed_values - expected_values, axis=-1)
        observed_norm = np.linalg.norm(observed_values, axis=-1)
        expected_norm = np.linalg.norm(expected_values, axis=-1)
        vector_valid = (observed_norm >= float(min_vector_norm_px)) & (expected_norm >= float(min_vector_norm_px))
        cosine = (
            np.sum(observed_values[vector_valid] * expected_values[vector_valid], axis=-1)
            / np.maximum(observed_norm[vector_valid] * expected_norm[vector_valid], 1e-8)
            if np.any(vector_valid) else np.asarray([], dtype=np.float64)
        )
        row.update({
            "status": "scored",
            "median_residual_px": float(np.median(residual_norm)),
            "median_observed_delta_px": float(np.median(observed_norm)),
            "median_expected_delta_px": float(np.median(expected_norm)),
            "direction_cosine": float(np.mean(cosine)) if len(cosine) else None,
            "direction_vector_fraction": float(np.mean(vector_valid)),
        })
        rows.append(row)
    scored = [row for row in rows if row["status"] == "scored"]
    return {
        "protocol": "iac-twin-differential-forward-consistency-v1",
        "candidate_blind": True,
        "metric_reconstruction_used": False,
        "interval_count": intervals,
        "status_counts": {status: sum(row["status"] == status for row in rows) for status in ("scored", "unavailable")},
        "interval_coverage": len(scored) / intervals if intervals else None,
        "median_residual_px": float(np.median([row["median_residual_px"] for row in scored])) if scored else None,
        "median_observed_delta_px": float(np.median([row["median_observed_delta_px"] for row in scored])) if scored else None,
        "median_expected_delta_px": float(np.median([row["median_expected_delta_px"] for row in scored])) if scored else None,
        "median_direction_cosine": float(np.median([row["direction_cosine"] for row in scored if row["direction_cosine"] is not None])) if any(row["direction_cosine"] is not None for row in scored) else None,
        "median_direction_vector_fraction": float(np.median([row["direction_vector_fraction"] for row in scored])) if scored else None,
        "rows": rows,
    }

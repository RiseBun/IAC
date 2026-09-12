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
        "metric_id": "MAS",
        "metric": "Motion Alignment Score",
        "legacy_metric": "CFAC-S",
        "metric_definition": "single_branch_visual_motion_vs_action_conditioned_structure",
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


def score_trajectory_conditioned_likelihood(
    observed_flows: np.ndarray,
    expected_flows: np.ndarray,
    *,
    fixed_support_mask: np.ndarray,
    expected_valid_mask: np.ndarray | None = None,
    observed_valid_mask: np.ndarray | None = None,
    min_fixed_support_fraction: float = 0.20,
    min_projection_valid_fraction: float = 0.20,
    likelihood_scale_px: float = 2.0,
    residual_inlier_threshold_px: float = 2.0,
    min_direction_cosine: float = 0.0,
) -> dict[str, Any]:
    """Score a supplied trajectory with a fixed, candidate-blind support set.

    This is the production-facing SE(2) primitive.  ``fixed_support_mask`` is
    chosen before a candidate trajectory is evaluated (for example ROI and
    observed-flow validity).  The candidate's projection validity is reported
    separately and never changes the input denominator.  A candidate that
    projects too few of the fixed points is ``unavailable``; it is not given a
    robust-loss value and cannot win by collapsing to a zero trajectory.

    The returned ``likelihood`` is an image-space ordering score, not a metric
    reconstruction.  It is deliberately monotone in residual and direction
    agreement so it can be used for MAS/RCS controls while retaining the raw
    diagnostics needed for audit.
    """
    observed = np.asarray(observed_flows, dtype=np.float64)
    expected = np.asarray(expected_flows, dtype=np.float64)
    if observed.shape != expected.shape or observed.ndim != 4 or observed.shape[-1] != 2:
        raise ValueError("observed_flows and expected_flows must match [T,H,W,2]")
    intervals, height, width = observed.shape[:3]
    fixed = np.asarray(fixed_support_mask, dtype=bool)
    if fixed.shape != (intervals, height, width):
        raise ValueError("fixed_support_mask must match [T,H,W]")
    observed_valid = (
        np.ones_like(fixed, dtype=bool)
        if observed_valid_mask is None
        else np.asarray(observed_valid_mask, dtype=bool)
    )
    if observed_valid.shape != fixed.shape:
        raise ValueError("observed_valid_mask must match [T,H,W]")
    expected_valid = (
        np.isfinite(expected).all(axis=-1)
        if expected_valid_mask is None
        else np.asarray(expected_valid_mask, dtype=bool)
    )
    if expected_valid.shape != fixed.shape:
        raise ValueError("expected_valid_mask must match [T,H,W]")
    if likelihood_scale_px <= 0.0:
        raise ValueError("likelihood_scale_px must be positive")

    rows: list[dict[str, Any]] = []
    for index in range(intervals):
        input_mask = fixed[index] & observed_valid[index]
        input_mask &= np.isfinite(observed[index]).all(axis=-1)
        input_count = int(input_mask.sum())
        input_fraction = float(input_count / max(height * width, 1))
        row: dict[str, Any] = {
            "interval_index": index,
            "fixed_support_pixels": input_count,
            "fixed_support_fraction": input_fraction,
        }
        if input_count == 0 or input_fraction < float(min_fixed_support_fraction):
            row.update({"status": "unavailable", "reason": "insufficient_fixed_support"})
            rows.append(row)
            continue
        projected = input_mask & expected_valid[index] & np.isfinite(expected[index]).all(axis=-1)
        projected_count = int(projected.sum())
        projection_fraction = float(projected_count / max(input_count, 1))
        row.update({
            "projected_support_pixels": projected_count,
            "projection_valid_fraction": projection_fraction,
        })
        if projected_count == 0 or projection_fraction < float(min_projection_valid_fraction):
            row.update({"status": "unavailable", "reason": "insufficient_candidate_projection"})
            rows.append(row)
            continue
        residual = observed[index] - expected[index]
        residual_norm = np.linalg.norm(residual, axis=-1)[projected]
        observed_norm = np.linalg.norm(observed[index], axis=-1)[projected]
        expected_norm = np.linalg.norm(expected[index], axis=-1)[projected]
        dot = np.sum(observed[index][projected] * expected[index][projected], axis=-1)
        vector_valid = (observed_norm > 1e-6) & (expected_norm > 1e-6)
        cosine = dot[vector_valid] / np.maximum(
            observed_norm[vector_valid] * expected_norm[vector_valid], 1e-8
        )
        direction = float(np.mean(cosine)) if len(cosine) else None
        residual_median = float(np.median(residual_norm))
        likelihood = float(np.exp(-residual_median / float(likelihood_scale_px)))
        if direction is not None:
            likelihood *= float(np.clip((direction + 1.0) / 2.0, 0.0, 1.0))
        row.update({
            "status": "scored",
            "median_residual_px": residual_median,
            "residual_inlier_fraction": float(np.mean(residual_norm <= float(residual_inlier_threshold_px))),
            "direction_cosine": direction,
            "likelihood": likelihood,
            "observed_flow_median_px": float(np.median(observed_norm)),
            "expected_flow_median_px": float(np.median(expected_norm)),
        })
        if direction is not None and direction < float(min_direction_cosine):
            row["status"] = "weak"
            row["reason"] = "direction_below_threshold"
        rows.append(row)

    scored = [row for row in rows if row["status"] == "scored"]
    usable = [row for row in rows if row["status"] in {"scored", "weak"}]
    return {
        "protocol": "iac-trajectory-conditioned-likelihood-v1",
        "metric_id": "MAS",
        "metric": "Motion Alignment Score",
        "legacy_metric": "CFAC-S",
        "metric_definition": "fixed_support_trajectory_conditioned_visual_likelihood",
        "candidate_blind_support": True,
        "metric_reconstruction_used": False,
        "interval_count": intervals,
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("scored", "weak", "unavailable")
        },
        "fixed_support_fraction": float(np.mean([row["fixed_support_fraction"] for row in rows])) if rows else None,
        "projection_valid_fraction": float(np.mean([row.get("projection_valid_fraction", 0.0) for row in rows])) if rows else None,
        "interval_coverage": len(usable) / intervals if intervals else None,
        "reliable_interval_fraction": len(scored) / intervals if intervals else None,
        # Keep the continuous ordering score on weak rows as well.  A reversed
        # control is expected to be weak by the direction gate, but dropping
        # its likelihood would make the negative control look unavailable.
        "score": float(np.median([row["likelihood"] for row in usable])) if usable else None,
        "reliable_score": float(np.median([row["likelihood"] for row in scored])) if scored else None,
        "median_residual_px": float(np.median([row["median_residual_px"] for row in usable])) if usable else None,
        "median_direction_cosine": float(np.median([row["direction_cosine"] for row in usable if row.get("direction_cosine") is not None])) if any(row.get("direction_cosine") is not None for row in usable) else None,
        "thresholds": {
            "min_fixed_support_fraction": float(min_fixed_support_fraction),
            "min_projection_valid_fraction": float(min_projection_valid_fraction),
            "likelihood_scale_px": float(likelihood_scale_px),
            "residual_inlier_threshold_px": float(residual_inlier_threshold_px),
            "min_direction_cosine": float(min_direction_cosine),
        },
        "missing_value_policy": "unavailable_never_zero_fill",
        "warning": "Projection validity is reported separately from fixed input support; this score is not a metric trajectory reconstruction.",
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
    aligned_cosines = [
        float(row["direction_cosine"])
        for row in scored
        if row.get("direction_cosine") is not None
    ]
    return {
        "protocol": "iac-twin-differential-forward-consistency-v1",
        "metric_id": "RCS",
        "metric": "Response Consistency Score",
        "legacy_metric": "CCFC-S",
        "metric_definition": "counterfactual_video_response_vs_action_intervention_response",
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
        "temporal_persistence": (
            float(np.mean(np.asarray(aligned_cosines) > 0.0))
            if aligned_cosines else None
        ),
        "median_support_fraction": float(np.median([row["support_fraction"] for row in scored])) if scored else None,
        "minimum_support_fraction": float(np.min([row["support_fraction"] for row in scored])) if scored else None,
        "rows": rows,
    }


STRUCTURAL_GROUNDING_DESCRIPTORS = (
    "median_flow_magnitude_px",
    "horizontal_flow_center",
    "vertical_flow_center",
    "divergence",
    "curl",
)


def score_structural_grounding(
    generated_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    descriptor_scales: dict[str, float],
    min_common_intervals: int = 3,
    min_descriptors_per_interval: int = 3,
) -> dict[str, Any]:
    """Compare generated flow structure with a same-source external future.

    This is the structural, metric-free Grounding Score candidate.  Scales
    must be frozen from a separate real-video calibration split; this function
    never estimates them from the compared rows.  Missing intervals and
    descriptors are omitted from their local evidence, but a branch is marked
    ``unavailable`` unless it has the configured minimum number of intervals.
    No zero-filled evidence is permitted.
    """
    if len(generated_rows) != len(reference_rows):
        raise ValueError("generated_rows and reference_rows must have equal length")
    if min_common_intervals < 1 or min_descriptors_per_interval < 1:
        raise ValueError("minimum evidence counts must be positive")
    scales = {str(k): float(v) for k, v in descriptor_scales.items()}
    for key in STRUCTURAL_GROUNDING_DESCRIPTORS:
        if key not in scales or not np.isfinite(scales[key]) or scales[key] <= 0:
            raise ValueError(f"missing positive calibration scale for {key}")

    def _transform(key: str, value: float) -> float:
        return float(np.log1p(max(value, 0.0))) if key == "median_flow_magnitude_px" else float(value)

    interval_scores: list[dict[str, Any]] = []
    for index, (generated, reference) in enumerate(zip(generated_rows, reference_rows)):
        if generated.get("interval_index", index) != reference.get("interval_index", index):
            raise ValueError("generated/reference interval axes do not match")
        row: dict[str, Any] = {"interval_index": int(generated.get("interval_index", index))}
        if not (generated.get("input_available", True) and reference.get("input_available", True)):
            row.update({"status": "unavailable", "reason": "missing_common_input"})
            interval_scores.append(row)
            continue
        components: dict[str, float] = {}
        for key in STRUCTURAL_GROUNDING_DESCRIPTORS:
            first, second = generated.get(key), reference.get(key)
            if first is None or second is None or not np.isfinite(first) or not np.isfinite(second):
                continue
            delta = abs(_transform(key, float(first)) - _transform(key, float(second)))
            components[key] = float(np.exp(-delta / scales[key]))
        if len(components) < min_descriptors_per_interval:
            row.update({"status": "unavailable", "reason": "insufficient_descriptors"})
        else:
            row.update({"status": "scored", "score": float(np.median(list(components.values()))), "components": components})
        interval_scores.append(row)
    scored = [row for row in interval_scores if row.get("status") == "scored"]
    branch_available = len(scored) >= int(min_common_intervals)
    branch_score = float(np.median([row["score"] for row in scored])) if branch_available else None
    return {
        "protocol": "iac-structural-grounding-score-v1",
        "metric_id": "GS",
        "metric": "Grounding Score",
        "legacy_metric": "FAU",
        "metric_definition": "candidate_blind_generated_flow_structure_vs_same_source_external_future",
        "status": "ok" if branch_available else "unavailable",
        "score": branch_score,
        "coverage": float(branch_available),
        "common_interval_count": len(scored),
        "interval_count": len(interval_scores),
        "minimum_common_intervals": int(min_common_intervals),
        "minimum_descriptors_per_interval": int(min_descriptors_per_interval),
        "descriptor_scales": scales,
        "intervals": interval_scores,
        "claim": "external_reality_grounding_not_future_to_action_causality",
        "missing_value_policy": "unavailable_never_zero_fill",
    }

"""History-conditioned Action Alignment Score (AS).

The evaluator receives a complete benchmark-v3 record, but keeps the visual
and action channels independent.  The image probe must be candidate-blind;
the action trajectory is only joined after visual motion has been extracted.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

import numpy as np

from .visual_output_layer import DEFAULT_CONFIG, score_longitudinal


def _image_context_quality(record: Mapping[str, Any]) -> dict[str, Any]:
    """Read the complete 4+4 clip for a lightweight continuity/quality gate."""
    history = list(record.get("history_frame_paths") or record.get("history_images") or [])
    future = list(record.get("future_frame_paths") or record.get("future_images") or [])
    paths = history + future
    try:
        import cv2  # type: ignore
    except ImportError:
        return {"status": "unavailable", "reason": "opencv_unavailable", "frame_count": len(paths)}
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(image is None for image in images):
        return {"status": "unavailable", "reason": "frame_missing", "frame_count": len(paths)}
    resized = [cv2.resize(image, (256, 144), interpolation=cv2.INTER_AREA) for image in images]
    blur = [float(cv2.Laplacian(image, cv2.CV_64F).var()) for image in resized]
    delta = [float(np.mean(np.abs(resized[index + 1].astype(np.float32) - resized[index].astype(np.float32))) / 255.0) for index in range(len(resized) - 1)]
    boundary = delta[len(history) - 1] if len(history) > 0 and len(delta) >= len(history) else None
    return {
        "status": "scored",
        "frame_count": len(paths),
        "history_frame_count": len(history),
        "future_frame_count": len(future),
        "median_laplacian_variance": float(np.median(blur)) if blur else None,
        "history_median_laplacian_variance": float(np.median(blur[:len(history)])) if history else None,
        "future_median_laplacian_variance": float(np.median(blur[len(history):])) if future else None,
        "median_adjacent_normalized_mae": float(np.median(delta)) if delta else None,
        "history_future_boundary_normalized_mae": boundary,
    }


def validate_benchmark_v3_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed 4-history/4-future/4-state contract."""
    history = list(record.get("history_frame_paths") or record.get("history_images") or [])
    future = list(record.get("future_frame_paths") or record.get("future_images") or [])
    trajectory = record.get("action_trajectory")
    issues: list[str] = []
    if len(history) != 4:
        issues.append("history_count_not_4")
    if len(future) != 4:
        issues.append("future_count_not_4")
    try:
        if trajectory and isinstance(trajectory[0], Mapping):
            action = np.asarray([item["pose"][:3] for item in trajectory], dtype=np.float64)
        else:
            action = np.asarray(trajectory, dtype=np.float64)
        if action.shape not in ((4, 3), (8, 3)):
            issues.append("action_trajectory_not_4x3")
    except (TypeError, ValueError):
        action = np.empty((0, 3), dtype=np.float64)
        issues.append("action_trajectory_invalid")
    history_times = list(record.get("history_times_s") or [])
    future_times = list(record.get("future_times_s") or [])
    if history_times != [-1.5, -1.0, -0.5, 0.0]:
        issues.append("history_times_not_fixed")
    if future_times != [1.0, 2.0, 3.0, 4.0] and future_times != [1, 2, 3, 4]:
        issues.append("future_times_not_fixed")
    return {
        "valid": not issues,
        "issues": issues,
        "history_count": len(history),
        "future_count": len(future),
        "trajectory_shape": list(action.shape),
        "history_context_available": len(history) == 4,
        "future_sequence_available": len(future) == 4,
        "action_state_available": action.shape in ((4, 3), (8, 3)),
        "trajectory_adapter": "half_second_to_1hz" if action.shape == (8, 3) else "identity",
    }


def _trajectory(record: Mapping[str, Any]) -> np.ndarray:
    raw = record.get("action_trajectory")
    if raw and isinstance(raw[0], Mapping):
        trajectory = np.asarray([item["pose"][:3] for item in raw], dtype=np.float64)
    else:
        trajectory = np.asarray(raw, dtype=np.float64)
    if trajectory.shape == (8, 3):
        return trajectory[1::2]
    return trajectory


def _interval_visual_yaw(interval: Mapping[str, Any], record: Mapping[str, Any]) -> float | None:
    transform = np.asarray(record.get("camera_to_ego", np.eye(4)), dtype=np.float64)
    if transform.shape != (4, 4):
        return None
    basis = transform[:3, :3]
    rotation = np.asarray(interval.get("reloc3r_rotation"), dtype=np.float64)
    if rotation.shape != (3, 3):
        return None
    ego_rotation = basis @ rotation @ basis.T
    return float(math.atan2(ego_rotation[1, 0], ego_rotation[0, 0]))


def _visual_yaw(row: Mapping[str, Any], record: Mapping[str, Any]) -> float | None:
    yaws = [
        value
        for interval in row.get("intervals", [])
        if (value := _interval_visual_yaw(interval, record)) is not None
    ]
    return float(np.sum(yaws)) if yaws else None


def score_row(row: Mapping[str, Any], record: Mapping[str, Any], *, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score one complete benchmark-v3 record after independent probing."""
    contract = validate_benchmark_v3_record(record)
    action = _trajectory(record)
    result: dict[str, Any] = {
        "sample_id": record.get("sample_id"),
        "model_id": record.get("wam_model_id"),
        "future_images_source": record.get("future_images_source"),
        "action_trajectory_source": record.get("action_trajectory_source"),
        "contract": contract,
        "history_context": {
            "used_as_visual_context": True,
            "history_count": contract["history_count"],
            "future_count": contract["future_count"],
            "candidate_blind_visual_branch": bool(record.get("metadata", {}).get("candidate_blind_image_branch", record.get("candidate_bank_used_by_decoder") is False)),
            "quality": _image_context_quality(record),
        },
    }
    if not contract["valid"] or action.shape != (4, 3):
        result.update({"status": "uncertain", "reason": "invalid_benchmark_v3_contract"})
        return result
    probe_intervals = list(row.get("intervals", []))
    visual_probe_valid = len(probe_intervals) == 4
    # Missing probe intervals are explicit abstentions, not absent benchmark
    # rows.  Extra intervals are also rejected instead of silently changing
    # the frozen four-interval horizon.
    scoring_intervals = probe_intervals if visual_probe_valid else [{} for _ in range(4)]
    # Trajectory states are poses relative to the current t=0 ego frame.
    # Prepending the origin yields the four observable benchmark-v3 intervals:
    # 0->1, 1->2, 2->3, and 3->4 seconds.
    action_delta = np.linalg.norm(
        np.diff(np.vstack([np.zeros((1, 2), dtype=np.float64), action[:, :2]]), axis=0),
        axis=1,
    )
    predicted: list[float] = []
    interval_scores: list[dict[str, Any]] = []
    for index, interval in enumerate(scoring_intervals):
        visual_interval_yaw = _interval_visual_yaw(interval, record)
        motion = interval.get("estimators", {}).get("all", {}).get("motion", {})
        match = interval.get("estimators", {}).get("all", {}).get("match", {})
        value = motion.get("longitudinal_m") if motion.get("available") else None
        if value is None or not np.isfinite(float(value)) or index >= len(action_delta):
            interval_scores.append({
                "status": "uncertain",
                "acceptable": False,
                "reason": "visual_motion_unavailable",
                "visual_yaw_rad": visual_interval_yaw,
            })
            continue
        predicted.append(float(value))
        interval_scores.append(
            {
                **score_longitudinal(
                float(value),
                float(action_delta[index]),
                geometry={
                    "matches": match.get("selected_matches"),
                    "inlier_fraction": motion.get("inlier_fraction"),
                    "median_reprojection_error_px": motion.get("median_reprojection_error_px"),
                },
                config=config,
                ),
                "visual_yaw_rad": visual_interval_yaw,
            }
        )
    visual_yaw = _visual_yaw({"intervals": scoring_intervals}, record) if visual_probe_valid else None
    boundary_yaw = (
        _interval_visual_yaw(scoring_intervals[0], record)
        if visual_probe_valid
        else None
    )
    action_yaw = float(action[-1, 2])
    yaw_threshold = float(
        (config or {}).get("yaw", {}).get(
            "straight_threshold_rad", DEFAULT_CONFIG["yaw"]["straight_threshold_rad"]
        )
    )
    yaw_applicable = abs(action_yaw) >= yaw_threshold
    yaw_match = None if not yaw_applicable or visual_yaw is None else bool(np.sign(visual_yaw) == np.sign(action_yaw))
    # A quality-eligible interval stays in the AS denominator whether it is
    # inside or outside the tolerance.  Only genuine geometry abstentions are
    # excluded; otherwise every conditional progress score would be 1.0.
    usable_scores = [item for item in interval_scores if item.get("status") != "uncertain"]
    progress_acceptance = (
        float(np.mean([item["acceptable"] for item in usable_scores])) if usable_scores else None
    )
    components: list[float] = []
    if progress_acceptance is not None:
        components.append(progress_acceptance)
    if yaw_match is not None:
        components.append(float(yaw_match))
    result.update(
        {
            "status": "scored" if components else "uncertain",
            "visual_probe_contract": {
                "valid": visual_probe_valid,
                "expected_interval_count": 4,
                "observed_interval_count": len(probe_intervals),
            },
            "intervals": interval_scores,
            "visual_yaw_intervals_rad": [item.get("visual_yaw_rad") for item in interval_scores],
            "progress": {
                "interval_count": len(interval_scores),
                "visual_available_count": len(predicted),
                "coverage": len(predicted) / len(interval_scores) if interval_scores else 0.0,
                "quality_scored_count": len(usable_scores),
                "quality_scored_coverage": len(usable_scores) / len(interval_scores) if interval_scores else 0.0,
                "accepted_rate": progress_acceptance,
                "predicted_longitudinal_m": predicted,
                "action_interval_distance_m": action_delta.tolist(),
            },
            "yaw": {
                "visual_yaw_rad": visual_yaw,
                "action_yaw_rad": action_yaw,
                "applicable": yaw_applicable,
                "direction_match": yaw_match,
            },
            "history_future_boundary": {
                "visual_yaw_rad": boundary_yaw,
                "absolute_yaw_deg": abs(math.degrees(boundary_yaw)) if boundary_yaw is not None else None,
                "geometry_eligible": bool(interval_scores and interval_scores[0].get("status") != "uncertain"),
                "normalized_mae": result["history_context"]["quality"].get("history_future_boundary_normalized_mae"),
            },
            "as_components": {
                "progress_acceptance": progress_acceptance,
                "yaw_direction_match": yaw_match,
                "composite_mean": float(np.mean(components)) if components else None,
                "component_count": len(components),
            },
        }
    )
    return result


def aggregate(
    rows: list[Mapping[str, Any]],
    *,
    mode: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scored = [row for row in rows if row.get("status") == "scored"]
    progress = [
        float(row["as_components"]["progress_acceptance"])
        for row in scored
        if row["as_components"].get("progress_acceptance") is not None
    ]
    yaw_applicable = [row for row in rows if row.get("yaw", {}).get("applicable")]
    yaw = [
        row["yaw"]["direction_match"]
        for row in yaw_applicable
        if row["yaw"].get("direction_match") is not None
    ]
    composite = [float(row["as_components"]["composite_mean"]) for row in scored]
    intervals = [interval for row in rows for interval in row.get("intervals", [])]
    # AS is defined on the fixed benchmark-v3 contract: every declared row
    # contributes four observable longitudinal intervals.  Missing or weak
    # geometry therefore receives no credit in the headline score, while the
    # conditional score below remains available for diagnosing the reader.
    expected_interval_count = 4 * len(rows)
    accepted_intervals = [interval for interval in intervals if interval.get("acceptable") is True]
    visual_available = [
        interval for interval in intervals if interval.get("reason") != "visual_motion_unavailable"
    ]
    quality_scored = [interval for interval in intervals if interval.get("status") != "uncertain"]
    progress_effective = (
        len(accepted_intervals) / expected_interval_count if expected_interval_count else None
    )
    progress_conditional_micro = (
        len(accepted_intervals) / len(quality_scored) if quality_scored else None
    )
    progress_score_coverage = (
        len(quality_scored) / expected_interval_count if expected_interval_count else None
    )
    yaw_correct_count = sum(value is True for value in yaw)
    yaw_conditional = yaw_correct_count / len(yaw) if yaw else None
    yaw_score_coverage = len(yaw) / len(yaw_applicable) if yaw_applicable else None
    yaw_effective = (
        yaw_correct_count / len(yaw_applicable) if yaw_applicable else None
    )
    as_conditional = (
        math.sqrt(progress_conditional_micro * yaw_conditional)
        if progress_conditional_micro is not None and yaw_conditional is not None
        else None
    )
    as_observability = (
        math.sqrt(progress_score_coverage * yaw_score_coverage)
        if progress_score_coverage is not None and yaw_score_coverage is not None
        else None
    )
    as_overall = (
        math.sqrt(progress_effective * yaw_effective)
        if progress_effective is not None and yaw_effective is not None
        else None
    )
    progress_abstentions = Counter(
        str(interval.get("reason") or "unspecified")
        for interval in intervals
        if interval.get("status") == "uncertain"
    )
    unaccounted_progress = max(
        expected_interval_count - len(quality_scored) - sum(progress_abstentions.values()),
        0,
    )
    if unaccounted_progress:
        progress_abstentions["missing_or_invalid_record"] += unaccounted_progress
    boundary = [row.get("history_future_boundary", {}) for row in rows]
    boundary_yaw_deg = [
        float(item["absolute_yaw_deg"])
        for item in boundary
        if item.get("absolute_yaw_deg") is not None
    ]
    boundary_geometry = [item for item in boundary if item.get("geometry_eligible")]
    boundary_geometry_coverage = len(boundary_geometry) / len(rows) if rows else 0.0
    audit_config = (config or {}).get("input_lineage_audit", {})
    large_yaw_threshold = float(audit_config.get("large_boundary_yaw_threshold_deg", 30.0))
    maximum_large_rate = float(audit_config.get("maximum_large_boundary_yaw_rate", 0.10))
    minimum_geometry_coverage = float(
        audit_config.get("minimum_boundary_geometry_coverage_for_pass", 0.20)
    )
    large_boundary_rate = (
        float(np.mean([value > large_yaw_threshold for value in boundary_yaw_deg]))
        if boundary_yaw_deg
        else None
    )
    if large_boundary_rate is not None and large_boundary_rate > maximum_large_rate:
        lineage_status = "failed"
    elif boundary_geometry_coverage < minimum_geometry_coverage:
        lineage_status = "warning"
    else:
        lineage_status = "passed"
    return {
        "mode": mode,
        "row_count": len(rows),
        "scored_row_count": len(scored),
        "row_coverage": len(scored) / len(rows) if rows else 0.0,
        "progress_row_count": len(progress),
        "progress_row_coverage": len(progress) / len(rows) if rows else 0.0,
        "interval_count": len(intervals),
        "expected_interval_count": expected_interval_count,
        "accepted_interval_count": len(accepted_intervals),
        "visual_available_interval_count": len(visual_available),
        "visual_interval_coverage": len(visual_available) / len(intervals) if intervals else 0.0,
        "quality_scored_interval_count": len(quality_scored),
        "quality_scored_interval_coverage": len(quality_scored) / len(intervals) if intervals else 0.0,
        "AS_progress_mean": float(np.mean(progress)) if progress else None,
        "AS_progress_conditional_micro": progress_conditional_micro,
        "AS_progress_score_coverage": progress_score_coverage,
        "AS_progress_effective": progress_effective,
        "AS_yaw_direction": yaw_conditional,
        "AS_yaw_conditional": yaw_conditional,
        "AS_yaw_score_coverage": yaw_score_coverage,
        "AS_yaw_effective": yaw_effective,
        "AS_composite_mean": float(np.mean(composite)) if composite else None,
        "AS_conditional": as_conditional,
        "AS_conditional_100": 100.0 * as_conditional if as_conditional is not None else None,
        "AS_observability": as_observability,
        "AS_observability_100": 100.0 * as_observability if as_observability is not None else None,
        "AS_overall": as_overall,
        "AS_overall_100": 100.0 * as_overall if as_overall is not None else None,
        "AS_decomposition": {
            "identity": "AS_overall = AS_conditional * AS_observability",
            "conditional_consistency": as_conditional,
            "observability_summary": as_observability,
            "coverage_aware_deployment_summary": as_overall,
            "note": "progress and yaw use different declared units; retain both channel coverages",
        },
        "status_counts": {
            "progress": {
                "declared": expected_interval_count,
                "scored": len(quality_scored),
                "abstain": max(expected_interval_count - len(quality_scored), 0),
            },
            "yaw": {
                "declared_applicable": len(yaw_applicable),
                "scored": len(yaw),
                "abstain": max(len(yaw_applicable) - len(yaw), 0),
            },
        },
        "abstention_reasons": {
            "progress": dict(sorted(progress_abstentions.items())),
            "yaw": {
                "visual_yaw_unavailable": max(len(yaw_applicable) - len(yaw), 0),
            },
        },
        "yaw_applicable_count": len(yaw_applicable),
        "yaw_turn_count": len(yaw),
        "yaw_correct_count": yaw_correct_count,
        "yaw_coverage": len(yaw) / len(yaw_applicable) if yaw_applicable else None,
        "input_lineage_audit": {
            "status": lineage_status,
            "formal_metric_eligible": lineage_status == "passed",
            "history_boundary_yaw_count": len(boundary_yaw_deg),
            "median_absolute_boundary_yaw_deg": float(np.median(boundary_yaw_deg)) if boundary_yaw_deg else None,
            "large_boundary_yaw_threshold_deg": large_yaw_threshold,
            "large_boundary_yaw_rate": large_boundary_rate,
            "maximum_allowed_large_boundary_yaw_rate": maximum_large_rate,
            "minimum_boundary_geometry_coverage_for_pass": minimum_geometry_coverage,
            "boundary_geometry_coverage": boundary_geometry_coverage,
        },
    }

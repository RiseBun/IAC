"""Lightweight road-structure evidence extracted from image-side masks/flow.

The functions here deliberately avoid metric reconstruction.  They expose the
parts of the scene that a human driver uses for a coarse trajectory support:
near-road position, far-road direction/curvature, a global focus of expansion,
and static points that remain trackable over time.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _mask_array(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("road mask must have shape [H,W]")
    return value


def split_near_far(mask: np.ndarray, *, near_start: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
    """Split a road mask by image rows; lower rows are the near road."""
    value = _mask_array(mask)
    if not 0.0 < near_start < 1.0:
        raise ValueError("near_start must be in (0,1)")
    cut = int(round(value.shape[0] * near_start))
    near = np.zeros_like(value)
    far = np.zeros_like(value)
    near[cut:] = value[cut:]
    far[:cut] = value[:cut]
    return near, far


def extract_road_boundaries(
    mask: np.ndarray,
    *,
    row_step: int = 4,
    left_quantile: float = 0.05,
    right_quantile: float = 0.95,
    polynomial_degree: int = 2,
) -> dict[str, Any]:
    """Fit smooth left/right image boundaries from row-wise road support."""
    value = _mask_array(mask)
    if row_step < 1 or not 0.0 <= left_quantile < right_quantile <= 1.0:
        raise ValueError("invalid boundary extraction parameters")
    rows: list[float] = []
    left: list[float] = []
    right: list[float] = []
    counts: list[float] = []
    for y in range(0, value.shape[0], int(row_step)):
        xs = np.flatnonzero(value[y])
        if len(xs) < 2:
            continue
        rows.append(float(y))
        left.append(float(np.quantile(xs, left_quantile)))
        right.append(float(np.quantile(xs, right_quantile)))
        counts.append(float(len(xs)))
    if len(rows) < 2:
        return {
            "valid": False,
            "rows": [],
            "left_x": [],
            "right_x": [],
            "left_coeff": [],
            "right_coeff": [],
            "confidence": 0.0,
        }
    degree = min(int(polynomial_degree), len(rows) - 1)
    y_arr = np.asarray(rows, dtype=np.float64)
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    # Normalize row coordinates before fitting to avoid poor conditioning on
    # full-resolution images; coefficients are in the normalized coordinate.
    y0 = max(float(value.shape[0] - 1), 1.0)
    yn = y_arr / y0
    left_coeff = np.polyfit(yn, left_arr, degree, w=np.sqrt(np.asarray(counts)))
    right_coeff = np.polyfit(yn, right_arr, degree, w=np.sqrt(np.asarray(counts)))
    left_fit = np.polyval(left_coeff, yn)
    right_fit = np.polyval(right_coeff, yn)
    residual = np.concatenate([np.abs(left_fit - left_arr), np.abs(right_fit - right_arr)])
    width = np.maximum(right_arr - left_arr, 1.0)
    width = np.concatenate([width, width])
    confidence = float(np.clip(np.mean(np.exp(-residual / width)) * min(1.0, len(rows) / 20.0), 0.0, 1.0))
    return {
        "valid": True,
        "rows": rows,
        "left_x": left,
        "right_x": right,
        "left_coeff": left_coeff.tolist(),
        "right_coeff": right_coeff.tolist(),
        "confidence": confidence,
        "image_height": int(value.shape[0]),
    }


def _boundary_descriptor(boundaries: dict[str, Any], image_width: int) -> dict[str, float | None]:
    if not boundaries.get("valid"):
        return {"center_offset_norm": None, "heading_rad": None, "curvature_norm": None}
    rows = np.asarray(boundaries["rows"], dtype=np.float64)
    if len(rows) < 2:
        return {"center_offset_norm": None, "heading_rad": None, "curvature_norm": None}
    yn = rows / max(float(boundaries.get("image_height", rows.max() + 1) - 1), 1.0)
    left = np.polyval(np.asarray(boundaries["left_coeff"], dtype=np.float64), yn)
    right = np.polyval(np.asarray(boundaries["right_coeff"], dtype=np.float64), yn)
    center = 0.5 * (left + right)
    far = int(np.argmin(rows))
    near = int(np.argmax(rows))
    center_offset = (float(center[near]) - 0.5 * float(image_width)) / max(float(image_width), 1.0)
    coeff = np.polyfit(yn, center, min(2, len(center) - 1))
    image_height = max(float(boundaries.get("image_height", rows.max() + 1) - 1), 1.0)
    slope = (float(np.polyval(np.polyder(coeff), yn[far])) / image_height) if len(coeff) > 1 else 0.0
    second = (float(np.polyval(np.polyder(coeff, 2), yn[far])) / (image_height * image_height)) if len(coeff) > 2 else 0.0
    return {
        "center_offset_norm": center_offset,
        "heading_rad": float(np.arctan(slope)),
        "curvature_norm": second,
    }


def estimate_focus_of_expansion(
    flows: np.ndarray,
    static_weights: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    *,
    intrinsics: np.ndarray | None = None,
    max_points: int = 2500,
    min_flow_px: float = 0.5,
    robust_iterations: int = 5,
) -> dict[str, Any]:
    """Estimate a robust FOE from flow-line intersection equations."""
    flow = np.asarray(flows, dtype=np.float64)
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("flows must have shape [T,H,W,2]")
    weights = np.ones(flow.shape[:-1], dtype=np.float64) if static_weights is None else np.asarray(static_weights, dtype=np.float64)
    if weights.shape != flow.shape[:-1]:
        raise ValueError("static_weights must match flows")
    roi = np.ones(flow.shape[1:3], dtype=bool) if roi_mask is None else _mask_array(roi_mask)
    if roi.shape != flow.shape[1:3]:
        raise ValueError("roi_mask must match flow image size")
    yy, xx = np.indices(flow.shape[1:3], dtype=np.float64)
    camera = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
    if camera is not None and (camera.shape != (3, 3) or not np.all(np.isfinite(camera))):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    fx = 1.0 if camera is None else float(camera[0, 0])
    fy = 1.0 if camera is None else float(camera[1, 1])
    cx = 0.0 if camera is None else float(camera[0, 2])
    cy = 0.0 if camera is None else float(camera[1, 2])
    points: list[np.ndarray] = []
    rhs: list[float] = []
    coeff_weights: list[float] = []
    for interval in range(flow.shape[0]):
        u = flow[interval, ..., 0]
        v = flow[interval, ..., 1]
        magnitude = np.hypot(u, v)
        valid = roi & np.isfinite(u) & np.isfinite(v) & (magnitude >= float(min_flow_px)) & (weights[interval] > 0.0)
        indices, _, _ = _spatial_flow_indices(
            valid,
            weights[interval],
            max_points=max_points,
            grid_rows=6,
            grid_cols=10,
        )
        for flat in indices:
            y, x = np.unravel_index(int(flat), roi.shape)
            point = np.asarray([(x - cx) / fx, (y - cy) / fy], dtype=np.float64)
            flow_normalized = np.asarray([u[y, x] / fx, v[y, x] / fy], dtype=np.float64)
            normal = np.asarray([-flow_normalized[1], flow_normalized[0]], dtype=np.float64)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-8:
                continue
            normal /= norm
            points.append(normal)
            rhs.append(float(normal @ point))
            coeff_weights.append(float(np.clip(weights[interval, y, x], 0.05, 1.0)))
    if len(points) < 8:
        return {"valid": False, "foe_xy": None, "confidence": 0.0, "num_lines": len(points)}
    matrix = np.asarray(points)
    vector = np.asarray(rhs)
    sqrt_w = np.sqrt(np.asarray(coeff_weights))
    lhs = matrix * sqrt_w[:, None]
    target = vector * sqrt_w
    solution, _, _, singular = np.linalg.lstsq(lhs, target, rcond=None)
    robust_weights = np.asarray(coeff_weights, dtype=np.float64)
    for _ in range(max(int(robust_iterations), 0)):
        signed_residual = matrix @ solution - vector
        median = float(np.median(signed_residual))
        mad = float(np.median(np.abs(signed_residual - median)))
        scale = max(1.4826 * mad, 1e-6)
        huber = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(signed_residual), 1e-12))
        sqrt_w = np.sqrt(robust_weights * huber)
        lhs = matrix * sqrt_w[:, None]
        target = vector * sqrt_w
        solution, _, _, singular = np.linalg.lstsq(lhs, target, rcond=None)
    residual = np.abs(matrix @ solution - vector)
    scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1e-6)
    inlier_fraction = float(np.mean(residual <= 2.5 * scale))
    conditioning = float(singular[-1] / max(singular[0], 1e-8)) if len(singular) else 0.0
    confidence = float(np.clip(
        inlier_fraction * min(1.0, len(points) / 300.0) * min(1.0, conditioning / 0.05),
        0.0,
        1.0,
    ))
    foe_pixel = [float(solution[0] * fx + cx), float(solution[1] * fy + cy)]
    median_residual = float(np.median(residual))
    return {
        "valid": bool(np.all(np.isfinite(solution))),
        "foe_xy": foe_pixel if np.all(np.isfinite(solution)) else None,
        "foe_normalized_xy": solution.tolist() if np.all(np.isfinite(solution)) else None,
        "confidence": confidence,
        "median_line_residual_normalized": median_residual,
        "median_line_residual_px": (
            median_residual if camera is None else median_residual * float(np.sqrt(fx * fy))
        ),
        "inlier_fraction": inlier_fraction,
        "num_lines": int(len(points)),
        "conditioning": conditioning,
    }


def _robust_affine_flow(
    points: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    base_weights = np.asarray(weights, dtype=np.float64)
    active_weights = base_weights.copy()
    coefficients = np.zeros((3, 2), dtype=np.float64)
    residual = np.zeros(len(points), dtype=np.float64)
    for _ in range(max(int(iterations), 0) + 1):
        sqrt_w = np.sqrt(np.maximum(active_weights, 0.0))
        coefficients = np.linalg.lstsq(
            design * sqrt_w[:, None], values * sqrt_w[:, None], rcond=None
        )[0]
        residual = np.linalg.norm(design @ coefficients - values, axis=1)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        scale = max(1.4826 * mad, 1e-6)
        huber = np.minimum(1.0, 1.5 * scale / np.maximum(residual, 1e-12))
        active_weights = base_weights * huber
    return coefficients, residual, active_weights


def _spatial_flow_indices(
    valid: np.ndarray,
    weights: np.ndarray,
    *,
    max_points: int,
    grid_rows: int,
    grid_cols: int,
) -> tuple[np.ndarray, int, int]:
    height, width = valid.shape
    yy, xx = np.indices(valid.shape)
    flat = np.flatnonzero(valid)
    if not len(flat):
        return flat, 0, 0
    cell = np.minimum(yy.reshape(-1)[flat] * grid_rows // max(height, 1), grid_rows - 1)
    cell = cell * grid_cols + np.minimum(xx.reshape(-1)[flat] * grid_cols // max(width, 1), grid_cols - 1)
    eligible_cells = int(len(np.unique(cell)))
    quota, remainder = divmod(int(max_points), max(grid_rows * grid_cols, 1))
    selected: list[np.ndarray] = []
    flat_weights = weights.reshape(-1)
    for cell_index in range(grid_rows * grid_cols):
        local = np.flatnonzero(cell == cell_index)
        if not len(local):
            continue
        candidates = flat[local]
        order = np.lexsort((candidates, -flat_weights[candidates]))
        count = quota + (1 if cell_index < remainder else 0)
        selected.append(candidates[order[:count]])
    chosen = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    occupied = int(len(np.unique(
        np.minimum(yy.reshape(-1)[chosen] * grid_rows // max(height, 1), grid_rows - 1)
        * grid_cols
        + np.minimum(xx.reshape(-1)[chosen] * grid_cols // max(width, 1), grid_cols - 1)
    ))) if len(chosen) else 0
    return chosen, occupied, eligible_cells


def flow_structure_profile(
    flows: np.ndarray,
    static_weights: np.ndarray,
    roi_mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    min_flow_px: float = 0.5,
    min_points: int = 100,
    min_spatial_cells: int = 6,
    max_points: int = 3000,
    grid_rows: int = 3,
    grid_cols: int = 6,
) -> dict[str, Any]:
    """Extract candidate-blind ordinal motion structure without metric fitting."""
    flow = np.asarray(flows, dtype=np.float64)
    weights = np.asarray(static_weights, dtype=np.float64)
    roi = _mask_array(roi_mask)
    camera = np.asarray(intrinsics, dtype=np.float64)
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("flows must have shape [T,H,W,2]")
    if weights.shape != flow.shape[:-1] or roi.shape != flow.shape[1:3]:
        raise ValueError("weights and roi_mask must match flows")
    if camera.shape != (3, 3) or not np.all(np.isfinite(camera)):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    fx, fy = float(camera[0, 0]), float(camera[1, 1])
    cx, cy = float(camera[0, 2]), float(camera[1, 2])
    yy, xx = np.indices(roi.shape, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for interval in range(len(flow)):
        magnitude = np.linalg.norm(flow[interval], axis=-1)
        valid = (
            roi
            & np.isfinite(flow[interval]).all(axis=-1)
            & np.isfinite(weights[interval])
            & (weights[interval] > 0.0)
            & (magnitude >= float(min_flow_px))
        )
        indices, occupied_cells, eligible_cells = _spatial_flow_indices(
            valid,
            weights[interval],
            max_points=max_points,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        available = len(indices) >= int(min_points) and occupied_cells >= int(min_spatial_cells)
        row: dict[str, Any] = {
            "interval_index": interval,
            "input_available": bool(available),
            "source_points": int(valid.sum()),
            "sampled_points": int(len(indices)),
            "occupied_spatial_cells": occupied_cells,
            "eligible_spatial_cells": eligible_cells,
            "spatial_coverage": float(occupied_cells / max(grid_rows * grid_cols, 1)),
            "roi_support_fraction": float(valid.sum() / max(roi.sum(), 1)),
            "median_flow_magnitude_px": float(np.median(magnitude[valid])) if valid.any() else None,
        }
        if not available:
            row.update({
                "foe": {"valid": False, "foe_xy": None, "confidence": 0.0, "num_lines": int(len(indices))},
                "affine_valid": False,
                "structure_confidence": 0.0,
            })
            rows.append(row)
            continue
        sy, sx = np.unravel_index(indices, roi.shape)
        points = np.column_stack([(sx - cx) / fx, (sy - cy) / fy])
        values = np.column_stack([
            flow[interval, sy, sx, 0] / fx,
            flow[interval, sy, sx, 1] / fy,
        ])
        sample_weights = weights[interval, sy, sx]
        coefficients, residual, robust_weights = _robust_affine_flow(
            points, values, sample_weights
        )
        predicted = np.column_stack([points, np.ones(len(points))]) @ coefficients
        value_norm = np.linalg.norm(values, axis=1)
        relative_residual = float(np.median(residual) / max(float(np.median(value_norm)), 1e-8))
        affine_explained = float(np.clip(1.0 - relative_residual, 0.0, 1.0))
        left = points[:, 0] <= 0.0
        right = points[:, 0] > 0.0
        horizontal_left = float(np.median(values[left, 0])) if left.any() else None
        horizontal_right = float(np.median(values[right, 0])) if right.any() else None
        divergence = float(coefficients[0, 0] + coefficients[1, 1])
        curl = float(coefficients[0, 1] - coefficients[1, 0])
        foe = estimate_focus_of_expansion(
            flow[interval:interval + 1],
            weights[interval:interval + 1],
            roi,
            intrinsics=camera,
            max_points=max_points,
            min_flow_px=min_flow_px,
        )
        robust_fraction = float(np.sum(robust_weights) / max(np.sum(sample_weights), 1e-8))
        structure_confidence = float(np.clip(
            row["spatial_coverage"] * robust_fraction * (0.5 + 0.5 * affine_explained),
            0.0,
            1.0,
        ))
        row.update({
            "foe": foe,
            "affine_valid": bool(np.all(np.isfinite(coefficients))),
            "affine_coefficients": coefficients.tolist(),
            "affine_median_relative_residual": relative_residual,
            "affine_explained_fraction": affine_explained,
            "affine_robust_weight_fraction": robust_fraction,
            "divergence": divergence,
            "curl": curl,
            "horizontal_flow_center": float(np.median(values[:, 0])),
            "vertical_flow_center": float(np.median(values[:, 1])),
            "left_horizontal_flow": horizontal_left,
            "right_horizontal_flow": horizontal_right,
            "left_right_horizontal_contrast": (
                None if horizontal_left is None or horizontal_right is None
                else float(horizontal_left - horizontal_right)
            ),
            "structure_confidence": structure_confidence,
        })
        rows.append(row)
    available_count = sum(bool(row["input_available"]) for row in rows)
    status = "usable" if available_count == len(rows) and rows else "partial" if available_count else "abstain"
    return {
        "protocol": "candidate-blind-flow-structure-v1",
        "metric_reconstruction_used": False,
        "candidate_bank_used": False,
        "measurement_available": bool(rows) and available_count == len(rows),
        "available_interval_fraction": float(available_count / max(len(rows), 1)),
        "status": status,
        "rows": rows,
        "parameters": {
            "min_flow_px": float(min_flow_px),
            "min_points": int(min_points),
            "min_spatial_cells": int(min_spatial_cells),
            "max_points": int(max_points),
            "grid_rows": int(grid_rows),
            "grid_cols": int(grid_cols),
        },
    }


def causal_boundary_keypoint_filter(
    masks: np.ndarray,
    *,
    observed_flows: np.ndarray | None = None,
    row_step: int = 4,
    max_jump_px: float = 28.0,
    huber_scale_px: float = 8.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter sparse road-boundary keypoints with a causal robust update.

    The current frame is never averaged with future frames.  A boundary point
    is accepted normally when its displacement from the previous observation is
    plausible; large jumps are Huber-shrunk toward the previous point.  This
    captures the temporal-prior idea of recent sparse lane work without
    pretending that unwarped image coordinates are stationary.
    """
    values = np.asarray(masks, dtype=bool)
    if values.ndim != 3 or not len(values):
        raise ValueError("masks must have shape [T,H,W]")
    outputs: list[dict[str, Any]] = []
    previous: dict[str, np.ndarray] | None = None
    clipped_count = 0
    total_count = 0
    for mask in values:
        current = extract_road_boundaries(mask, row_step=row_step, polynomial_degree=2)
        if not current.get("valid"):
            outputs.append(current)
            continue
        rows = np.asarray(current["rows"], dtype=np.float64)
        filtered: dict[str, Any] = dict(current)
        for side in ("left_x", "right_x"):
            point = np.asarray(current[side], dtype=np.float64)
            if previous is not None and previous["rows"].size:
                prior_rows = previous["rows"]
                prior_x = previous[side]
                if observed_flows is not None and len(observed_flows) >= len(outputs):
                    flow = np.asarray(observed_flows[len(outputs) - 1], dtype=np.float64)
                    height, width = flow.shape[:2]
                    px = np.rint(prior_x).astype(np.int64).clip(0, width - 1)
                    py = np.rint(prior_rows).astype(np.int64).clip(0, height - 1)
                    valid_flow = np.isfinite(flow[py, px]).all(axis=-1)
                    warped_x = prior_x + np.where(valid_flow, flow[py, px, 0], 0.0)
                    warped_rows = prior_rows + np.where(valid_flow, flow[py, px, 1], 0.0)
                    order = np.argsort(warped_rows)
                    prior_rows, prior_x = warped_rows[order], warped_x[order]
                prior = np.interp(rows, prior_rows, prior_x, left=np.nan, right=np.nan)
                valid = np.isfinite(point) & np.isfinite(prior)
                delta = point - prior
                total_count += int(valid.sum())
                large = valid & (np.abs(delta) > float(max_jump_px))
                clipped_count += int(large.sum())
                shrink = np.ones_like(delta)
                shrink[valid] = np.minimum(1.0, float(huber_scale_px) / np.maximum(np.abs(delta[valid]), 1e-6))
                point = np.where(valid, prior + shrink * delta, point)
            filtered[side] = point.tolist()
        yn = rows / max(float(mask.shape[0] - 1), 1.0)
        filtered["left_coeff"] = np.polyfit(yn, np.asarray(filtered["left_x"]), min(2, len(rows) - 1)).tolist()
        filtered["right_coeff"] = np.polyfit(yn, np.asarray(filtered["right_x"]), min(2, len(rows) - 1)).tolist()
        previous = {"rows": rows, "left_x": np.asarray(filtered["left_x"]), "right_x": np.asarray(filtered["right_x"])}
        outputs.append(filtered)
    return outputs, {"protocol": "causal-boundary-keypoint-filter", "clipped_fraction": float(clipped_count / max(total_count, 1)), "num_frames": len(values)}


def boundary_pixels_to_ego(
    boundaries: dict[str, Any],
    intrinsics: np.ndarray,
    camera_to_ego: np.ndarray,
    *,
    ground_z: float = 0.0,
    min_forward_m: float = 0.5,
    max_forward_m: float = 150.0,
) -> dict[str, Any]:
    """Project sparse image boundaries onto a ground plane in ego coordinates."""
    if not boundaries.get("valid"):
        return {"valid": False, "left_xy": [], "right_xy": [], "rows": []}
    rows = np.asarray(boundaries.get("rows", []), dtype=np.float64)
    height = max(float(boundaries.get("image_height", rows.max() + 1 if len(rows) else 1.0)), 1.0)
    K = np.asarray(intrinsics, dtype=np.float64)
    T = np.asarray(camera_to_ego, dtype=np.float64)
    if K.shape != (3, 3) or T.shape != (4, 4) or len(rows) < 2:
        return {"valid": False, "left_xy": [], "right_xy": [], "rows": []}
    K_inv = np.linalg.inv(K)
    origin = T[:3, 3]
    rotation = T[:3, :3]
    result: dict[str, Any] = {"valid": True, "rows": rows.tolist(), "image_height": int(round(height))}
    for side in ("left", "right"):
        xs = np.asarray(boundaries.get(f"{side}_x", []), dtype=np.float64)
        pixels = np.stack([xs, rows, np.ones_like(rows)], axis=1)
        rays_camera = (K_inv @ pixels.T).T
        rays_ego = (rotation @ rays_camera.T).T
        denom = rays_ego[:, 2]
        scale = (float(ground_z) - float(origin[2])) / np.where(np.abs(denom) > 1e-8, denom, np.nan)
        points = origin[None, :] + scale[:, None] * rays_ego
        valid = np.isfinite(points).all(axis=1) & np.isfinite(scale) & (scale > 0.0)
        valid &= points[:, 0] >= float(min_forward_m)
        valid &= points[:, 0] <= float(max_forward_m)
        result[f"{side}_xy"] = points[valid, :2].tolist()
        result[f"{side}_valid"] = valid.tolist()
    result["valid"] = len(result.get("left_xy", [])) >= 2 and len(result.get("right_xy", [])) >= 2
    return result


def fuse_ego_boundary_keypoints(
    ego_boundaries: list[dict[str, Any]],
    ego_to_anchor: list[np.ndarray] | None = None,
    *,
    max_lateral_jump_m: float = 1.2,
    huber_scale_m: float = 0.35,
    width_shrink: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Causally fuse road keypoints after mapping them to a common ego frame."""
    outputs: list[dict[str, Any]] = []
    previous: dict[str, np.ndarray] | None = None
    clipped = total = 0
    for index, item in enumerate(ego_boundaries):
        if not item.get("valid"):
            outputs.append(item)
            continue
        current = dict(item)
        transformed: dict[str, np.ndarray] = {}
        T = np.eye(4, dtype=np.float64) if ego_to_anchor is None or index >= len(ego_to_anchor) else np.asarray(ego_to_anchor[index], dtype=np.float64)
        for side in ("left", "right"):
            points = np.asarray(item.get(f"{side}_xy", []), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2:
                transformed[side] = points
                continue
            homogeneous = np.c_[points, np.zeros(len(points)), np.ones(len(points))]
            transformed[side] = (T @ homogeneous.T).T[:, :2]
            if previous is not None and len(previous[side]) and len(transformed[side]):
                prior = previous[side]
                query = np.linspace(0.0, 1.0, len(transformed[side]))
                prior_query = np.linspace(0.0, 1.0, len(prior))
                prior_y = np.interp(query, prior_query, prior[:, 1])
                delta = transformed[side][:, 1] - prior_y
                valid = np.isfinite(delta)
                total += int(valid.sum())
                large = valid & (np.abs(delta) > float(max_lateral_jump_m))
                clipped += int(large.sum())
                scale = np.ones_like(delta)
                scale[valid] = np.minimum(1.0, float(huber_scale_m) / np.maximum(np.abs(delta[valid]), 1e-6))
                transformed[side][:, 1] = np.where(valid, prior_y + scale * delta, transformed[side][:, 1])
        for side in ("left", "right"):
            current[f"{side}_xy"] = transformed[side].tolist()
        shrink = float(np.clip(width_shrink, 0.6, 1.4))
        if abs(shrink - 1.0) > 1e-3 and len(transformed["left"]) and len(transformed["right"]):
            count = min(len(transformed["left"]), len(transformed["right"]))
            center = 0.5 * (transformed["left"][:count] + transformed["right"][:count])
            transformed["left"][:count] = center + shrink * (transformed["left"][:count] - center)
            transformed["right"][:count] = center + shrink * (transformed["right"][:count] - center)
            current["left_xy"] = transformed["left"].tolist()
            current["right_xy"] = transformed["right"].tolist()
        previous = transformed
        outputs.append(current)
    return outputs, {"protocol": "ego-frame-boundary-fusion", "clipped_fraction": float(clipped / max(total, 1)), "num_frames": len(ego_boundaries)}


def chain_static_tracks(
    flows: np.ndarray,
    static_weights: np.ndarray,
    roi_mask: np.ndarray,
    *,
    max_tracks: int = 300,
    min_weight: float = 0.25,
    min_track_fraction: float = 0.33,
) -> dict[str, Any]:
    """Chain image points through consecutive flow fields."""
    flow = np.asarray(flows, dtype=np.float64)
    weights = np.asarray(static_weights, dtype=np.float64)
    roi = _mask_array(roi_mask)
    if flow.ndim != 4 or flow.shape[-1] != 2 or weights.shape != flow.shape[:-1] or roi.shape != flow.shape[1:3]:
        raise ValueError("flow, static_weights, and roi_mask have incompatible shapes")
    yy, xx = np.indices(roi.shape)
    # Exclude a small border where even a valid static flow would immediately
    # leave the image and look like a failed track.
    border = 2
    interior = roi & (xx >= border) & (xx < roi.shape[1] - border) & (yy >= border) & (yy < roi.shape[0] - border)
    finite_seed = interior & np.isfinite(flow[0]).all(axis=-1)
    valid = finite_seed & (weights[0] >= float(min_weight))
    seed = np.flatnonzero(valid)
    if len(seed) > max_tracks:
        order = np.argsort(weights[0].reshape(-1)[seed])[::-1][:max_tracks]
        seed = seed[order]
    if not 0.0 < min_track_fraction <= 1.0:
        raise ValueError("min_track_fraction must be in (0,1]")
    tracks: list[list[list[float]]] = []
    lengths: list[int] = []
    all_lengths: list[int] = []
    for flat in seed:
        y, x = np.unravel_index(int(flat), roi.shape)
        points = [[float(x), float(y)]]
        px, py = float(x), float(y)
        ok = True
        for interval in range(flow.shape[0]):
            ix, iy = int(round(px)), int(round(py))
            if not (0 <= ix < roi.shape[1] and 0 <= iy < roi.shape[0]):
                ok = False
                break
            candidates: list[tuple[float, int, int]] = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    cx, cy = ix + dx, iy + dy
                    if 0 <= cx < roi.shape[1] and 0 <= cy < roi.shape[0] and weights[interval, cy, cx] >= float(min_weight) and np.isfinite(flow[interval, cy, cx]).all():
                        candidates.append((float(dx * dx + dy * dy), cx, cy))
            if not candidates:
                ok = False
                break
            _, cx, cy = min(candidates)
            px += float(flow[interval, cy, cx, 0])
            py += float(flow[interval, cy, cx, 1])
            if not (0 <= px < roi.shape[1] and 0 <= py < roi.shape[0]):
                ok = False
                break
            points.append([px, py])
        all_lengths.append(len(points))
        if len(points) >= max(2, int(np.ceil((flow.shape[0] + 1) * min_track_fraction))):
            tracks.append(points)
            lengths.append(len(points))
    coverage = float(len(tracks) / max(len(seed), 1))
    return {
        "tracks": tracks,
        "num_tracks": len(tracks),
        "seed_count": len(seed),
        "finite_seed_count": int(finite_seed.sum()),
        "weighted_seed_count": int(valid.sum()),
        "gated_seed_fraction": float(valid.sum() / max(int(finite_seed.sum()), 1)),
        "mean_track_length": float(np.mean(lengths)) if lengths else 0.0,
        "track_length_fraction": float(np.mean(lengths) / max(flow.shape[0] + 1, 1)) if lengths else 0.0,
        "max_partial_track_length": int(max(all_lengths)) if all_lengths else 0,
        "mean_partial_track_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "coverage": coverage,
    }


def build_road_structure(
    road_masks: np.ndarray,
    flows: np.ndarray,
    static_weights: np.ndarray,
    roi_mask: np.ndarray,
    *,
    near_start: float = 0.55,
) -> dict[str, Any]:
    """Build all road-relative image evidence without metric reconstruction."""
    masks = np.asarray(road_masks, dtype=bool)
    if masks.ndim != 3 or len(masks) == 0:
        raise ValueError("road_masks must have shape [T,H,W]")
    anchor = masks[0]
    near, far = split_near_far(anchor, near_start=near_start)
    near_boundary = extract_road_boundaries(near)
    far_boundary = extract_road_boundaries(far)
    return {
        "protocol": "road-structure-evidence",
        "near_road": {"mask_fraction": float(np.mean(near)), "boundaries": near_boundary, "descriptor": _boundary_descriptor(near_boundary, anchor.shape[1])},
        "far_road": {"mask_fraction": float(np.mean(far)), "boundaries": far_boundary, "descriptor": _boundary_descriptor(far_boundary, anchor.shape[1])},
        "global_flow_heading": estimate_focus_of_expansion(flows, static_weights, roi_mask),
        "long_term_static_tracks": chain_static_tracks(flows, static_weights, roi_mask),
        "road_mask_fraction": float(np.mean(masks)),
    }

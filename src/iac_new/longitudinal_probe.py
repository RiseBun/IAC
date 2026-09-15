"""Metric longitudinal-motion probes from static cross-frame geometry.

This module deliberately does not infer action causality.  It estimates the
camera displacement that is visible in a pair of frames, using either a known
road plane (homography) or metric 3-D points (PnP).  The two estimates are
kept independent and may only be fused after an explicit agreement gate.

The relative-pose convention is

    X_next = R @ X_current + t

and the road-plane homography follows the IAC convention

    H = K @ (R - t n^T / d) @ inv(K).

Here ``t`` is the translation in the next-camera coordinate frame.  The
camera-centre displacement expressed in the current camera frame is
``-R.T @ t``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite_matrix(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _normalise_plane(normal: np.ndarray, distance: float) -> tuple[np.ndarray, float]:
    n = np.asarray(normal, dtype=np.float64).reshape(-1)
    if n.shape != (3,) or not np.isfinite(n).all():
        raise ValueError("plane normal must be a finite [3] vector")
    norm = float(np.linalg.norm(n))
    if norm <= 1e-8 or not np.isfinite(distance) or abs(float(distance)) <= 1e-8:
        raise ValueError("plane normal and distance must be non-zero")
    return n / norm, float(distance) / norm


def _normalise_homography(value: np.ndarray) -> np.ndarray:
    h = _finite_matrix(value, (3, 3), "homography")
    scale = float(h[2, 2])
    if abs(scale) <= 1e-10:
        scale = float(np.linalg.norm(h))
    if abs(scale) <= 1e-10:
        raise ValueError("homography has zero scale")
    return h / scale


def _dlt_homography(current_points: np.ndarray, next_points: np.ndarray) -> np.ndarray:
    """Estimate a homography without requiring OpenCV.

    This fallback is intended for deterministic tests and small diagnostic
    runs.  Production callers should use ``cv2.findHomography`` with RANSAC
    when OpenCV is available.
    """
    source = np.asarray(current_points, dtype=np.float64)
    target = np.asarray(next_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("point correspondences must both have shape [N,2]")
    if len(source) < 4:
        raise ValueError("at least four point correspondences are required")
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source, target):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64), full_matrices=False)
    h = vh[-1].reshape(3, 3)
    return _normalise_homography(h)


def estimate_homography_from_correspondences(
    current_points: np.ndarray,
    next_points: np.ndarray,
    *,
    ransac_reprojection_threshold_px: float = 2.0,
) -> dict[str, Any]:
    """Estimate a frame-to-frame homography and report geometric support.

    OpenCV RANSAC is used when available.  The deterministic DLT fallback is
    intentionally fail-soft and reports all correspondences as inliers.
    """
    source = np.asarray(current_points, dtype=np.float64)
    target = np.asarray(next_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("point correspondences must both have shape [N,2]")
    if len(source) < 4:
        return {"available": False, "reason": "insufficient_correspondences", "num_points": int(len(source))}
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("point correspondences must be finite")
    threshold = float(ransac_reprojection_threshold_px)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("ransac_reprojection_threshold_px must be positive")
    homography = None
    inliers = None
    backend = "dlt"
    try:
        import cv2  # type: ignore

        homography, inliers = cv2.findHomography(
            source.astype(np.float32),
            target.astype(np.float32),
            cv2.RANSAC,
            threshold,
        )
        if homography is not None:
            homography = _normalise_homography(homography)
            backend = "opencv_ransac"
    except Exception:
        # OpenCV is optional for the core module.  The deterministic DLT path
        # below keeps the probe testable in the lightweight local runtime;
        # production runs should report the backend in their JSON output.
        homography = None
        inliers = None
    if homography is None:
        homography = _dlt_homography(source, target)
        inliers = np.ones((len(source), 1), dtype=np.uint8)
    projected = _project_points(homography, source)
    residual = np.linalg.norm(projected - target, axis=1)
    mask = np.asarray(inliers, dtype=bool).reshape(-1)
    finite = np.isfinite(residual)
    mask &= finite
    return {
        "available": bool(mask.sum() >= 4),
        "backend": backend,
        "homography": homography.tolist(),
        "inlier_mask": mask.tolist(),
        "num_points": int(len(source)),
        "num_inliers": int(mask.sum()),
        "inlier_fraction": float(mask.mean()),
        "median_reprojection_error_px": float(np.median(residual[mask])) if mask.any() else None,
        "p95_reprojection_error_px": float(np.quantile(residual[mask], 0.95)) if mask.any() else None,
    }


def _project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    h = _finite_matrix(homography, (3, 3), "homography")
    xy = np.asarray(points, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("points must have shape [N,2]")
    homogeneous = np.c_[xy, np.ones(len(xy), dtype=np.float64)].T
    projected = h @ homogeneous
    denominator = projected[2]
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-10)
    result = np.full((len(xy), 2), np.nan, dtype=np.float64)
    result[valid] = (projected[:2, valid] / denominator[valid]).T
    return result


def _solve_metric_translation(
    homography: np.ndarray,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    normal: np.ndarray,
    distance: float,
) -> tuple[np.ndarray, float]:
    """Solve the metric translation from H, R, n and d.

    Homographies are only known up to a scalar.  The scalar and the scaled
    translation are solved jointly by linear least squares, after which the
    translation is de-scaled.
    """
    K = _finite_matrix(intrinsics, (3, 3), "intrinsics")
    R = _finite_matrix(rotation, (3, 3), "rotation")
    n, d = _normalise_plane(normal, float(distance))
    H = _finite_matrix(homography, (3, 3), "homography")
    A = np.linalg.inv(K) @ H @ K
    design: list[list[float]] = []
    target: list[float] = []
    for row in range(3):
        for col in range(3):
            design.append([R[row, col], -(1.0 / d) * (1.0 if row == 0 else 0.0) * n[col], -(1.0 / d) * (1.0 if row == 1 else 0.0) * n[col], -(1.0 / d) * (1.0 if row == 2 else 0.0) * n[col]])
            target.append(float(A[row, col]))
    solution, _, _, _ = np.linalg.lstsq(np.asarray(design), np.asarray(target), rcond=None)
    scale = float(solution[0])
    if not np.isfinite(scale) or abs(scale) <= 1e-8:
        raise ValueError("homography decomposition has degenerate scale")
    translation = np.asarray(solution[1:4], dtype=np.float64) / scale
    return translation, scale


def estimate_planar_translation(
    homography: np.ndarray,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance_m: float,
    *,
    camera_to_ego: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate metric camera-centre displacement from a known road plane."""
    try:
        translation_next_camera, homography_scale = _solve_metric_translation(
            homography,
            intrinsics,
            rotation,
            plane_normal_camera,
            plane_distance_m,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"available": False, "reason": str(exc)}
    R = _finite_matrix(rotation, (3, 3), "rotation")
    centre_current_camera = -R.T @ translation_next_camera
    centre_ego = centre_current_camera.copy()
    if camera_to_ego is not None:
        transform = _finite_matrix(camera_to_ego, (4, 4), "camera_to_ego")
        centre_ego = transform[:3, :3] @ centre_current_camera
    return {
        "available": bool(np.isfinite(centre_ego).all()),
        "translation_next_camera_m": translation_next_camera.tolist(),
        "camera_center_displacement_current_camera_m": centre_current_camera.tolist(),
        "camera_center_displacement_ego_m": centre_ego.tolist(),
        "longitudinal_m": float(centre_ego[0]),
        "lateral_m": float(centre_ego[1]),
        "homography_scale": homography_scale,
        "plane_distance_m": float(plane_distance_m),
    }


def estimate_pnp_translation(
    points_3d_current_m: np.ndarray,
    points_2d_next: np.ndarray,
    intrinsics: np.ndarray,
    *,
    camera_to_ego: np.ndarray | None = None,
    reprojection_error_px: float = 2.0,
    iterations_count: int = 100,
) -> dict[str, Any]:
    """Estimate metric translation from first-frame metric 3-D points.

    This backend intentionally requires OpenCV.  Missing OpenCV is an explicit
    unavailable result rather than a silent fallback to a non-metric estimate.
    """
    points_3d = np.asarray(points_3d_current_m, dtype=np.float64)
    points_2d = np.asarray(points_2d_next, dtype=np.float64)
    K = _finite_matrix(intrinsics, (3, 3), "intrinsics")
    if points_3d.ndim != 2 or points_3d.shape[1] != 3 or points_2d.shape != (len(points_3d), 2):
        raise ValueError("PnP points must have shapes [N,3] and [N,2]")
    if len(points_3d) < 6:
        return {"available": False, "reason": "insufficient_3d_correspondences", "num_points": int(len(points_3d))}
    try:
        import cv2  # type: ignore
    except ImportError:
        return {"available": False, "reason": "opencv_unavailable"}
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d.astype(np.float32),
        points_2d.astype(np.float32),
        K,
        None,
        iterationsCount=int(iterations_count),
        reprojectionError=float(reprojection_error_px),
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or rvec is None or tvec is None:
        return {"available": False, "reason": "pnp_failed", "num_points": int(len(points_3d))}
    R, _ = cv2.Rodrigues(rvec)
    translation_next_camera = np.asarray(tvec, dtype=np.float64).reshape(3)
    centre_current_camera = -R.T @ translation_next_camera
    centre_ego = centre_current_camera.copy()
    if camera_to_ego is not None:
        transform = _finite_matrix(camera_to_ego, (4, 4), "camera_to_ego")
        centre_ego = transform[:3, :3] @ centre_current_camera
    inlier_mask = np.zeros(len(points_3d), dtype=bool)
    if inliers is not None:
        inlier_mask[np.asarray(inliers).reshape(-1)] = True
    projected, _ = cv2.projectPoints(points_3d.astype(np.float32), rvec, tvec, K, None)
    residual = np.linalg.norm(projected.reshape(-1, 2) - points_2d, axis=1)
    return {
        "available": bool(np.isfinite(centre_ego).all() and inlier_mask.sum() >= 6),
        "rotation": np.asarray(R, dtype=np.float64).tolist(),
        "translation_next_camera_m": translation_next_camera.tolist(),
        "camera_center_displacement_current_camera_m": centre_current_camera.tolist(),
        "camera_center_displacement_ego_m": centre_ego.tolist(),
        "longitudinal_m": float(centre_ego[0]),
        "lateral_m": float(centre_ego[1]),
        "num_points": int(len(points_3d)),
        "num_inliers": int(inlier_mask.sum()),
        "inlier_fraction": float(inlier_mask.mean()),
        "median_reprojection_error_px": float(np.median(residual[inlier_mask])) if inlier_mask.any() else None,
        "p95_reprojection_error_px": float(np.quantile(residual[inlier_mask], 0.95)) if inlier_mask.any() else None,
    }


def estimate_translation_with_known_rotation(
    points_3d_current_m: np.ndarray,
    points_2d_next: np.ndarray,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    *,
    camera_to_ego: np.ndarray | None = None,
    reprojection_error_px: float = 3.0,
    max_trials: int = 400,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Estimate only translation while keeping relative rotation frozen.

    With ``X_next = R X_current + t``, each 3-D/2-D correspondence gives two
    linear equations in ``t``.  A deterministic two-point RANSAC rejects
    dynamic objects and bad depth before a final least-squares refinement.
    """
    points_3d = np.asarray(points_3d_current_m, dtype=np.float64)
    points_2d = np.asarray(points_2d_next, dtype=np.float64)
    K = _finite_matrix(intrinsics, (3, 3), "intrinsics")
    R = _finite_matrix(rotation, (3, 3), "rotation")
    if points_3d.ndim != 2 or points_3d.shape[1] != 3 or points_2d.shape != (len(points_3d), 2):
        raise ValueError("known-rotation points must have shapes [N,3] and [N,2]")
    if len(points_3d) < 6:
        return {"available": False, "reason": "insufficient_3d_correspondences", "num_points": int(len(points_3d))}
    normalized = (np.linalg.inv(K) @ np.c_[points_2d, np.ones(len(points_2d))].T).T
    normalized = normalized[:, :2] / normalized[:, 2:3]
    rotated = (R @ points_3d.T).T
    design = np.zeros((2 * len(points_3d), 3), dtype=np.float64)
    target = np.zeros(2 * len(points_3d), dtype=np.float64)
    design[0::2, 0] = 1.0
    design[0::2, 2] = -normalized[:, 0]
    design[1::2, 1] = 1.0
    design[1::2, 2] = -normalized[:, 1]
    target[0::2] = normalized[:, 0] * rotated[:, 2] - rotated[:, 0]
    target[1::2] = normalized[:, 1] * rotated[:, 2] - rotated[:, 1]

    def residual(translation: np.ndarray) -> np.ndarray:
        moved = rotated + translation.reshape(1, 3)
        projected = (K @ moved.T).T
        xy = projected[:, :2] / projected[:, 2:3]
        error = np.linalg.norm(xy - points_2d, axis=1)
        error[(moved[:, 2] <= 1e-6) | ~np.isfinite(error)] = np.inf
        return error

    rng = np.random.default_rng(int(random_seed))
    best_mask = np.zeros(len(points_3d), dtype=bool)
    best_median = np.inf
    trials = min(int(max_trials), max(1, len(points_3d) * 4))
    for _ in range(trials):
        chosen = rng.choice(len(points_3d), size=2, replace=False)
        rows = np.ravel(np.c_[2 * chosen, 2 * chosen + 1])
        candidate, _, rank, _ = np.linalg.lstsq(design[rows], target[rows], rcond=None)
        if rank < 3:
            continue
        error = residual(candidate)
        mask = error <= float(reprojection_error_px)
        median = float(np.median(error[mask])) if mask.any() else np.inf
        if int(mask.sum()) > int(best_mask.sum()) or (int(mask.sum()) == int(best_mask.sum()) and median < best_median):
            best_mask, best_median = mask, median
    if int(best_mask.sum()) < 6:
        return {"available": False, "reason": "known_rotation_ransac_failed", "num_points": int(len(points_3d))}
    rows = np.repeat(best_mask, 2)
    translation, _, rank, _ = np.linalg.lstsq(design[rows], target[rows], rcond=None)
    if rank < 3:
        return {"available": False, "reason": "known_rotation_translation_degenerate", "num_points": int(len(points_3d))}
    error = residual(translation)
    inlier_mask = error <= float(reprojection_error_px)
    centre_current_camera = -R.T @ translation
    centre_ego = centre_current_camera.copy()
    if camera_to_ego is not None:
        transform = _finite_matrix(camera_to_ego, (4, 4), "camera_to_ego")
        centre_ego = transform[:3, :3] @ centre_current_camera
    return {
        "available": bool(np.isfinite(centre_ego).all() and inlier_mask.sum() >= 6),
        "rotation": R.tolist(),
        "translation_next_camera_m": translation.tolist(),
        "camera_center_displacement_current_camera_m": centre_current_camera.tolist(),
        "camera_center_displacement_ego_m": centre_ego.tolist(),
        "longitudinal_m": float(centre_ego[0]),
        "lateral_m": float(centre_ego[1]),
        "num_points": int(len(points_3d)),
        "num_inliers": int(inlier_mask.sum()),
        "inlier_fraction": float(inlier_mask.mean()),
        "median_reprojection_error_px": float(np.median(error[inlier_mask])) if inlier_mask.any() else None,
        "p95_reprojection_error_px": float(np.quantile(error[inlier_mask], 0.95)) if inlier_mask.any() else None,
    }


def agreement_gate(
    planar: dict[str, Any],
    pnp: dict[str, Any],
    *,
    max_relative_disagreement: float = 0.35,
    near_zero_m: float = 0.10,
) -> dict[str, Any]:
    """Fuse A/B only when sign and scale agree; otherwise abstain."""
    if not planar.get("available") or not pnp.get("available"):
        return {"status": "uncertain", "reason": "backend_unavailable"}
    a = float(planar.get("longitudinal_m", np.nan))
    b = float(pnp.get("longitudinal_m", np.nan))
    if not np.isfinite(a) or not np.isfinite(b):
        return {"status": "uncertain", "reason": "nonfinite_backend_estimate"}
    sign_conflict = abs(a) > near_zero_m and abs(b) > near_zero_m and np.sign(a) != np.sign(b)
    denominator = max(abs(a), abs(b), near_zero_m)
    relative_disagreement = abs(a - b) / denominator
    if sign_conflict:
        return {"status": "uncertain", "reason": "sign_disagreement", "planar_m": a, "pnp_m": b, "relative_disagreement": relative_disagreement}
    if relative_disagreement > float(max_relative_disagreement):
        return {"status": "uncertain", "reason": "scale_disagreement", "planar_m": a, "pnp_m": b, "relative_disagreement": relative_disagreement}
    fused = float(np.median([a, b]))
    return {
        "status": "scored",
        "reason": "backend_agreement",
        "planar_m": a,
        "pnp_m": b,
        "fused_longitudinal_m": fused,
        "relative_disagreement": relative_disagreement,
    }


def cumulative_progress_from_intervals(
    interval_longitudinal_m: np.ndarray,
    frame_times_s: np.ndarray,
) -> dict[str, Any]:
    """Accumulate only observed frame-to-frame intervals.

    For frames at t=[1,2,3,4], the input contains 1->2, 2->3, 3->4 and
    produces progress at endpoints [2,3,4].  No unobserved 0->1 distance is
    invented.
    """
    displacement = np.asarray(interval_longitudinal_m, dtype=np.float64).reshape(-1)
    times = np.asarray(frame_times_s, dtype=np.float64).reshape(-1)
    if len(times) != len(displacement) + 1:
        raise ValueError("frame_times_s must contain one more timestamp than intervals")
    if len(times) and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0)):
        raise ValueError("frame_times_s must be finite and strictly increasing")
    valid = np.isfinite(displacement)
    if not valid.all():
        return {"status": "uncertain", "reason": "nonfinite_interval", "endpoint_times_s": times[1:].tolist()}
    cumulative = np.cumsum(displacement)
    return {
        "status": "scored",
        "endpoint_times_s": times[1:].tolist(),
        "interval_longitudinal_m": displacement.tolist(),
        "cumulative_progress_m": cumulative.tolist(),
        "observed_interval_count": int(len(displacement)),
        "interval_scale_drift_m": float(np.max(displacement) - np.min(displacement)) if len(displacement) else 0.0,
    }

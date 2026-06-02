#!/usr/bin/env python3
"""Geometry-based trajectory metrics for IAC (VIPe replacement).

iWorld-Bench uses VIPe (Huang et al., 2025), a third-party unsupervised
video pose estimator, to extract camera trajectories from WAM-generated
videos. We don't have VIPe locally and the project shouldn't depend on a
growing list of third-party packages, so this module provides two
geometry-based estimators built on OpenCV's standard SfM primitives:

  - estimate_trajectory_from_video(): recovers a 2D ego-motion trajectory
    from optical flow + essential matrix. Sufficient for "did the camera
    move forward or left?", which is what IAC needs.

  - compute_trajectory_accuracy(): cosine similarity between the recovered
    trajectory's tangent direction and the GT trajectory direction.

  - compute_trajectory_tolerance(): same as accuracy but uses an
    "estimator-agnostic" reference (e.g. nuPlan's own ego_pose) to
    cancel estimation bias, mirroring iWorld-Bench's Eq. 14.

  - compute_trajectory_alignment(): round-trip / loop-closure mirror
    alignment for memory tasks, mirroring iWorld-Bench's Eq. 17.

These are weaker than VIPe (no full 6-DoF SfM with bundle adjustment) but
they have zero extra dependencies beyond OpenCV + NumPy, which are
already in the nuPlan stack, and they preserve iWorld-Bench's "geometric
ground truth" idea: even if the IAC critic is wrong, the recovered
trajectory can be checked against ego_pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ───────────────────────── Trajectory data structures ─────────────────────────


@dataclass
class Traj2D:
    """A 2D trajectory in ego (vehicle) frame, units = metres.

    Convention: x=forward, y=left, yaw=heading angle (CCW positive).
    """
    xs: np.ndarray  # (T,)
    ys: np.ndarray  # (T,)
    yaws: np.ndarray  # (T,)

    @property
    def T(self) -> int:
        return len(self.xs)

    def to_global(self, x0: float = 0.0, y0: float = 0.0, yaw0: float = 0.0) -> "Traj2D":
        """Transform to global frame by chaining ego-frame poses."""
        xs, ys, yaws = [x0], [y0], [yaw0]
        x, y, yaw = x0, y0, yaw0
        for dx, dy, dyaw in zip(np.diff(self.xs), np.diff(self.ys), np.diff(self.yaws)):
            cy, sy = math.cos(yaw), math.sin(yaw)
            x += cy * dx - sy * dy
            y += sy * dx + cy * dy
            yaw += dyaw
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
        return Traj2D(np.array(xs), np.array(ys), np.array(yaws))

    def tangent(self) -> np.ndarray:
        """Per-step 2D tangent (forward, lateral, dyaw) — used for
        cosine-similarity accuracy metrics (iWorld-Bench Eq. 13)."""
        if self.T < 2:
            return np.zeros((0, 3), dtype=np.float32)
        dxs = np.diff(self.xs)
        dys = np.diff(self.ys)
        dyaws = np.diff(self.yaws)
        return np.stack([dxs, dys, dyaws], axis=1)


# ───────────────────────── Optical-flow-based 2D recovery ─────────────────────────


def _load_gray(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def estimate_trajectory_from_video(
    frame_paths: Sequence[str],
    intrinsics: Optional[np.ndarray] = None,
    size: int = 224,
    method: str = "essential",
) -> Traj2D:
    """Recover a 2D ego-motion trajectory from a sequence of frames.

    Args:
        frame_paths: ordered list of image paths (>=2)
        intrinsics: 3x3 camera intrinsic matrix. If None, uses a
            reasonable default for a forward-looking dashcam at the
            given size.
        size: square resize for both feature detection and intrinsics.
        method: 'essential' (recover pose via essential matrix) or
            'flow' (simpler Lucas–Kanade translation only).

    Returns:
        Traj2D with T=len(frame_paths) entries, in arbitrary scale
        (the metric downstream rescales by GT length).
    """
    if len(frame_paths) < 2:
        raise ValueError("Need at least 2 frames")

    if intrinsics is None:
        # Rough pinhole: fov ~ 60°, principal at center
        f = size / (2.0 * math.tan(math.radians(30.0)))
        intrinsics = np.array([[f, 0, size / 2.0], [0, f, size / 2.0], [0, 0, 1]], dtype=np.float32)

    grays = [_load_gray(p, size) for p in frame_paths]
    feature = cv2.SIFT_create(nfeatures=200)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    xs, ys, yaws = [0.0], [0.0], [0.0]
    x, y, yaw = 0.0, 0.0, 0.0

    for i in range(len(grays) - 1):
        kp1, des1 = feature.detectAndCompute(grays[i], None)
        kp2, des2 = feature.detectAndCompute(grays[i + 1], None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            dx, dy, dyaw = 0.0, 0.0, 0.0
        else:
            matches = matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance][:50]
            if len(good) < 8:
                dx, dy, dyaw = 0.0, 0.0, 0.0
            else:
                pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float32)
                pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float32)
                if method == "essential":
                    E, mask = cv2.findEssentialMat(
                        pts1, pts2, intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0
                    )
                    if E is None:
                        dx, dy, dyaw = 0.0, 0.0, 0.0
                    else:
                        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, intrinsics, mask=mask)
                        # t is the camera-frame translation (z forward).
                        # We integrate in vehicle frame: forward=z, lateral=-x.
                        dx, dy = float(t[2, 0]), float(-t[0, 0])
                        dyaw = float(math.atan2(R[1, 0], R[0, 0]))
                else:  # flow fallback
                    pts1r = pts1.reshape(-1, 1, 2)
                    pts2r = pts2.reshape(-1, 1, 2)
                    dx_pix = float(np.median(pts2r[..., 0] - pts1r[..., 0]))
                    dy_pix = float(np.median(pts2r[..., 1] - pts1r[..., 1]))
                    dx = dy_pix * 0.05  # rough scale
                    dy = -dx_pix * 0.05
                    dyaw = 0.0

        cy, sy = math.cos(yaw), math.sin(yaw)
        x += cy * dx - sy * dy
        y += sy * dx + cy * dy
        yaw += dyaw
        xs.append(x)
        ys.append(y)
        yaws.append(yaw)

    return Traj2D(np.array(xs), np.array(ys), np.array(yaws))


# ───────────────────────── Metrics (iWorld-Bench §3.3.2) ─────────────────────────


def _upper_convex(x: np.ndarray, k: float = 10.0) -> np.ndarray:
    """Logarithmic transform (Eq. 10) that emphasises low-mid scores."""
    x_safe = np.clip(x.astype(np.float64), 0.0, 1.0)
    return np.log(1.0 + k * x_safe) / np.log(1.0 + k)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _rescale_to_match_length(est: Traj2D, gt: Traj2D) -> Traj2D:
    """Scale the estimated trajectory to match GT length (since optical
    flow + essential matrix recover motion only up to scale)."""
    est_len = np.linalg.norm(np.diff(est.xs).sum(), np.diff(est.ys).sum()) + 1e-8
    gt_len = np.linalg.norm(np.diff(gt.xs).sum(), np.diff(gt.ys).sum()) + 1e-8
    scale = gt_len / est_len
    return Traj2D(est.xs * scale, est.ys * scale, est.yaws)


def compute_trajectory_accuracy(est: Traj2D, gt: Traj2D) -> float:
    """iWorld-Bench §3.3.2 Trajectory Accuracy (Eq. 13).

    Per-step cosine similarity of the (dx, dy, dyaw) tangent, averaged
    across the trajectory, passed through the upper-convex transform to
    amplify low-mid scores.
    """
    est_r = _rescale_to_match_length(est, gt)
    min_t = min(est_r.T, gt.T) - 1
    if min_t <= 0:
        return 0.0
    a_tan = est_r.tangent()[:min_t]
    b_tan = gt.tangent()[:min_t]
    sims = np.array([_cosine(a_tan[t], b_tan[t]) for t in range(min_t)])
    return float(_upper_convex(np.abs(sims)).mean())


def compute_trajectory_tolerance(est: Traj2D, ref: Traj2D, gt: Traj2D) -> float:
    """iWorld-Bench §3.3.2 Trajectory Tolerance (Eq. 14).

    Uses an estimator-agnostic reference (e.g. another SfM baseline) to
    cancel estimation uncertainty. The paper calls for ground-truth
    extrinsic E_gt; we use ref as that anchor.
    """
    est_r = _rescale_to_match_length(est, gt)
    ref_r = _rescale_to_match_length(ref, gt)
    min_t = min(est_r.T, ref_r.T, gt.T) - 1
    if min_t <= 0:
        return 0.0
    a_tan = est_r.tangent()[:min_t]
    b_tan = ref_r.tangent()[:min_t]
    sims = np.array([_cosine(a_tan[t], b_tan[t]) for t in range(min_t)])
    return float(_upper_convex(np.abs(sims)).mean())


def compute_trajectory_alignment(forward: Traj2D, reverse: Traj2D) -> float:
    """iWorld-Bench §3.3.2 Trajectory Alignment (Eq. 17).

    For a memory task: 'do a forward action, then reverse it, end up
    where you started?'. Score is the mirror symmetry between the
    first half (forward) and the second half (reverse), measured as
    mean of |S(E_{t+1}, -E_{T-t+1})|.

    Note: the recovered trajectory is in arbitrary scale, so we
    rescale both halves to unit length first.
    """
    T_f, T_r = forward.T, reverse.T
    if T_f < 2 or T_r < 2:
        return 0.0
    L_f = max(np.linalg.norm(np.diff(forward.xs).sum(), np.diff(forward.ys).sum()), 1e-8)
    L_r = max(np.linalg.norm(np.diff(reverse.xs).sum(), np.diff(reverse.ys).sum()), 1e-8)
    f_n = Traj2D(forward.xs / L_f, forward.ys / L_f, forward.yaws)
    r_n = Traj2D(reverse.xs / L_r, reverse.ys / L_r, reverse.yaws)

    # Mirror second half: flip y axis (in vehicle frame, going backward
    # means forward becomes negative).
    n = min(T_f - 1, T_r - 1)
    sims = []
    for t in range(1, n + 1):
        a = np.array([np.diff(f_n.xs)[t - 1], np.diff(f_n.ys)[t - 1], np.diff(f_n.yaws)[t - 1]])
        b = np.array([-np.diff(r_n.xs)[t - 1], -np.diff(r_n.ys)[t - 1], np.diff(r_n.yaws)[t - 1]])
        sims.append(abs(_cosine(a, b)))
    return float(_upper_convex(np.array(sims)).mean())


# ───────────────────────── aggregate entry point ─────────────────────────


def evaluate_video_geometry(
    frame_paths: Sequence[str],
    gt_traj: Traj2D,
    ref_frame_paths: Optional[Sequence[str]] = None,
    intrinsics: Optional[np.ndarray] = None,
    size: int = 224,
) -> dict:
    """Run the iWorld-Bench geometry metrics on a recovered trajectory.

    Returns a dict with keys: trajectory_accuracy, trajectory_tolerance
    (only if ref_frame_paths given), trajectory_alignment (only if the
    sequence has a memory-task structure — caller can pass `reverse_paths`
    to enable).
    """
    est = estimate_trajectory_from_video(frame_paths, intrinsics=intrinsics, size=size)
    out: dict = {"trajectory_accuracy": compute_trajectory_accuracy(est, gt_traj)}
    if ref_frame_paths is not None and len(ref_frame_paths) >= 2:
        ref = estimate_trajectory_from_video(ref_frame_paths, intrinsics=intrinsics, size=size)
        out["trajectory_tolerance"] = compute_trajectory_tolerance(est, ref, gt_traj)
    return out


# ───────────────────────── nuPlan GT trajectory helper ─────────────────────────


def ego_state_to_traj(ego_states: Sequence[Sequence[float]], dt_s: float = 0.5) -> Traj2D:
    """Convert nuPlan ego_state[5] = (vx, vy, yaw, ax, yaw_rate) sequence
    plus a constant dt into a Traj2D, integrating forward.

    This is a *truth* trajectory and is what the recovered one should
    match. We return it in ego-initial frame: starts at (0, 0, 0) and
    accumulates x, y, yaw from per-step motion.
    """
    xs, ys, yaws = [0.0], [0.0], [0.0]
    x, y, yaw = 0.0, 0.0, 0.0
    for s in ego_states[1:]:
        vx, vy, yaw_i, *_ = s
        dx = float(vx) * dt_s
        dy = float(vy) * dt_s
        cy, sy = math.cos(yaw), math.sin(yaw)
        x += cy * dx - sy * dy
        y += sy * dx + cy * dy
        yaw = float(yaw_i)
        xs.append(x)
        ys.append(y)
        yaws.append(yaw)
    return Traj2D(np.array(xs), np.array(ys), np.array(yaws))


def candidate_traj_to_traj(candidate_traj: Sequence[Sequence[float]]) -> Traj2D:
    """Convert a candidate_traj[T, 3] = (dx, dy, dyaw) list to Traj2D by
    cumulative summing in ego frame."""
    arr = np.array(candidate_traj, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    xs = np.cumsum(arr[:, 0])
    ys = np.cumsum(arr[:, 1])
    yaws = np.cumsum(arr[:, 2])
    return Traj2D(np.concatenate([[0.0], xs]), np.concatenate([[0.0], ys]), np.concatenate([[0.0], yaws]))


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description="Estimate trajectory from a video")
    p.add_argument("--frames", nargs="+", required=True, help="frame paths in order")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--method", choices=["essential", "flow"], default="essential")
    args = p.parse_args()

    traj = estimate_trajectory_from_video(args.frames, size=args.size, method=args.method)
    print(json.dumps({
        "xs": traj.xs.tolist(),
        "ys": traj.ys.tolist(),
        "yaws": traj.yaws.tolist(),
    }, indent=2))

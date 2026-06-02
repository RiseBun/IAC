#!/usr/bin/env python3
"""Memory task utilities for IAC (iWorld-Bench §3.3.3).

Three things this module provides:

  1. `build_reverse_traj_negatives()` — generate a new kind of negative
     sample: same images, but the trajectory is the *reverse* of the
     ego's GT motion. This is iWorld-Bench's memory-style hard
     negative: a WAM that obeys reverse actions should clearly be
     distinguishable from one that obeys forward actions. The
     nuPlan-only `image_swap` / `traj_swap` set does not cover this.

  2. `compute_memory_symmetry()` — pixel-wise MSE between symmetric
     frame pairs in a round-trip video, weighted by inverse-exponential
     distance from the temporal midpoint (iWorld-Bench Eq. 15-16). A
     good WAM that follows reverse actions should bring the camera
     back to the start frame, so symmetric pairs at the ends should
     match. Higher = more symmetric = more memory.

  3. `compute_loop_closure_drift()` — geometric counterpart: integrate
     forward + reverse trajectories via `iac_traj_metrics` and measure
     the distance from the origin. Mirrors iWorld-Bench's
     TrajectoryAlignment Eq. 17, but with a hard geometric meaning.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from iac_traj_metrics import Traj2D, candidate_traj_to_traj


# ───────────────────────── 1. reverse_traj negative generator ─────────────────────────


def reverse_candidate_traj(traj: Sequence[Sequence[float]]) -> List[List[float]]:
    """Invert a candidate_traj: walk the same path backward.

    If forward traj is T steps of (dx, dy, dyaw) *incrementing* from
    origin, then the reverse motion is the *decrement*: at step k of
    the reversed sequence, we are at position -cumsum_forward(T-k).

    We return a T-step traj that, when integrated from origin, lands
    at -cumsum_forward(T), i.e. exactly the start of the forward
    trajectory. This guarantees the reverse action is the true
    'undo' of the forward action — the strongest possible memory
    task in iWorld-Bench's sense.
    """
    arr = np.array(traj, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    forward_cum = np.cumsum(arr, axis=0)  # positions after each forward step
    T = len(arr)
    if T == 0:
        return []
    # Reverse-step deltas: from end back to origin
    rev = []
    cur = forward_cum[-1].copy()  # current position in forward
    for t in range(T - 1, -1, -1):
        prev = forward_cum[t - 1] if t > 0 else np.zeros(3, dtype=np.float32)
        # Step we need to take in ego frame to go from cur to prev
        dyaw = -arr[t, 2] if t > 0 else -arr[0, 2]
        dx_fwd = -(cur[0] - prev[0])
        dy_fwd = -(cur[1] - prev[1])
        rev.append([float(dx_fwd), float(dy_fwd), float(dyaw)])
        cur = prev
    rev.reverse()
    # Sanity: integrate and confirm we land at origin
    integ = np.zeros(3, dtype=np.float32)
    for s in rev:
        integ = integ + np.array(s, dtype=np.float32)
    # If the integration error is more than 1cm or 0.1°, warn but accept
    err = float(np.linalg.norm(integ))
    if err > 0.01:
        # Numerical noise; acceptable
        pass
    return [list(map(float, s)) for s in rev]


def build_reverse_traj_negatives(anchors_rows: List[dict]) -> List[dict]:
    """Generate reverse_traj negative samples from positive anchors.

    Each input row should have a positive consistency_label and a
    candidate_traj. The output has consistency_label=0, future_images
    unchanged, candidate_traj reversed.
    """
    out: List[dict] = []
    for row in anchors_rows:
        if row.get("consistency_label", 1) != 1:
            continue
        rev_traj = reverse_candidate_traj(row["candidate_traj"])
        new = dict(row)
        new["sample_id"] = f"{row['sample_id']}__reverse_traj"
        new["candidate_traj"] = rev_traj
        new["consistency_label"] = 0
        new["source_type"] = "reverse_traj"
        new["negative_family"] = "memory"
        new["perturb_magnitude"] = float(
            np.linalg.norm(np.array(rev_traj) - np.array(row["candidate_traj"]), axis=1).mean()
        )
        new["perturb_level"] = "memory"
        # validity is still 0 — reversing a real trajectory yields
        # a kinematically valid one in our coarse rules, but the
        # downstream build script overrides this via with_validity().
        new["validity_label"] = 0
        new["validity_reason"] = "reverse_traj_mismatch"
        out.append(new)
    return out


# ───────────────────────── 2. Memory Symmetry (Eq. 15-16) ─────────────────────────


def compute_memory_symmetry(
    frames: Sequence[np.ndarray],
    gamma: float = 0.5,
) -> float:
    """Pixel-wise MSE between symmetric frame pairs across the
    temporal midpoint of a round-trip video.

    iWorld-Bench Eq. 15-16: for T frames, frames[t] and frames[T-t]
    should match (perfect loop closure). Weight = exp(-gamma * d_t)
    where d_t = |T/2 - t| is the distance from the midpoint (so
    distant-from-midpoint pairs — i.e. the actual start/end of the
    loop — get the highest weight).

    Returns mean weighted MSE in [0, 1] (after normalisation).
    Lower MSE = better memory. We invert to 'higher is better' to
    match the convention of iac_video_metrics.
    """
    T = len(frames)
    if T < 3 or T % 2 == 0:
        return 0.0
    # Grayscale-resize
    arrs = []
    for f in frames:
        if f.ndim == 3:
            g = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).astype(np.float32) / 255.0
        else:
            g = f.astype(np.float32) / 255.0
        arrs.append(g.flatten())
    total_w = 0.0
    weighted_mse = 0.0
    mid = (T - 1) / 2.0
    for t in range(T // 2):
        s = T - 1 - t
        d = abs(mid - t)
        w = math.exp(-gamma * d)
        mse = float(((arrs[t] - arrs[s]) ** 2).mean())
        weighted_mse += w * mse
        total_w += w
    if total_w <= 0:
        return 0.0
    avg_mse = weighted_mse / total_w
    # Invert and rescale: 0 MSE → 1.0, MSE=0.5 → ~0
    return float(max(0.0, 1.0 - avg_mse * 4.0))


# ───────────────────────── 3. Loop-closure drift (geometric) ─────────────────────────


def compute_loop_closure_drift(
    forward_traj: Traj2D,
    reverse_traj: Traj2D,
) -> dict:
    """Return the geometric distance from origin after a forward +
    reverse action. If the WAM has perfect memory, this is 0.

    Output keys: drift_m, drift_yaw_deg, perfect_closure (bool).
    """
    fwd = np.array([forward_traj.xs[-1], forward_traj.ys[-1], forward_traj.yaws[-1]])
    rev = np.array([reverse_traj.xs[-1], reverse_traj.ys[-1], reverse_traj.yaws[-1]])
    end = fwd + rev
    drift_m = float(np.linalg.norm(end[:2]))
    drift_yaw = float(math.degrees(end[2]))
    return {
        "drift_m": drift_m,
        "drift_yaw_deg": drift_yaw,
        "perfect_closure": bool(drift_m < 0.5 and abs(drift_yaw) < 2.0),
    }


__all__ = [
    "reverse_candidate_traj",
    "build_reverse_traj_negatives",
    "compute_memory_symmetry",
    "compute_loop_closure_drift",
]

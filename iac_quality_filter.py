#!/usr/bin/env python3
"""Data quality filter for IAC anchor building.

iWorld-Bench §A.3 (Algorithm 1) runs a two-stage refinement on raw
videos before they enter the benchmark:

  1. Single-frame anomaly detection: Z-score the per-frame
     brightness-gradient magnitude M_t^light. Flag frames where the
     magnitude's deviation from the local-window median exceeds 4σ.

  2. Local-density filtering: compute a 1D box-convolution density
     ρ_t of an anomaly indicator I_t ∈ {0,1}. Frames with ρ_t < 0.06
     are 'high-fidelity zones'; everything else is suspect.

The paper then merges neighbouring high-fidelity zones with
gap-merge G_merge and discards segments shorter than L_min.

For IAC we don't process videos — we process `anchors` (4 history
frames + 4 future frames). The same idea applies:

  - Flag anchors where any frame's brightness gradient is anomalously
    high (likely scene cut / camera glitch / blur spike).
  - Flag anchors where the *inter-frame* brightness change is
    suspiciously low (stationary / frozen camera — useless signal
    for the consistency critic).
  - Drop anchors whose GT trajectory has implausibly long segments
    (likely interpolated missing data).

This is intentionally cheap — no model inference, only NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image


@dataclass
class FilterStats:
    total: int = 0
    brightness_outlier: int = 0
    stationary: int = 0
    bad_resolution: int = 0
    traj_interpolated: int = 0
    kept: int = 0

    def summary(self) -> dict:
        return {
            "total": self.total,
            "brightness_outlier": self.brightness_outlier,
            "stationary": self.stationary,
            "bad_resolution": self.bad_resolution,
            "traj_interpolated": self.traj_interpolated,
            "kept": self.kept,
            "dropped": self.total - self.kept,
        }


# ───────────────────────── single-frame brightness gradient ─────────────────────────


def _frame_light_gradient(path: str, size: int = 64) -> float:
    """Sum of |gx|+|gy| on a downsampled grayscale image.

    Cheaper than the full Tenengrad and good enough for anomaly
    detection. We use 64x64 to keep the filter fast on 100k+ anchors.
    """
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size, size))
        arr = np.asarray(im, dtype=np.float32)
    except Exception:
        return 0.0
    gx = np.abs(np.diff(arr, axis=1)).sum()
    gy = np.abs(np.diff(arr, axis=0)).sum()
    return float(gx + gy)


def compute_brightness_gradients(paths: Sequence[str], size: int = 64) -> np.ndarray:
    """Return per-frame gradient magnitudes (T,)."""
    return np.array([_frame_light_gradient(p, size) for p in paths], dtype=np.float32)


# ───────────────────────── Algorithm 1 (iWorld-Bench §A.3) ─────────────────────────


def detect_brightness_outliers(
    grad_mag: np.ndarray,
    window: int = 5,
    z_thresh: float = 4.0,
) -> np.ndarray:
    """Flag frames where the per-frame gradient magnitude deviates
    from the local-window median by > z_thresh × σ_res.

    Implements the first stage of iWorld-Bench Algorithm 1, Eq. (1).
    """
    T = len(grad_mag)
    if T < 3:
        return np.zeros(T, dtype=bool)
    is_outlier = np.zeros(T, dtype=bool)
    for t in range(T):
        lo = max(0, t - window // 2)
        hi = min(T, t + window // 2 + 1)
        win = grad_mag[lo:hi]
        med = float(np.median(win))
        res = grad_mag[t] - med
        # σ_res computed on residuals of the local window
        res_win = win - med
        sigma = float(np.sqrt((res_win ** 2).mean())) + 1e-6
        z = abs(res) / (3.0 * sigma)  # 3σ matches the iWorld paper heuristic
        if z > z_thresh:
            is_outlier[t] = True
    return is_outlier


def detect_stationary(
    grad_mag: np.ndarray,
    threshold: float = 5.0,
) -> np.ndarray:
    """Flag frames whose gradient is below `threshold` (likely frozen
    / black / camera stuck). 5.0 is empirical; gradient < 5 means
    the frame is essentially a single colour block."""
    return grad_mag < threshold


def compute_local_density(
    indicator: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """ρ_t = (1/W) Σ I_k, k∈[t-W/2, t+W/2]. Frames with ρ_t < 0.06
    are 'high-fidelity zones' (i.e. the local window is mostly
    non-flagged).

    Implements Algorithm 1 step 9.
    """
    T = len(indicator)
    half = window // 2
    rho = np.zeros(T, dtype=np.float32)
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        rho[t] = float(indicator[lo:hi].mean())
    return rho


def high_fidelity_mask(
    grad_mag: np.ndarray,
    brightness_window: int = 5,
    brightness_z: float = 4.0,
    stationary_thresh: float = 5.0,
    density_window: int = 5,
    density_thresh: float = 0.06,
) -> np.ndarray:
    """Combine the two stages into a single boolean mask: True means
    the frame passes the quality filter.
    """
    bright_out = detect_brightness_outliers(grad_mag, brightness_window, brightness_z)
    stat = detect_stationary(grad_mag, stationary_thresh)
    bad = bright_out | stat
    rho = compute_local_density(bad.astype(np.float32), density_window)
    # A frame is high-fidelity if it's not bad itself AND its
    # local-window density of bad frames is low.
    return (rho < density_thresh) & (~bad)


# ───────────────────────── Trajectory plausibility ─────────────────────────


def traj_has_interpolated_gaps(
    traj: Sequence[Sequence[float]],
    max_step_norm: float = 8.0,
    max_dyaw_deg: float = 25.0,
) -> bool:
    """Drop anchors where any single step's dx/dy norm or dyaw magnitude
    is implausibly large. nuPlan occasionally has missing samples that
    the loader interpolates, leaving one giant step that breaks the
    critic's understanding of physically plausible motion.
    """
    if len(traj) < 2:
        return True
    arr = np.array(traj, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    dxy = np.linalg.norm(arr[:, :2], axis=1)
    dyaw = np.abs(np.rad2deg(arr[:, 2]))
    return bool(dxy.max() > max_step_norm or dyaw.max() > max_dyaw_deg)


# ───────────────────────── End-to-end anchor filter ─────────────────────────


def filter_anchors(
    anchors: List[dict],
    history_key: str = "history_images",
    future_key: str = "future_images",
    traj_key: str = "candidate_traj",
    sample_stride_for_stats: int = 1,
    max_step_norm: float = 8.0,
    max_dyaw_deg: float = 25.0,
    image_root: Optional[Path] = None,
    stats: Optional[FilterStats] = None,
) -> List[dict]:
    """Apply the full filter pipeline to a list of anchor dicts.

    Returns the filtered list. Mutates `stats` in place.
    """
    if stats is None:
        stats = FilterStats()
    stats.total += len(anchors)
    out: List[dict] = []
    for a in anchors:
        # Resolve image paths
        hist_paths = _resolve_paths(a[history_key], image_root)
        fut_paths = _resolve_paths(a[future_key], image_root)
        all_paths = hist_paths + fut_paths

        # Cheap resolution check
        if not all(p.exists() for p in all_paths):
            stats.bad_resolution += 1
            continue

        # Brightness/stationary check
        grad = compute_brightness_gradients(all_paths)
        if not high_fidelity_mask(grad).all():
            n_bad = int((~high_fidelity_mask(grad)).sum())
            if n_bad > 1:  # tolerate one odd frame per anchor
                stats.brightness_outlier += 1
                if detect_stationary(grad).sum() > len(grad) // 2:
                    stats.stationary += 1
                continue

        # Trajectory plausibility
        if traj_has_interpolated_gaps(a[traj_key], max_step_norm, max_dyaw_deg):
            stats.traj_interpolated += 1
            continue

        out.append(a)
    stats.kept += len(out)
    return out


def _resolve_paths(values: Sequence[str], root: Optional[Path]) -> List[Path]:
    if not values:
        return []
    if root is None:
        return [Path(v) for v in values]
    out = []
    for v in values:
        p = Path(v)
        out.append(p if p.is_absolute() else root / p)
    return out


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Test the quality filter on a JSONL")
    p.add_argument("--input", required=True, help="consistency_*.jsonl path")
    p.add_argument("--image-root", default=None)
    p.add_argument("--max-samples", type=int, default=0)
    args = p.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if args.max_samples and len(rows) >= args.max_samples:
                break

    stats = FilterStats()
    kept = filter_anchors(rows, image_root=Path(args.image_root) if args.image_root else None, stats=stats)
    print(json.dumps(stats.summary(), indent=2))
    print(f"Kept {len(kept)}/{len(rows)} rows")

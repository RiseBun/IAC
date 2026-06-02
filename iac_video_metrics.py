#!/usr/bin/env python3
"""No-reference visual quality metrics for IAC.

Borrowed from iWorld-Bench (ICML 2026, Fang et al.) to provide orthogonal
cross-validation against the trained IAC critic. The critic can overfit to
nuPlan quirks; these metrics are formula-closed and do not learn from
training data.

All functions take a list of frame tensors or numpy arrays and return a
float in [0, 1] (higher is better).

Usage::

    from iac_video_metrics import compute_all_visual_metrics
    metrics = compute_all_visual_metrics(frames)  # list of (H, W, 3) uint8
    # metrics = {"image_quality": 0.62, "brightness": 0.91, "color": 0.87, "sharpness": 0.83}

Optional dependencies:
  - musiq (pip install musiq) for MUSIQ-based image quality. If missing,
    falls back to a BRISQUE-lite (variance-of-laplacian + gradient) proxy.
  - colour-science / cv2 (opencv-python) for HSV conversion; cv2 is in the
    nuPlan stack already.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


# ───────────────────────── helpers ─────────────────────────


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    """Accept float [0,1] or uint8, return uint8 HxWx3."""
    if frame.dtype != np.uint8:
        f = np.clip(frame, 0.0, 1.0) if frame.max() <= 1.0 else np.clip(frame, 0, 255)
        return (f * 255.0).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
    return frame


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        # ITU-R BT.601
        return (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]).astype(np.float32)
    return frame.astype(np.float32)


# ───────────────────────── 1. Image Quality (MUSIQ preferred, BRISQUE-lite fallback) ─────────────────────────


def compute_image_quality(frames: Sequence[np.ndarray]) -> float:
    """Per-frame image quality, then averaged.

    Primary: MUSIQ (Ke et al., 2021) — multi-scale image quality transformer.
    Fallback: BRISQUE-lite proxy combining Laplacian variance (sharpness
    surrogate) and a local-variance penalty (artificial-texture detector).
    """
    musiq_scores: List[float] = []
    try:
        import musiq  # type: ignore
        for f in frames:
            musiq_scores.append(float(musiq.score(_to_uint8(f))))
    except Exception:
        musiq_scores = []

    if musiq_scores:
        # MUSIQ returns ~[0, 100] (higher better); rescale to [0, 1] conservatively
        return float(np.clip(np.mean(musiq_scores) / 100.0, 0.0, 1.0))

    # Fallback proxy: combine Laplacian variance with mean local std.
    proxy = []
    for f in frames:
        g = _to_gray(f)
        # Laplacian via 4-neighbour difference
        lap = (
            np.abs(g[:-2, 1:-1] - g[2:, 1:-1]).mean()
            + np.abs(g[1:-1, :-2] - g[1:-1, 2:]).mean()
        ) * 0.5
        local_std = float(g.std())
        # Map to [0,1] with a soft sigmoid; tunings are heuristic.
        s = 1.0 - math.exp(-lap / 25.0)
        s *= math.tanh(local_std / 60.0)
        proxy.append(float(s))
    return float(np.mean(proxy)) if proxy else 0.0


# ───────────────────────── 2. Brightness Consistency (Fang et al., §3.3.1) ─────────────────────────


def compute_brightness_consistency(
    frames: Sequence[np.ndarray],
    alpha: float = -0.2,
) -> float:
    """Cosine similarity of per-frame brightness histograms, weighted by
    exponential decay (more distant frames penalised more).

    Args:
        frames: list of (H, W, 3) arrays
        alpha: decay rate (negative = decay, matching iWorld-Bench Eq. 5)
    """
    T = len(frames)
    if T < 2:
        return 0.0

    bins = 32
    vecs = []
    for f in frames:
        g = _to_gray(f).flatten()
        hist, _ = np.histogram(g, bins=bins, range=(0, 255), density=True)
        vecs.append(hist.astype(np.float32))
    V = np.stack(vecs, axis=0)  # (T, bins)

    # Cosine similarity matrix S_{Brightness}
    S = np.zeros((T, T), dtype=np.float32)
    for t in range(T):
        for d in range(1, T - t):
            a, b = V[t], V[t + d]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            S[t, t + d] = float(np.dot(a, b) / denom)
            S[t + d, t] = S[t, t + d]

    # Distance and weights
    dist = np.abs(np.subtract.outer(np.arange(T), np.arange(T))).astype(np.float32)
    w = np.exp(alpha * dist)  # alpha<0 → closer frames have higher weight
    upper = np.triu_indices(T, k=1)
    num = (w * S)[upper].sum()
    den = w[upper].sum() + 1e-8
    return float(num / den)


# ───────────────────────── 3. Color Temperature Constraint (Fang et al., §3.3.1) ─────────────────────────


def compute_color_temperature(
    frames: Sequence[np.ndarray],
    alpha: float = -0.2,
    beta: float = 1.5,
) -> float:
    """Hue similarity in HSV, with strict exponential decay (β>α) so
    distant frames penalise 'color drift' more aggressively.

    Lower score = more color drift = worse.
    """
    T = len(frames)
    if T < 2:
        return 0.0

    try:
        import cv2
        hsv_vecs = []
        for f in frames:
            hsv = cv2.cvtColor(_to_uint8(f), cv2.COLOR_RGB2HSV)
            # Hue: 0..179 in OpenCV. 7 bins × 25.7° per bin.
            h = hsv[..., 0].flatten()
            hist, _ = np.histogram(h, bins=7, range=(0, 180), density=True)
            hsv_vecs.append(hist.astype(np.float32))
    except Exception:
        return 0.0

    V = np.stack(hsv_vecs, axis=0)
    S = np.zeros((T, T), dtype=np.float32)
    for t in range(T):
        for d in range(1, T - t):
            a, b = V[t], V[t + d]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            S[t, t + d] = float(np.dot(a, b) / denom)
            S[t + d, t] = S[t, t + d]

    dist = np.abs(np.subtract.outer(np.arange(T), np.arange(T))).astype(np.float32)
    w_a = np.exp(alpha * dist)
    w_b = np.exp(beta * alpha * dist)  # β>1 means faster decay for far frames
    upper = np.triu_indices(T, k=1)
    num = (w_b * S)[upper].sum()
    den = w_a[upper].sum() + 1e-8
    return float(num / den)


# ───────────────────────── 4. Sharpness Retention (Fang et al., §3.3.1) ─────────────────────────


def compute_sharpness_retention(
    frames: Sequence[np.ndarray],
    n_t: int = 5,
    tau: float = 0.6,
) -> float:
    """Vectored Tenengrad + BRISQUE-style noise gate to distinguish true
    texture from high-frequency artifacts.

    Pipeline:
      1. Compute |Gx|+|Gy| per frame → 1D sharpness vector g_t
      2. BRISQUE noise score n_t (high-frequency noise proxy); flag frames
         where n_t > τ (treat as noise-induced, not real texture)
      3. Lower-convex log transform to compress high end
      4. Cosine similarity to first frame, weighted by decay
    """
    T = len(frames)
    if T < 2:
        return 0.0

    grads_h, grads_v, noise = [], [], []
    for f in frames:
        g = _to_gray(f)
        gh = np.abs(np.diff(g, axis=0)).sum()
        gv = np.abs(np.diff(g, axis=1)).sum()
        # BRISQUE-lite noise proxy: std of high-pass residual (1 - blur)
        try:
            from scipy.ndimage import uniform_filter
            smooth = uniform_filter(g, size=3)
            noise.append(float((g - smooth).std()))
        except Exception:
            noise.append(0.0)
        grads_h.append(gh)
        grads_v.append(gv)

    s = np.array(grads_h) + np.array(grads_v)  # (T,)
    n = np.array(noise)
    n_t = (n > tau).astype(np.float32)  # noise mask

    # Lower-convex log
    def L(x: np.ndarray) -> np.ndarray:
        return np.log(1.0 + x) / np.log(1.0 + np.maximum(s, 1e-3))

    M = np.where(
        (n_t == 0) & (s < tau),
        L(s),
        np.where(n_t > 0, 0.0, 0.5),  # noise or blur-belt → reduced score
    )

    # Cosine similarity to first frame, weighted by exponential decay
    base = M[0] + 1e-8
    sims = []
    for t in range(1, T):
        denom = (np.linalg.norm(M[0]) * np.linalg.norm(M[t])) + 1e-8
        sims.append(float(np.dot(M[0], M[t]) / denom))
    sims = np.array(sims)
    dist = np.arange(1, T).astype(np.float32)
    w = np.exp(-0.3 * dist)
    return float((w * sims).sum() / (w.sum() + 1e-8))


# ───────────────────────── aggregate ─────────────────────────


def compute_all_visual_metrics(
    frames: Sequence[np.ndarray],
    include_image_quality: bool = True,
) -> dict:
    """Run all four metrics and return a dict in [0, 1] (higher better)."""
    if not frames:
        return {"image_quality": 0.0, "brightness": 0.0, "color": 0.0, "sharpness": 0.0}
    out = {
        "brightness": compute_brightness_consistency(frames),
        "color": compute_color_temperature(frames),
        "sharpness": compute_sharpness_retention(frames),
    }
    if include_image_quality:
        out["image_quality"] = compute_image_quality(frames)
    else:
        out["image_quality"] = None  # type: ignore[assignment]
    return out


def load_frames_from_paths(paths: Sequence[str], size: int = 224) -> List[np.ndarray]:
    """Helper: load & resize a list of image paths to a uniform HxWx3 uint8 array."""
    from PIL import Image
    out = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB").resize((size, size))
        out.append(np.asarray(im, dtype=np.uint8))
    return out


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(description="Compute no-reference visual metrics on a video/sequence")
    p.add_argument("--frames", nargs="+", required=True, help="Image paths")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--no-iq", action="store_true", help="Skip MUSIQ-based image quality")
    args = p.parse_args()

    paths = [str(Path(x)) for x in args.frames]
    frames = load_frames_from_paths(paths, size=args.size)
    metrics = compute_all_visual_metrics(frames, include_image_quality=not args.no_iq)
    print(json.dumps(metrics, indent=2))
    sys.exit(0)

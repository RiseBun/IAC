"""Manifest loading, camera scaling and static frame matching for AS probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array or JSONL manifest."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("manifest JSON must contain a list")
    return value


def scaled_intrinsics(record: dict[str, Any], image: np.ndarray) -> np.ndarray:
    """Scale manifest intrinsics to the decoded image resolution."""
    matrix = np.asarray(record.get("intrinsics"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("a finite 3x3 intrinsics matrix is required")
    source_size = record.get("intrinsics_source_size")
    if not source_size or len(source_size) != 2:
        raise ValueError("intrinsics_source_size=[width,height] is required")
    source_width, source_height = map(int, source_size)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("intrinsics_source_size must be positive")
    target_height, target_width = image.shape[:2]
    scaled = matrix.copy()
    scaled[0, :] *= target_width / source_width
    scaled[1, :] *= target_height / source_height
    return scaled


def _resize(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    import cv2  # type: ignore

    height, width = image.shape[:2]
    if max_width <= 0 or width <= max_width:
        return image, 1.0
    scale = float(max_width) / float(width)
    return (
        cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        ),
        scale,
    )


def sift_matches(
    current: np.ndarray,
    nxt: np.ndarray,
    max_width: int,
    *,
    static_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ratio-tested SIFT matches in original-image coordinates."""
    import cv2  # type: ignore

    current_small, current_scale = _resize(current, max_width)
    next_small, next_scale = _resize(nxt, max_width)
    detector = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.02)
    key_current, desc_current = detector.detectAndCompute(
        cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY), None
    )
    key_next, desc_next = detector.detectAndCompute(
        cv2.cvtColor(next_small, cv2.COLOR_BGR2GRAY), None
    )
    if desc_current is None or desc_next is None:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty, {"available": False, "reason": "no_descriptors", "selected_matches": 0}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_current, desc_next, k=2)
    good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    source = np.asarray(
        [key_current[item.queryIdx].pt for item in good], dtype=np.float64
    ).reshape(-1, 2) / current_scale
    target = np.asarray(
        [key_next[item.trainIdx].pt for item in good], dtype=np.float64
    ).reshape(-1, 2) / next_scale
    if static_masks is not None and len(source):
        mask_current, mask_next = static_masks
        xy_current = np.rint(source).astype(np.int64)
        xy_next = np.rint(target).astype(np.int64)
        inside = (
            (xy_current[:, 0] >= 0)
            & (xy_current[:, 0] < mask_current.shape[1])
            & (xy_current[:, 1] >= 0)
            & (xy_current[:, 1] < mask_current.shape[0])
            & (xy_next[:, 0] >= 0)
            & (xy_next[:, 0] < mask_next.shape[1])
            & (xy_next[:, 1] >= 0)
            & (xy_next[:, 1] < mask_next.shape[0])
        )
        keep = np.zeros(len(source), dtype=bool)
        keep[inside] = (
            mask_current[xy_current[inside, 1], xy_current[inside, 0]]
            & mask_next[xy_next[inside, 1], xy_next[inside, 0]]
        )
        source, target = source[keep], target[keep]
    return source, target, {
        "available": True,
        "detector": "SIFT",
        "keypoints_current": len(key_current),
        "keypoints_next": len(key_next),
        "ratio_threshold": 0.72,
        "candidate_matches": len(good),
        "selected_matches": int(len(source)),
        "mask": "static_auxiliary" if static_masks is not None else "all",
        "resize_scale_current": current_scale,
        "resize_scale_next": next_scale,
    }

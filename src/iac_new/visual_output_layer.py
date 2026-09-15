"""Frozen, tolerance-aware output semantics for the IAC visual layer.

The layer intentionally exposes raw geometry for diagnostics but scores only
coarse observable quantities.  A single-view metric-depth estimate is not
treated as exact distance: acceptance uses a broad calibrated tolerance and
an ordinal bin check, with explicit abstention for weak geometric support.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


DEFAULT_CONFIG: dict[str, Any] = {
    "yaw": {"straight_threshold_rad": 0.0128},
    "longitudinal": {
        "distance_bin_edges_m": [0.5, 3.0, 6.0, 9.0],
        "distance_abs_tolerance_m": 3.0,
        "distance_relative_tolerance": 0.5,
        "adjacent_bin_tolerance": 1,
    },
    "geometry_quality": {
        "minimum_matches": 40,
        "minimum_inlier_fraction": 0.10,
        "high_confidence_inlier_fraction": 0.20,
        "maximum_median_reprojection_error_px": 3.0,
    },
}


def _section(config: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    result = dict(DEFAULT_CONFIG[name])
    if config is not None:
        result.update(dict(config.get(name, {})))
    return result


def classify_yaw(signed_yaw_rad: float, *, config: Mapping[str, Any] | None = None) -> str:
    """Return the frozen left/straight/right label."""
    value = float(signed_yaw_rad)
    if not np.isfinite(value):
        return "uncertain"
    threshold = float(_section(config, "yaw")["straight_threshold_rad"])
    if abs(value) <= threshold:
        return "straight"
    return "left" if value < 0.0 else "right"


def classify_distance_bin(distance_m: float, *, config: Mapping[str, Any] | None = None) -> str:
    """Map an observed forward displacement to a coarse, non-metric label."""
    value = float(distance_m)
    if not np.isfinite(value) or value < 0.0:
        return "uncertain"
    edges = np.asarray(_section(config, "longitudinal")["distance_bin_edges_m"], dtype=np.float64)
    if edges.ndim != 1 or len(edges) != 4 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("distance_bin_edges_m must contain four increasing values")
    labels = ("stop", "very_short", "short", "medium", "long")
    return labels[int(np.digitize(value, edges))]


def _bin_index(label: str) -> int | None:
    labels = ("stop", "very_short", "short", "medium", "long")
    return labels.index(label) if label in labels else None


def geometry_quality(
    *,
    matches: int | None,
    inlier_fraction: float | None,
    median_reprojection_error_px: float | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert geometric support into usable/high-confidence/uncertain."""
    section = _section(config, "geometry_quality")
    reasons: list[str] = []
    n = int(matches or 0)
    fraction = float(inlier_fraction) if inlier_fraction is not None else float("nan")
    reprojection = float(median_reprojection_error_px) if median_reprojection_error_px is not None else float("nan")
    if n < int(section["minimum_matches"]):
        reasons.append("insufficient_matches")
    if not np.isfinite(fraction) or fraction < float(section["minimum_inlier_fraction"]):
        reasons.append("weak_inlier_support")
    if not np.isfinite(reprojection) or reprojection > float(section["maximum_median_reprojection_error_px"]):
        reasons.append("high_reprojection_error")
    if reasons:
        status = "uncertain"
    elif fraction >= float(section["high_confidence_inlier_fraction"]):
        status = "high"
    else:
        status = "usable"
    return {
        "status": status,
        "confidence": 1.0 if status == "high" else (0.6 if status == "usable" else 0.0),
        "reasons": reasons,
        "matches": n,
        "inlier_fraction": fraction if np.isfinite(fraction) else None,
        "median_reprojection_error_px": reprojection if np.isfinite(reprojection) else None,
    }

def score_longitudinal(
    predicted_distance_m: float,
    reference_distance_m: float,
    *,
    interval_seconds: float = 1.0,
    geometry: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score coarse progress with explicit tolerance and abstention.

    ``predicted_distance_m`` remains an observed raw estimate.  It is never
    silently clipped or recalibrated.  The formal acceptance allows either a
    broad absolute/relative error tolerance or an adjacent coarse bin.
    """
    predicted = float(predicted_distance_m)
    reference = float(reference_distance_m)
    dt = float(interval_seconds)
    if not np.isfinite(predicted) or not np.isfinite(reference) or not np.isfinite(dt) or dt <= 0.0:
        return {"status": "uncertain", "acceptable": False, "abstention_reason": "nonfinite_input"}
    quality = geometry_quality(**geometry, config=config) if geometry is not None else {"status": "usable", "confidence": 0.6, "reasons": []}
    predicted_nonnegative = max(0.0, predicted)
    predicted_bin = classify_distance_bin(predicted_nonnegative, config=config)
    reference_bin = classify_distance_bin(max(0.0, reference), config=config)
    predicted_index = _bin_index(predicted_bin)
    reference_index = _bin_index(reference_bin)
    error = abs(predicted - reference)
    longitudinal = _section(config, "longitudinal")
    tolerance = max(
        float(longitudinal["distance_abs_tolerance_m"]),
        float(longitudinal["distance_relative_tolerance"]) * abs(reference),
    )
    within_tolerance = bool(error <= tolerance)
    adjacent = bool(
        predicted_index is not None
        and reference_index is not None
        and abs(predicted_index - reference_index) <= int(longitudinal["adjacent_bin_tolerance"])
    )
    acceptable = bool(quality["status"] != "uncertain" and (within_tolerance or adjacent))
    status = "scored" if acceptable else ("uncertain" if quality["status"] == "uncertain" else "outside_tolerance")
    return {
        "status": status,
        "acceptable": acceptable,
        "distance_bin": predicted_bin,
        "reference_distance_bin": reference_bin,
        "speed_bin": classify_distance_bin(predicted_nonnegative / dt, config=config),
        "reference_speed_bin": classify_distance_bin(max(0.0, reference) / dt, config=config),
        "distance_m_raw": predicted,
        "speed_mps_raw": predicted / dt,
        "absolute_error_m": error,
        "accepted_tolerance_m": tolerance,
        "within_tolerance": within_tolerance,
        "adjacent_bin_match": adjacent,
        "geometry_quality": quality,
        "confidence": float(quality["confidence"] if acceptable else 0.0),
        "abstention_reason": None if acceptable else (";".join(quality["reasons"]) if quality["reasons"] else "outside_tolerance"),
    }

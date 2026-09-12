"""First-principles visual evidence layer shared by MAS, RCS and GS.

Step1 is intentionally an evidence extractor, not a metric-trajectory
decoder.  It consumes candidate-blind flow-structure profiles and exposes
three independent evidence products:

* temporal motion evidence for a single branch;
* same-source response evidence for a counterfactual pair;
* generated-vs-reference grounding evidence.

Metric depth is an optional adapter.  It may improve geometric validity or
grounding diagnostics, but it cannot manufacture a missing visual response,
select pixels using the candidate action, or convert this layer into a
metre-valued trajectory estimator.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .visual_consistency import score_structural_grounding


EVIDENCE_CHANNELS = (
    "temporal_motion",
    "counterfactual_response",
    "grounding",
    "reliability",
)


def _finite(value: Any) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows = profile.get("rows")
    if not isinstance(rows, list):
        raise ValueError("profile must contain a rows list")
    return rows


def _signed_persistence(values: Iterable[float], *, deadband: float) -> float | None:
    signs = np.asarray([np.sign(v) for v in values if _finite(v) and abs(float(v)) > deadband], dtype=np.float64)
    if not len(signs):
        return None
    return float(max(np.mean(signs > 0.0), np.mean(signs < 0.0)))


def temporal_motion_evidence(
    profile: dict[str, Any],
    *,
    min_intervals: int = 2,
    direction_deadband: float = 1e-6,
    persistence_min: float = 0.5,
) -> dict[str, Any]:
    """Summarize branch motion without fitting a trajectory."""
    rows = _rows(profile)
    available = [row for row in rows if bool(row.get("input_available"))]
    centers = [row.get("horizontal_flow_center") for row in available]
    persistence = _signed_persistence(centers, deadband=direction_deadband)
    status = "unavailable"
    reason = "insufficient_temporal_evidence"
    if len(available) >= min_intervals and persistence is not None:
        status = "scored" if persistence >= persistence_min else "weak"
        reason = None if status == "scored" else "unstable_temporal_sign"
    return {
        "status": status,
        "reason": reason,
        "evidence_type": "ordinal_temporal_motion",
        "interval_count": len(rows),
        "available_intervals": len(available),
        "coverage": len(available) / max(len(rows), 1),
        "horizontal_sequence": [float(v) if _finite(v) else None for v in centers],
        "temporal_persistence": persistence,
        "median_motion_energy_px": (
            float(np.median([float(row["median_flow_magnitude_px"]) for row in available if _finite(row.get("median_flow_magnitude_px"))]))
            if any(_finite(row.get("median_flow_magnitude_px")) for row in available) else None
        ),
        "metric_reconstruction_used": False,
        "candidate_selection_used": False,
    }


def counterfactual_response_evidence(
    left_profile: dict[str, Any],
    right_profile: dict[str, Any],
    action_delta: float | Iterable[float],
    *,
    min_intervals: int = 2,
    action_deadband: float = 1e-6,
    response_deadband: float = 1e-6,
    persistence_min: float = 0.5,
) -> dict[str, Any]:
    """Compare same-source visual differences with an action difference.

    The action is used only after candidate-blind profiles have been
    extracted.  A zero action contrast is unavailable, never a zero score.
    """
    left_rows = {int(row.get("interval_index", i)): row for i, row in enumerate(_rows(left_profile))}
    right_rows = {int(row.get("interval_index", i)): row for i, row in enumerate(_rows(right_profile))}
    common = sorted(set(left_rows) & set(right_rows))
    visual_delta: list[float] = []
    intervals: list[int] = []
    for index in common:
        left, right = left_rows[index], right_rows[index]
        if not (bool(left.get("input_available")) and bool(right.get("input_available"))):
            continue
        lv, rv = left.get("horizontal_flow_center"), right.get("horizontal_flow_center")
        if _finite(lv) and _finite(rv):
            intervals.append(index)
            visual_delta.append(float(lv) - float(rv))
    action_values = np.asarray([action_delta] if np.isscalar(action_delta) else list(action_delta), dtype=np.float64)
    if not len(action_values) or not np.all(np.isfinite(action_values)):
        raise ValueError("action_delta must be finite")
    if len(action_values) == 1:
        action_for_intervals = np.repeat(action_values, len(visual_delta))
    elif len(action_values) == len(visual_delta):
        action_for_intervals = action_values
    else:
        raise ValueError("action_delta must be scalar or match common intervals")
    if not len(visual_delta) or float(np.max(np.abs(action_for_intervals))) <= action_deadband:
        return {
            "status": "unavailable",
            "reason": "zero_or_missing_action_contrast",
            "evidence_type": "same_source_signed_response",
            "common_intervals": len(visual_delta),
            "coverage": len(visual_delta) / max(len(common), 1),
            "metric_reconstruction_used": False,
            "candidate_selection_used": False,
        }
    signed = np.asarray(visual_delta) * np.sign(action_for_intervals)
    usable = np.abs(np.asarray(visual_delta)) > response_deadband
    hits = signed[usable] > 0.0
    persistence = float(np.mean(hits)) if len(hits) else None
    status = "unavailable"
    reason = "insufficient_counterfactual_intervals"
    if len(visual_delta) >= min_intervals and persistence is not None:
        status = "scored" if max(persistence, 1.0 - persistence) >= persistence_min else "weak"
        reason = None if status == "scored" else "unstable_response_sign"
    return {
        "status": status,
        "reason": reason,
        "evidence_type": "same_source_signed_response",
        "common_intervals": len(visual_delta),
        "coverage": len(visual_delta) / max(len(common), 1),
        "visual_delta_sequence": [float(v) for v in visual_delta],
        "action_delta_sequence": [float(v) for v in action_for_intervals],
        "direction_accuracy": float(np.mean(hits)) if len(hits) else None,
        "temporal_persistence": persistence,
        "median_abs_visual_delta": float(np.median(np.abs(visual_delta))) if visual_delta else None,
        "metric_reconstruction_used": False,
        "candidate_selection_used": False,
    }


def grounding_evidence(
    generated_profile: dict[str, Any],
    reference_profile: dict[str, Any],
    *,
    descriptor_scales: dict[str, float],
    min_common_intervals: int = 3,
) -> dict[str, Any]:
    """Compute descriptor-level generated/reference grounding evidence."""
    result = score_structural_grounding(
        _rows(generated_profile),
        _rows(reference_profile),
        descriptor_scales=descriptor_scales,
        min_common_intervals=min_common_intervals,
    )
    return {
        "status": "scored" if result.get("score") is not None else "unavailable",
        "evidence_type": "descriptor_grounding",
        "score": result.get("score"),
        "interval_scores": result.get("intervals", []),
        "metric_reconstruction_used": False,
        "candidate_selection_used": False,
    }


def reliability_evidence(
    *,
    flow_support_fraction: float | None = None,
    depth_valid_mask: np.ndarray | None = None,
    depth_confidence: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return auditable support diagnostics; depth is optional, never required."""
    result: dict[str, Any] = {
        "status": "scored" if flow_support_fraction is not None else "unavailable",
        "flow_support_fraction": None if flow_support_fraction is None else float(flow_support_fraction),
        "depth": {"provided": depth_valid_mask is not None, "required": False, "role": "optional_adapter"},
        "metric_reconstruction_used": False,
        "candidate_selection_used": False,
    }
    if depth_valid_mask is not None:
        mask = np.asarray(depth_valid_mask, dtype=bool)
        result["depth"]["valid_fraction"] = float(np.mean(mask))
        if depth_confidence is not None:
            confidence = np.asarray(depth_confidence, dtype=np.float64)
            if confidence.shape != mask.shape:
                raise ValueError("depth_confidence must match depth_valid_mask")
            finite = confidence[np.isfinite(confidence)]
            result["depth"]["median_confidence"] = float(np.median(finite)) if len(finite) else None
    return result


def assemble_step1_evidence(
    *,
    branch_profile: dict[str, Any],
    left_profile: dict[str, Any] | None = None,
    right_profile: dict[str, Any] | None = None,
    action_delta: float | Iterable[float] | None = None,
    reference_profile: dict[str, Any] | None = None,
    descriptor_scales: dict[str, float] | None = None,
    flow_support_fraction: float | None = None,
    depth_valid_mask: np.ndarray | None = None,
    depth_confidence: np.ndarray | None = None,
) -> dict[str, Any]:
    """Assemble the shared Step1 evidence object consumed by downstream metrics."""
    evidence: dict[str, Any] = {
        "protocol": "iac-step1-evidence-layer-v1",
        "status": "experimental_not_promoted",
        "channels": {
            "temporal_motion": temporal_motion_evidence(branch_profile),
            "counterfactual_response": {"status": "unavailable", "reason": "missing_twin"},
            "grounding": {"status": "unavailable", "reason": "missing_reference"},
            "reliability": reliability_evidence(flow_support_fraction=flow_support_fraction, depth_valid_mask=depth_valid_mask, depth_confidence=depth_confidence),
        },
        "metric_reconstruction_used": False,
        "candidate_selection_used": False,
        "depth_policy": "optional_adapter_not_required_for_MAS_RCS_GS",
    }
    if left_profile is not None and right_profile is not None and action_delta is not None:
        evidence["channels"]["counterfactual_response"] = counterfactual_response_evidence(left_profile, right_profile, action_delta)
    if reference_profile is not None:
        if descriptor_scales is None:
            raise ValueError("descriptor_scales are required for grounding evidence")
        evidence["channels"]["grounding"] = grounding_evidence(branch_profile, reference_profile, descriptor_scales=descriptor_scales)
    statuses = [channel.get("status") for channel in evidence["channels"].values()]
    evidence["available_channel_count"] = sum(status == "scored" for status in statuses)
    evidence["claim_boundary"] = "Evidence products support MAS/RCS/GS-specific scoring; they do not establish future-to-action causality or a full metric trajectory."
    return evidence

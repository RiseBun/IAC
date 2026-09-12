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

RESPONSE_DESCRIPTORS = {
    "yaw_direction": "horizontal_flow_center",
    "lateral_direction": "left_right_horizontal_contrast",
    "longitudinal_order": "median_flow_magnitude_px",
}


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
    motion_energy_floor_px: float | None = None,
    persistence_min: float = 0.5,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Summarize branch motion without fitting a trajectory.

    Two questions are kept separate, because conflating them is how a static
    scene with tiny nonzero flow reads as motion:

    ``motion_present``
        Is there more motion energy than the descriptor noise floor?  With four
        intervals a sign run alone cannot answer this: four coin flips land on
        one side 12.5% of the time, so a pure-noise branch would be "stable"
        roughly a quarter of the time.
    ``temporal_persistence``
        When motion is present, does it keep one direction?  This is reported
        but never used as evidence that motion exists.

    ``direction_deadband`` must come from independent repeated real-flow runs.
    A value near the numerical floor (the historical ``1e-6``) makes the
    persistence statistic meaningless, so the report states whether the floor
    was calibrated instead of implying that it was.
    """
    rows = _rows(profile)
    available = [row for row in rows if bool(row.get("input_available"))]
    centers = [row.get("horizontal_flow_center") for row in available]
    energies = [
        float(row["median_flow_magnitude_px"])
        for row in available
        if _finite(row.get("median_flow_magnitude_px"))
    ]
    median_energy = float(np.median(energies)) if energies else None
    persistence = _signed_persistence(centers, deadband=direction_deadband)
    evidence: dict[str, Any] = {
        "evidence_type": "ordinal_temporal_motion",
        "interval_count": len(rows),
        "available_intervals": len(available),
        "coverage": len(available) / max(len(rows), 1),
        "horizontal_sequence": [float(v) if _finite(v) else None for v in centers],
        "temporal_persistence": persistence,
        "median_motion_energy_px": median_energy,
        "direction_deadband": float(direction_deadband),
        "direction_deadband_calibrated": False,
        "motion_energy_floor_px": None if motion_energy_floor_px is None else float(motion_energy_floor_px),
        "metric_reconstruction_used": False,
        "candidate_selection_used": False,
    }
    if len(available) < min_intervals or persistence is None:
        return {**evidence, "status": "unavailable", "reason": "insufficient_temporal_evidence", "motion_present": None}
    if median_energy is None:
        return {**evidence, "status": "unavailable", "reason": "missing_motion_energy", "motion_present": None}
    if motion_energy_floor_px is not None and median_energy < float(motion_energy_floor_px):
        # Input exists and flow was computed, but there is no motion above the
        # independent noise floor.  This is the case the old scalar channel
        # silently reported as a direction.
        evidence["motion_energy_floor_px"] = float(motion_energy_floor_px)
        return {**evidence, "status": "unavailable", "reason": "no_motion_above_energy_floor", "motion_present": False}
    # A branch whose per-interval sign run is as short as chance allows carries no
    # temporal evidence at all; report it as weak rather than as stable motion.
    run = max(persistence, 1.0 - persistence)
    p_value = _binomial_tail_p(int(round(run * len(centers))), len(centers))
    evidence["chance_p_value"] = p_value
    evidence["motion_present"] = True
    if run >= persistence_min and (p_value or 1.0) <= alpha:
        return {**evidence, "status": "scored", "reason": None}
    return {**evidence, "status": "weak", "reason": "unstable_temporal_sign"}


def _binomial_tail_p(hits: int, trials: int) -> float | None:
    """Exact one-sided probability of at least ``hits`` successes at p=0.5.

    Intervals inside one source share texture, illumination and scene content, so
    this is not a substitute for a source-level control.  It is the floor a
    per-interval response must clear before its sign run is worth reporting.
    """
    if trials <= 0:
        return None
    from math import comb

    total = sum(comb(trials, k) for k in range(hits, trials + 1))
    return float(total / (2 ** trials))


def counterfactual_response_evidence(
    left_profile: dict[str, Any],
    right_profile: dict[str, Any],
    action_delta: float | Iterable[float],
    *,
    visual_descriptor: str = "horizontal_flow_center",
    min_intervals: int = 2,
    action_deadband: float = 1e-6,
    response_deadband: float = 1e-6,
    persistence_min: float = 0.5,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare same-source visual differences with an action difference.

    The action is used only after candidate-blind profiles have been extracted.
    A zero action contrast is unavailable, never a zero score.

    ``persistence_min`` is applied to the *action-aligned* direction only.  A
    model whose response is consistently inverted is reported as ``inverted``,
    not as a consistency of one: scoring |persistence - 0.5| would rate a
    perfectly anti-correlated model as maximally reliable, which is how an
    inverted shortcut would pass RCS.  ``alpha`` additionally requires the
    aligned run to beat the exact binomial chance level.
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
        lv, rv = left.get(visual_descriptor), right.get(visual_descriptor)
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
    base = {
        "evidence_type": "same_source_signed_response",
        "common_intervals": len(visual_delta),
        "coverage": len(visual_delta) / max(len(common), 1),
        "metric_reconstruction_used": False,
        "candidate_selection_used": False,
    }


    if not len(visual_delta) or float(np.max(np.abs(action_for_intervals))) <= action_deadband:
        return {**base, "status": "unavailable", "reason": "zero_or_missing_action_contrast"}

    # A response below the descriptor noise floor is not evidence of a response
    # at all.  The floor must come from independent repeated real-flow runs, not
    # from the confirmation set; until it is calibrated the caller passes the
    # pilot value and the report says so.
    magnitude = np.abs(np.asarray(visual_delta))
    usable = magnitude > response_deadband
    aligned = np.asarray(visual_delta)[usable] * np.sign(action_for_intervals)[usable]
    hits = int(np.sum(aligned > 0.0))
    misses = int(np.sum(aligned < 0.0))
    trials = hits + misses
    evidence = {
        **base,
        "visual_delta_sequence": [float(v) for v in visual_delta],
        "action_delta_sequence": [float(v) for v in action_for_intervals],
        "usable_intervals": int(trials),
        "below_response_deadband": int(np.sum(~usable)),
        "response_deadband": float(response_deadband),
        "response_deadband_calibrated": False,
        "direction_accuracy": (hits / trials) if trials else None,
        "direction_hits": hits,
        "direction_misses": misses,
        "chance_p_value": _binomial_tail_p(max(hits, misses), trials) if trials else None,
        "median_abs_visual_delta": float(np.median(magnitude)) if len(magnitude) else None,
    }
    if trials < min_intervals:
        return {**evidence, "status": "unavailable", "reason": "insufficient_counterfactual_intervals"}
    aligned_share = hits / trials
    if misses > hits and aligned_share == 0.0:
        return {
            **evidence,
            "status": "inverted",
            "reason": "response_consistently_opposite_to_action",
            "temporal_persistence": 0.0,
        }
    if aligned_share >= persistence_min and (evidence["chance_p_value"] or 1.0) <= alpha:
        return {**evidence, "status": "scored", "temporal_persistence": aligned_share}
    if max(aligned_share, 1.0 - aligned_share) >= persistence_min:
        return {
            **evidence,
            "status": "inverted" if aligned_share < 0.5 else "weak",
            "reason": "aligned_run_does_not_beat_chance"
            if aligned_share >= 0.5 else "response_consistently_opposite_to_action",
            "temporal_persistence": aligned_share,
        }
    return {
        **evidence,
        "status": "weak",
        "reason": "unstable_response_sign",
        "temporal_persistence": aligned_share,
    }


def response_channel_evidence(
    left_profile: dict[str, Any],
    right_profile: dict[str, Any],
    action_deltas: dict[str, float | Iterable[float]],
    *,
    min_intervals: int = 2,
) -> dict[str, dict[str, Any]]:
    """Evaluate declared ordinal response channels independently."""
    output: dict[str, dict[str, Any]] = {}
    for channel, descriptor in RESPONSE_DESCRIPTORS.items():
        if channel not in action_deltas:
            output[channel] = {
                "status": "unavailable",
                "reason": "missing_action_channel",
                "descriptor": descriptor,
            }
            continue
        result = counterfactual_response_evidence(
            left_profile,
            right_profile,
            action_deltas[channel],
            visual_descriptor=descriptor,
            min_intervals=min_intervals,
        )
        result["descriptor"] = descriptor
        result["action_channel"] = channel
        output[channel] = result
    return output


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


# The four outcomes the redesign calls for.  They are deliberately distinct
# because they license different conclusions: `no_motion` says the future is
# static, `inconsistent_with_action` says there is motion and it disagrees with
# the action, `insufficient_input` says nothing can be concluded, and
# `inverted_response` says the model is reliably anti-correlated.  Collapsing
# the last two into "low score" is how an inverted shortcut becomes invisible.
UNIT_STATUSES = (
    "scored",
    "no_motion",
    "inconsistent_with_action",
    "inverted_response",
    "insufficient_input",
)


def unit_status(channel: dict[str, Any]) -> str:
    """Map a channel verdict onto the shared four-way abstention vocabulary."""
    status = str(channel.get("status") or "unavailable")
    reason = str(channel.get("reason") or "")
    if status == "scored":
        return "scored"
    if status == "inverted":
        return "inverted_response"
    if reason in {"no_motion_above_energy_floor", "missing_motion_energy"} or channel.get("motion_present") is False:
        return "no_motion"
    if status == "weak" and reason in {"unstable_response_sign", "aligned_run_does_not_beat_chance", "unstable_temporal_sign"}:
        return "inconsistent_with_action" if channel.get("evidence_type") == "same_source_signed_response" else "insufficient_input"
    if status in {"unavailable", "weak"}:
        return "insufficient_input"
    return "insufficient_input"


def reliability_evidence(
    *,
    flow_support_fraction: float | None = None,
    depth_valid_mask: np.ndarray | None = None,
    depth_confidence: np.ndarray | None = None,
    channels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return auditable support diagnostics; depth is optional, never required.

    When ``channels`` is supplied, the reliability product also carries the
    four-way unit status per channel plus the abstention-reason tally, so a
    consumer can tell "could not read it" from "read it and it disagrees"
    without re-deriving the mapping.
    """
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
    if channels is not None:
        statuses = {name: unit_status(channel) for name, channel in channels.items() if isinstance(channel, dict)}
        result["unit_status"] = statuses
        tally: dict[str, int] = {name: 0 for name in UNIT_STATUSES}
        for value in statuses.values():
            tally[value] = tally.get(value, 0) + 1
        result["unit_status_counts"] = tally
        result["abstention_reasons"] = {
            name: channel.get("reason")
            for name, channel in channels.items()
            if isinstance(channel, dict) and channel.get("status") != "scored"
        }
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

"""Capability-stratified IAC scorecard.

Optional capabilities (MAS, RCS, GS, future-to-action mediation, and the legacy
CCFC/FAU/FCS aliases) are reported as ``unavailable`` when a model does not
expose the required interface.  ``missing`` is reserved for a claimed capability
whose evidence is incomplete; ``ineligible`` is reserved for hard protocol
violations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CAPABILITIES = (
    "native_action_conditioned",
    "externally_controlled_video",
    "video_only",
    "action_only",
)

CELLS = (
    "l1", "a2f", "f2a",
    # Canonical metric ids.
    "mas", "rcs", "gs", "future_to_action_mediation",
    # Legacy aliases retained for compatibility with older scorecards.
    "cfac", "ccfc", "fau_f", "fau_a", "fau", "fcs", "coverage",
)

CLAIMED = {
    "native_action_conditioned": ("l1", "a2f", "f2a", "cfac"),
    "externally_controlled_video": ("a2f",),
    "video_only": (),
    "action_only": (),
}

STATUSES = ("pass", "fail", "pilot", "unavailable", "ineligible", "missing")
OPTIONAL_CELLS = frozenset({
    "mas", "rcs", "gs", "future_to_action_mediation",
    "ccfc", "fau_f", "fau_a", "fau", "fcs", "coverage",
})
PROJECTION_GATED_CELLS = frozenset({
    "cfac", "ccfc", "fau_f", "fau_a", "fau",
})
CANONICAL_COVERAGE_CELLS = frozenset({"mas", "rcs", "gs", "future_to_action_mediation"})

# These boundaries are part of the output contract rather than documentation
# only.  Downstream consumers must be able to distinguish an observed visual
#/action relationship from a causal future-to-action claim without consulting
# a second file or guessing from the cell name.
CLAIM_BOUNDARIES = {
    "mas": {
        "evidence_type": "visual_action_consistency",
        "causal_status": "not_mediation",
        "scope": "directional_yaw_alignment_only",
    },
    "rcs": {
        "evidence_type": "counterfactual_visual_action_response",
        "causal_status": "not_mediation",
        "scope": "directional_yaw_response_only",
    },
    "gs": {
        "evidence_type": "external_future_grounding",
        "causal_status": "not_mediation",
        "scope": "generated_structure_vs_external_future",
    },
    "future_to_action_mediation": {
        "evidence_type": "future_pathway_intervention",
        "causal_status": "causal_only_if_promotion_passed",
        "scope": "future_to_action_pathway_dependence",
    },
    "fcs": {
        "evidence_type": "independent_execution",
        "causal_status": "not_mediation",
        "scope": "task_success_under_external_rollout",
    },
}

# The benchmark is conditional by design: a WAM is not required to expose a
# future-driven pathway in order to be evaluated on the channels it supports.
# This metadata is emitted with every scorecard so consumers cannot mistake an
# unavailable capability for a failed model or a zero score.
CONDITIONAL_SCORING_POLICY = {
    "mode": "conditional_evaluation",
    "future_driven_assumption": False,
    "score_each_supported_channel_independently": True,
    "unavailable_policy": "exclude_from_metric_denominator_and_report_reason",
    "zero_fill_unavailable": False,
    "score_denominator": "units_with_evidence_and_quality_gate_passed",
    "coverage_denominator": "all_declared_units_in_the_split",
    "abstention_is_not_failure": True,
    "unit_statuses": {
        "scored": "required evidence is present and the frozen quality gate passed; contributes to the conditional score",
        "abstain": "the channel was attempted but the unit did not meet the quality gate; excluded from the score and counted in coverage",
        "weak": "the channel was attempted but the unit did not meet the quality gate; equivalent to abstain at unit level",
        "unavailable": "the channel or required reference is not supplied or cannot be observed; no score is reported",
        "missing": "the submission claims the channel but its evidence is incomplete",
        "ineligible": "a hard submission contract was violated; the submission is not benchmark-eligible",
    },
    "required_report_fields": [
        "conditional_score",
        "score_coverage",
        "status_counts",
        "abstention_reasons",
        "confidence_interval",
    ],
    "aggregate_score": "not_defined",
}

RANKING_POLICY = {
    "natural_model_quality_ranking": "not_supported",
    "aggregate_across_capabilities": "not_defined",
    "allowed_comparison": "same_metric_source_disjoint_or_paired_with_uncertainty",
    "reason": "Directional consistency and grounding are capability evidence, not a calibrated total utility scale.",
}


def claimed_cells(capability: str) -> tuple[str, ...]:
    if capability not in CLAIMED:
        raise ValueError(f"unknown capability: {capability}")
    return CLAIMED[capability]


def empty_cell(status: str, *, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status}")
    row = {"status": status}
    if reason:
        row["reason"] = reason
    row.update(extra)
    return row


def summarize_conditional_units(
    units: list[dict[str, Any]],
    *,
    score_key: str = "score",
    scored_statuses: tuple[str, ...] = ("scored",),
) -> dict[str, Any]:
    """Summarize a metric without turning non-evaluable units into failures.

    ``units`` is deliberately metric-agnostic: MAS may use branches, RCS may
    use twins, and GS may use sources.  The caller must choose the status that
    means "quality gate passed" for that metric.  The returned score is
    conditional on those units only; coverage always uses all declared units.
    """
    if not isinstance(scored_statuses, tuple) or not scored_statuses:
        raise ValueError("scored_statuses must be a non-empty tuple")
    known = set(CONDITIONAL_SCORING_POLICY["unit_statuses"])
    counts = {status: 0 for status in known}
    values: list[float] = []
    for unit in units:
        status = str(unit.get("status") or "unavailable")
        if status not in known:
            raise ValueError(f"unknown conditional unit status: {status}")
        counts[status] += 1
        if status in scored_statuses and unit.get(score_key) is not None:
            value = float(unit[score_key])
            if not np.isfinite(value):
                raise ValueError(f"non-finite {score_key} on a scored unit")
            values.append(value)
    total = len(units)
    scored = sum(counts.get(status, 0) for status in scored_statuses)
    if scored != len(values):
        raise ValueError("every scored unit must provide exactly one finite score")
    abstention_reasons: dict[str, int] = {}
    for unit in units:
        status = str(unit.get("status") or "unavailable")
        if status not in {"abstain", "weak"}:
            continue
        reason = str(unit.get("reason") or "unspecified")
        abstention_reasons[reason] = abstention_reasons.get(reason, 0) + 1
    return {
        "conditional_score": float(np.mean(values)) if values else None,
        "score_coverage": float(scored / total) if total else None,
        "declared_units": total,
        "scored_units": scored,
        "status_counts": counts,
        "abstention_reasons": abstention_reasons,
        "score_scope": "scored_units_only",
        "coverage_scope": "all_declared_units",
        "zero_fill_unavailable": False,
    }


def validate_submission_row(
    row: dict[str, Any],
    *,
    public_ids: set[str],
    expected_future_count: int | None = None,
) -> list[str]:
    issues: list[str] = []
    sample_id = str(row.get("sample_id") or row.get("source_key") or "")
    if not sample_id:
        issues.append("missing_sample_id")
    elif sample_id not in public_ids:
        issues.append("sample_id_not_in_public_split")
    capability = str(row.get("capability") or "")
    if capability not in CLAIMED:
        issues.append("missing_or_unknown_capability")
        return issues
    if not str(row.get("wam_model_id") or ""):
        issues.append("missing_wam_model_id")
    claimed = claimed_cells(capability)
    images = row.get("future_images") or row.get("generated_future_images") or []
    times = np.asarray(row.get("future_times_s"), dtype=np.float64) if row.get("future_times_s") is not None else np.asarray([])
    needs_video = any(cell in claimed for cell in ("l1", "a2f", "f2a", "ccfc"))
    if needs_video:
        count = len(images) if isinstance(images, list) else 0
        if expected_future_count is not None:
            valid_count = count == expected_future_count
        else:
            valid_count = count >= 4
        if row.get("future_images_source") != "wam_generated":
            issues.append("future_images_source_is_not_wam_generated")
        if not valid_count:
            expected = str(expected_future_count) if expected_future_count is not None else "at_least_4"
            issues.append(f"future_images_must_have_{expected}_paths")
        if times.shape != (count,) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            issues.append("future_times_s_invalid")
        elif float(times[0]) <= 0.0 or not (3.95 <= float(times[-1]) <= 4.05):
            issues.append("future_times_s_does_not_cover_0p5_to_4p0_seconds")
    action = row.get("action_trajectory")
    if action is None:
        action = (row.get("action_condition") or {}).get("trajectory")
    action_array = np.asarray(action, dtype=np.float64) if action is not None else np.zeros((0, 3))
    if "l1" in claimed or "ccfc" in claimed or "f2a" in claimed:
        action_count = len(images) if isinstance(images, list) else expected_future_count or 0
        if action_array.shape != (action_count, 3) or not np.all(np.isfinite(action_array)):
            issues.append("native_action_trajectory_invalid")
        source = str(row.get("action_source") or "")
        if capability == "native_action_conditioned" and (
            not source or source in {"logged", "oracle", "proxy", "candidate"}
        ):
            issues.append("action_source_is_not_native")
        if capability == "externally_controlled_video" and source not in {"external_control", "injected_pose"}:
            issues.append("external_control_source_required")
    if row.get("realized_future_ego_state") is not None:
        issues.append("realized_future_state_leakage")
    return issues


def validate_submission(
    rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    public_ids = {
        str(row.get("sample_id") or row.get("source_key") or "")
        for row in public_rows
    }
    public_ids.discard("")
    issues = []
    sample_ids: dict[str, int] = {}
    model_ids: set[str] = set()
    capabilities: set[str] = set()
    for index, row in enumerate(rows):
        row_issues = validate_submission_row(row, public_ids=public_ids)
        sample_id = str(row.get("sample_id") or row.get("source_key") or "")
        if sample_id:
            if sample_id in sample_ids:
                row_issues.append(f"duplicate_sample_id:first_row_{sample_ids[sample_id]}")
            else:
                sample_ids[sample_id] = index
        model_id = str(row.get("wam_model_id") or "")
        capability = str(row.get("capability") or "")
        if model_id:
            model_ids.add(model_id)
        if capability:
            capabilities.add(capability)
        if row_issues:
            issues.append({
                "row": index,
                "sample_id": row.get("sample_id") or row.get("source_key"),
                "issues": row_issues,
            })
    if len(model_ids) > 1:
        issues.append({
            "row": -1,
            "sample_id": None,
            "issues": ["submission_mixes_multiple_wam_model_ids"],
            "model_ids": sorted(model_ids),
        })
    if len(capabilities) > 1:
        issues.append({
            "row": -1,
            "sample_id": None,
            "issues": ["submission_mixes_multiple_capabilities"],
            "capabilities": sorted(capabilities),
        })
    pair_ids = {}
    for row in rows:
        group = str(row.get("counterfactual_group_id") or "")
        if group:
            pair_ids.setdefault(group, set()).add(str(row.get("branch_mode") or row.get("branch_id") or ""))
    return {
        "protocol": "iac-wam-submission-audit",
        "rows": len(rows),
        "invalid_rows": len(issues),
        "ready": bool(rows) and not issues,
        "issues": issues,
        "counterfactual_groups": {
            group: sorted(modes) for group, modes in sorted(pair_ids.items())
        },
    }


def _cell_from_measurement(
    measurement: dict[str, Any] | None,
    claimed: bool,
    cell_name: str,
) -> dict[str, Any]:
    if measurement:
        status = str(measurement.get("status") or "missing")
        if status not in STATUSES:
            raise ValueError(f"unknown status: {status}")
        result = {"status": status, **{k: v for k, v in measurement.items() if k != "status"}}
        if cell_name in CANONICAL_COVERAGE_CELLS and status in {"pass", "fail", "pilot"}:
            # Canonical structural reports use a semantic coverage name while
            # legacy projection cells use post-projection n/total counts.
            if "coverage" not in result:
                for alias in ("pair_coverage", "branch_coverage", "source_coverage"):
                    if result.get(alias) is not None:
                        result["coverage"] = float(result[alias])
                        result["coverage_basis"] = alias
                        break
            if "coverage" not in result:
                return empty_cell("missing", reason="canonical_metric_coverage_required")
            coverage = float(result["coverage"])
            if not np.isfinite(coverage) or coverage < 0.0 or coverage > 1.0:
                raise ValueError(f"invalid canonical metric coverage for {cell_name}")
        if cell_name in PROJECTION_GATED_CELLS or cell_name == "coverage":
            if status not in {"pass", "fail", "pilot"}:
                return result
            if "n" not in result or "total" not in result:
                return empty_cell(
                    "missing",
                    reason=(
                        "post_projection_coverage_counts_required"
                        if cell_name in PROJECTION_GATED_CELLS
                        else "coverage_counts_required"
                    ),
                )
            evaluated = int(result["n"])
            total = int(result["total"])
            if total <= 0 or evaluated < 0 or evaluated > total:
                raise ValueError(f"invalid post-projection coverage counts for {cell_name}")
            result["coverage"] = evaluated / total
            result["coverage_basis"] = (
                "post_projection_abstention"
                if cell_name in PROJECTION_GATED_CELLS
                else "reported_n_over_total"
            )
        return result
    if not claimed:
        status = "unavailable" if cell_name in OPTIONAL_CELLS else "ineligible"
        return empty_cell(status, reason="capability_not_declared")
    return empty_cell("missing", reason="no_measurement")


def build_model_scorecard(
    *,
    model_id: str,
    capability: str,
    measurements: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    claimed = claimed_cells(capability)
    measurements = measurements or {}
    cells = {
        cell: _cell_from_measurement(measurements.get(cell), cell in claimed, cell)
        for cell in CELLS
    }
    cell_policy = {
        "score_scope": CONDITIONAL_SCORING_POLICY["score_denominator"],
        "coverage_scope": CONDITIONAL_SCORING_POLICY["coverage_denominator"],
        "abstention_is_not_failure": True,
        "zero_fill_unavailable": False,
    }
    for cell in cells.values():
        cell["conditional_evaluation"] = dict(cell_policy)
    for cell, boundary in CLAIM_BOUNDARIES.items():
        cells[cell].update(boundary)
    return {
        "scoring_policy": dict(CONDITIONAL_SCORING_POLICY),
        "ranking_policy": dict(RANKING_POLICY),
        "model_id": model_id,
        "capability": capability,
        "claimed_cells": list(claimed),
        "cells": cells,
    }

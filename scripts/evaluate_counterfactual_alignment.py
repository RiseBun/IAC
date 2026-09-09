#!/usr/bin/env python3
"""Evaluate paired risk/clear WAM branches in continuous motion space."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.continuous_motion import (
    compare_counterfactual_motion_deltas,
    compare_counterfactual_se2_consistency,
    image_motion_profile,
    trajectory_to_motion_profile,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _wrapped(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 3 or np.ptp(first) <= 0.0 or np.ptp(second) <= 0.0:
        return None
    return float(np.corrcoef(_rankdata(first), _rankdata(second))[0, 1])


def _bootstrap_spearman(
    first: np.ndarray, second: np.ndarray, *, draws: int = 4000, seed: int = 6102
) -> list[float] | None:
    if _spearman(first, second) is None:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = rng.integers(0, len(first), size=len(first))
        estimate = _spearman(first[indices], second[indices])
        if estimate is not None:
            values.append(estimate)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _wilson_interval(hits: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    probability = hits / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [float(center - half), float(center + half)]


def shape_gate(row: dict[str, Any], count: int) -> tuple[list[str], list[float], list[str]]:
    """Fail closed unless input geometry and output projection both support shape."""
    statuses, observability, reasons = [], [], []
    fit_status = str(
        row.get("motion_explanation_status")
        or row.get("geometry_fit_status")
        or "abstain"
    )
    if fit_status != "explained":
        return (
            ["abstain"] * count,
            [0.0] * count,
            [f"motion_not_explained:{fit_status}"] * count,
        )
    intervals = list(row.get("observability_by_future_interval") or [])
    for index in range(count):
        item = intervals[index] if index < len(intervals) else {}
        if item.get("projection_supported") is not True:
            statuses.append("abstain")
            observability.append(0.0)
            reasons.append("projection_support_failed")
            continue
        direction = bool(item.get("direction_observable"))
        if direction:
            statuses.append("usable")
            reasons.append("direct_flow_geometry")
        else:
            statuses.append("abstain")
            reasons.append("no_shape_support")
        observability.append(
            float(np.clip(item.get("composite_observability", 0.0), 0.0, 1.0))
        )
    return statuses, observability, reasons


def classify_counterfactual_claim(
    *, intervention_types: list[str], specificity_controls: list[str], structurally_ready: bool
) -> dict[str, Any]:
    """Separate structural, internal-foresight, and semantic-hazard claims."""
    command_only = intervention_types == ["navigation_command_onehot"]
    internal_future_only = intervention_types == ["internal_future_latent_permutation"]
    # The unified protocol admits command/left-right interventions. They are
    # formal CCFC at the command-conditioned scope, but not semantic hazard
    # claims. Specificity controls remain diagnostic-only.
    formal_foresight_ready = structurally_ready and not specificity_controls
    semantic_hazard_ready = formal_foresight_ready and not internal_future_only and not command_only
    scope = (
        "specificity_control" if specificity_controls else
        "command_conditioned_action_image_consistency" if command_only else
        "internal_foresight_mediation" if internal_future_only else
        "semantic_foresight_counterfactual_consistency"
    )
    return {
        "formal_foresight_ready": bool(formal_foresight_ready),
        "semantic_hazard_ready": bool(semantic_hazard_ready),
        "claim_scope": scope,
    }


def audit_pair(
    group_id: str,
    roles: dict[str, dict[str, Any]],
    *,
    role_a: str = "clear",
    role_b: str = "risk",
) -> list[str]:
    issues = []
    for field in ("history_fingerprint", "wam_model_id", "nuisance_seed"):
        values = [roles[role].get(field) for role in (role_a, role_b)]
        if any(value is None for value in values):
            issues.append(f"missing_{field}")
        elif values[0] != values[1]:
            issues.append(f"mismatched_{field}")
    for role, row in roles.items():
        if row.get("future_images_source") != "wam_generated":
            issues.append(f"{role}_future_not_wam_generated")
        source = str(row.get("action_trajectory_source") or "").lower()
        if not source:
            issues.append(f"{role}_missing_action_trajectory_source")
        elif any(token in source for token in ("logged", "oracle", "proxy", "candidate")):
            issues.append(f"{role}_non_native_action_source")
        if row.get("candidate_bank_used_by_decoder") is not False:
            issues.append(f"{role}_candidate_blind_audit_failed")
    clear_times = np.asarray(roles[role_a].get("future_times_s") or [], dtype=np.float64)
    risk_times = np.asarray(roles[role_b].get("future_times_s") or [], dtype=np.float64)
    if clear_times.shape != risk_times.shape or not np.allclose(clear_times, risk_times, atol=1e-6, rtol=0.0):
        issues.append("mismatched_future_timestamps")
    clear_action = np.asarray(roles[role_a].get("action_trajectory") or [], dtype=np.float64)
    risk_action = np.asarray(roles[role_b].get("action_trajectory") or [], dtype=np.float64)
    if clear_action.shape != risk_action.shape or clear_action.ndim != 2 or clear_action.shape[1:] != (3,):
        issues.append("invalid_or_mismatched_action_trajectories")
    elif float(np.max(np.abs(clear_action - risk_action))) <= 1e-4:
        issues.append("action_intervention_has_no_effect")
    if not group_id or group_id == "None":
        issues.append("missing_counterfactual_group_id")
    return sorted(set(issues))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-eight-frame-four-second", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--role-a", default="clear")
    parser.add_argument("--role-b", default="risk")
    parser.add_argument("--minimum-yaw-action-delta", type=float, default=0.01)
    args = parser.parse_args()

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.records):
        groups[str(row.get("counterfactual_group_id"))].append(row)
    reports = []
    for group_id, branches in sorted(groups.items()):
        roles = {str(row.get("branch_role")): row for row in branches}
        if set(roles) != {args.role_a, args.role_b}:
            raise ValueError(f"{group_id}: exactly one {args.role_a} and one {args.role_b} branch are required")
        issues = audit_pair(group_id, roles, role_a=args.role_a, role_b=args.role_b)
        if args.require_ready and issues:
            raise ValueError(f"{group_id}: counterfactual readiness failed: {issues}")
        profiles = {}
        for role, row in roles.items():
            times = list(row.get("future_times_s") or [])
            if args.require_eight_frame_four_second and (len(times) != 8 or abs(float(times[-1]) - 4.0) > 0.05):
                raise ValueError(f"{group_id}/{role}: expected 8 frames ending at 4.0 seconds")
            if row.get("candidate_bank_used_by_decoder") is not False:
                raise ValueError(f"{group_id}/{role}: candidate-blind audit failed")
            history = row.get("history_ego_state") or []
            initial_speed = float(history[-1][3]) if history and len(history[-1]) >= 4 else None
            decoder = dict(row.get("decoder") or {})
            statuses, observability, reasons = shape_gate(row, len(times))
            decoder["shape_status_by_interval"] = statuses
            decoder["shape_observability_by_interval"] = observability
            decoder["shape_fallback_reason_by_interval"] = reasons
            profiles[role] = (
                image_motion_profile(decoder, times, initial_speed_mps=initial_speed),
                trajectory_to_motion_profile(row.get("action_trajectory"), times, initial_speed_mps=initial_speed),
            )
        image_clear, action_clear = profiles[args.role_a]
        image_risk, action_risk = profiles[args.role_b]
        fit_statuses = {
            role: str(
                roles[role].get("motion_explanation_status")
                or roles[role].get("geometry_fit_status")
                or "abstain"
            )
            for role in (args.role_a, args.role_b)
        }
        both_explained = all(value == "explained" for value in fit_statuses.values())
        common_projection_intervals = int(sum(
            clear_item.get("projection_supported") is True
            and risk_item.get("projection_supported") is True
            for clear_item, risk_item in zip(
                roles[args.role_a].get("observability_by_future_interval") or [],
                roles[args.role_b].get("observability_by_future_interval") or [],
            )
        ))
        direction_supported_intervals = int(sum(
            clear_row.get("shape_status") in {"usable", "uncertain"}
            and risk_row.get("shape_status") in {"usable", "uncertain"}
            for clear_row, risk_row in zip(image_clear["rows"], image_risk["rows"])
        ))
        reports.append({
            "counterfactual_group_id": group_id,
            "causal_claim_eligible": bool(
                not issues
                and both_explained
                and common_projection_intervals == len(times)
            ),
            "readiness_issues": issues,
            "branch_motion_explanation_status": fit_statuses,
            "pair_measurement_status": (
                "explained" if both_explained else
                "weak" if all(value in {"explained", "weak"} for value in fit_statuses.values()) else
                "abstain"
            ),
            "total_intervals": len(times),
            "projection_supported_intervals": common_projection_intervals,
            "direction_supported_intervals": direction_supported_intervals,
            "primary_yaw_response": {
                "image_delta_rad": _wrapped(
                    float(image_risk["rows"][-1]["heading_rad"])
                    - float(image_clear["rows"][-1]["heading_rad"])
                ),
                "action_delta_rad": _wrapped(
                    float(action_risk["rows"][-1]["heading_rad"])
                    - float(action_clear["rows"][-1]["heading_rad"])
                ),
                "minimum_material_action_delta_rad": float(args.minimum_yaw_action_delta),
            },
            "comparison": compare_counterfactual_motion_deltas(
                image_clear, image_risk, action_clear, action_risk
            ),
            "continuous_cfc": {
                "metric": compare_counterfactual_se2_consistency(
                    image_clear, image_risk, action_clear, action_risk, scale_mode="metric"
                ),
                "scale_free": compare_counterfactual_se2_consistency(
                    image_clear, image_risk, action_clear, action_risk, scale_mode="scale_free"
                ),
                "arc_relative": compare_counterfactual_se2_consistency(
                    image_clear, image_risk, action_clear, action_risk, scale_mode="arc_relative"
                ),
            },
        })
    eligible = [row for row in reports if row["causal_claim_eligible"]]
    primary_yaw_rows = [
        row for row in eligible
        if abs(float(row["primary_yaw_response"]["action_delta_rad"]))
        >= float(args.minimum_yaw_action_delta)
    ]
    yaw_image = np.asarray([
        row["primary_yaw_response"]["image_delta_rad"] for row in primary_yaw_rows
    ], dtype=np.float64)
    yaw_action = np.asarray([
        row["primary_yaw_response"]["action_delta_rad"] for row in primary_yaw_rows
    ], dtype=np.float64)
    yaw_hits = int(np.sum(np.sign(yaw_image) == np.sign(yaw_action)))
    yaw_rho = _spearman(yaw_image, yaw_action)
    metric_scores = [
        row["continuous_cfc"]["metric"]["score"]
        for row in eligible
        if row["continuous_cfc"]["metric"].get("score") is not None
    ]
    scale_free_scores = [
        row["continuous_cfc"]["scale_free"]["score"]
        for row in eligible
        if row["continuous_cfc"]["scale_free"].get("score") is not None
    ]
    arc_relative_scores = [
        row["continuous_cfc"]["arc_relative"]["score"]
        for row in eligible
        if row["continuous_cfc"]["arc_relative"].get("score") is not None
    ]
    intervention_types = sorted({
        str(row.get("intervention_type"))
        for branches in groups.values()
        for row in branches
        if row.get("intervention_type") is not None
    })
    specificity_controls = sorted({
        str(row.get("specificity_control"))
        for branches in groups.values()
        for row in branches
        if row.get("specificity_control") is not None
    })
    # Honest abstention is a measurement outcome, not a structural protocol
    # failure.  Structural readiness covers lineage and intervention audits;
    # pair-level measurement coverage is reported separately.
    structurally_eligible = bool(reports) and all(
        not row["readiness_issues"] for row in reports
    )
    claim = classify_counterfactual_claim(
        intervention_types=intervention_types,
        specificity_controls=specificity_controls,
        structurally_ready=structurally_eligible,
    )
    formal_foresight_eligible = claim["formal_foresight_ready"]
    semantic_hazard_eligible = claim["semantic_hazard_ready"]
    for report in reports:
        report["structural_pair_eligible"] = not report["readiness_issues"]
        report["formal_foresight_claim_eligible"] = bool(
            report["causal_claim_eligible"] and formal_foresight_eligible
        )
        report["semantic_hazard_claim_eligible"] = bool(
            report["causal_claim_eligible"] and semantic_hazard_eligible
        )
        report["causal_claim_eligible"] = report["formal_foresight_claim_eligible"]
    output = {
        "protocol": "counterfactual-continuous-alignment-report",
        "primary_metric": "explained_pair_terminal_yaw_response",
        "pair_roles": [args.role_a, args.role_b],
        "groups": len(reports),
        "causal_claim_eligible": formal_foresight_eligible,
        "structural_pair_eligible": structurally_eligible,
        "formal_foresight_claim_eligible": formal_foresight_eligible,
        "semantic_hazard_claim_eligible": semantic_hazard_eligible,
        "claim_scope": claim["claim_scope"],
        "intervention_types": intervention_types,
        "specificity_controls": specificity_controls,
        "summary": {
            "eligible_groups": len(eligible),
            "pair_measurement_coverage": len(eligible) / len(reports) if reports else 0.0,
            "pair_measurement_coverage_basis": "both_branches_explained_and_post_projection",
            "primary_yaw": {
                "field": "yaw_rate_radps",
                "terminal_response_unit": "rad",
                "material_pairs": len(primary_yaw_rows),
                "direction_hits": yaw_hits,
                "direction_accuracy": yaw_hits / len(primary_yaw_rows) if primary_yaw_rows else None,
                "direction_accuracy_ci95": _wilson_interval(yaw_hits, len(primary_yaw_rows)),
                "spearman": yaw_rho,
                "spearman_ci95": _bootstrap_spearman(yaw_image, yaw_action),
            },
            "metric_score_mean": None if not metric_scores else float(np.mean(metric_scores)),
            "scale_free_score_mean": None if not scale_free_scores else float(np.mean(scale_free_scores)),
            "metric_score_count": len(metric_scores),
            "scale_free_score_count": len(scale_free_scores),
            "arc_relative_score_mean": None if not arc_relative_scores else float(np.mean(arc_relative_scores)),
            "arc_relative_score_count": len(arc_relative_scores),
            "arc_relative_coverage": len(arc_relative_scores) / len(reports) if reports else 0.0,
            "coverage_basis": "both_branches_explained_and_post_projection",
            "projection_eligible_groups": sum(
                row["projection_supported_intervals"] == row["total_intervals"]
                for row in reports
            ),
            "projection_supported_interval_pairs": sum(
                row["projection_supported_intervals"] for row in reports
            ),
            "projection_interval_pair_coverage": (
                sum(row["projection_supported_intervals"] for row in reports)
                / sum(row["total_intervals"] for row in reports)
                if reports and sum(row["total_intervals"] for row in reports) > 0
                else None
            ),
            "direction_supported_interval_pairs": sum(
                row["direction_supported_intervals"] for row in reports
            ),
            "promotion_criteria": {
                "pair_measurement_coverage_min": 0.90,
                "yaw_direction_accuracy_ci95_lower_min": 0.75,
                "minimum_distinguishable_models": 2,
            },
        },
        "reports": reports,
    }
    primary_yaw = output["summary"]["primary_yaw"]
    yaw_ci = primary_yaw.get("direction_accuracy_ci95")
    model_count = len({
        str(row.get("wam_model_id"))
        for branches in groups.values()
        for row in branches
        if row.get("wam_model_id") is not None
    })
    output["summary"]["evaluated_wam_models"] = model_count
    output["summary"]["promotion_ready"] = bool(
        output["summary"]["pair_measurement_coverage"] >= 0.90
        and yaw_ci is not None
        and yaw_ci[0] >= 0.75
        and model_count >= 2
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()

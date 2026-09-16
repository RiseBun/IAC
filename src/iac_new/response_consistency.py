"""Same-source intervention response consistency for the frozen AS reader."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    index = 0
    while index < len(array):
        end = index + 1
        while end < len(array) and array[order[end]] == array[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x, y = _average_ranks(left), _average_ranks(right)
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else None


def _action_magnitude_bin(value: float, threshold: float) -> str:
    magnitude = abs(float(value))
    bounds = [float(threshold), 0.03, 0.06, 0.12]
    if magnitude < bounds[0]:
        return "zero"
    if magnitude < bounds[1]:
        return "material_0.0128_0.03"
    if magnitude < bounds[2]:
        return "material_0.03_0.06"
    if magnitude < bounds[3]:
        return "material_0.06_0.12"
    return "material_ge_0.12"


def _future_only_visual_yaw(row: Mapping[str, Any]) -> float | None:
    values = list(row.get("visual_yaw_intervals_rad") or [])
    if len(values) != 4 or any(value is None or not np.isfinite(float(value)) for value in values):
        return None
    return float(np.sum([float(value) for value in values[1:]]))


def delta_state(value: float | None, threshold: float) -> str | None:
    """Quantize a signed response into negative/zero/positive."""
    if value is None or not np.isfinite(float(value)):
        return None
    number = float(value)
    if abs(number) < float(threshold):
        return "zero"
    return "positive" if number > 0.0 else "negative"


def _pair_issues(left: Mapping[str, Any], right: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    intervention = config["intervention"]
    if left.get("counterfactual_group_id") != right.get("counterfactual_group_id"):
        issues.append("counterfactual_group_mismatch")
    if left.get("history_fingerprint") != right.get("history_fingerprint"):
        issues.append("history_mismatch")
    if left.get("nuisance_seed") != right.get("nuisance_seed"):
        issues.append("nuisance_seed_mismatch")
    allowed_types = set(intervention.get("allowed_types") or [intervention.get("type")])
    if (
        left.get("intervention_type") != right.get("intervention_type")
        or left.get("intervention_type") not in allowed_types
    ):
        issues.append("intervention_type_mismatch")
    if left.get("wam_model_id") != right.get("wam_model_id"):
        issues.append("model_id_mismatch")
    allowed_sources = set(intervention.get("allowed_future_images_sources") or [])
    if allowed_sources and (
        left.get("future_images_source") not in allowed_sources
        or right.get("future_images_source") not in allowed_sources
    ):
        issues.append("future_images_source_invalid")
    if not intervention.get("validation_only_derivatives_allowed", False):
        left_degradation = (left.get("metadata") or {}).get("controlled_degradation") or {}
        right_degradation = (right.get("metadata") or {}).get("controlled_degradation") or {}
        if left_degradation.get("validation_only") or right_degradation.get("validation_only"):
            issues.append("validation_only_derivative")
    if intervention.get("same_history_required") and (
        (left.get("lineage") or {}).get("same_history_seed") is False
        or (right.get("lineage") or {}).get("same_history_seed") is False
    ):
        issues.append("lineage_same_history_failed")
    for role, row in (("left", left), ("right", right)):
        runner_branch = (row.get("lineage") or {}).get("runner_branch")
        if runner_branch is not None and runner_branch != role:
            issues.append(f"{role}_runner_branch_mismatch")
    left_future = list(left.get("future_frame_paths") or left.get("future_images") or [])
    right_future = list(right.get("future_frame_paths") or right.get("future_images") or [])
    if not left_future or not right_future:
        issues.append("future_images_missing")
    elif left_future == right_future:
        issues.append("branches_not_regenerated")
    return issues


def _bootstrap_interval(values: Sequence[bool], *, draws: int, seed: int) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def _progress_diagnostic(left: Mapping[str, Any], right: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    progress_config = config["progress"]
    left_intervals = list(left.get("intervals") or [])
    right_intervals = list(right.get("intervals") or [])
    action_left = list(left.get("progress", {}).get("action_interval_distance_m") or [])
    action_right = list(right.get("progress", {}).get("action_interval_distance_m") or [])
    if not (len(left_intervals) == len(right_intervals) == len(action_left) == len(action_right) == 4):
        return {"status": "unavailable", "reason": "interval_contract_invalid"}
    common = [
        index
        for index, (first, second) in enumerate(zip(left_intervals, right_intervals))
        if first.get("status") != "uncertain" and second.get("status") != "uncertain"
    ]
    if len(common) < int(progress_config["required_common_intervals"]):
        return {
            "status": "unavailable",
            "reason": "insufficient_common_quality_intervals",
            "common_intervals": len(common),
        }
    action_delta = float(np.sum(action_left) - np.sum(action_right))
    visual_delta = float(
        np.sum(
            [
                left_intervals[index]["distance_m_raw"] - right_intervals[index]["distance_m_raw"]
                for index in common
            ]
        )
    )
    action_state = delta_state(action_delta, float(progress_config["action_delta_threshold_m"]))
    visual_state = delta_state(visual_delta, float(progress_config["visual_delta_threshold_m"]))
    return {
        "status": "scored",
        "action_delta_m": action_delta,
        "visual_delta_m": visual_delta,
        "action_state": action_state,
        "visual_state": visual_state,
        "match": action_state == visual_state,
        "common_intervals": len(common),
    }


def score_response_consistency(
    manifest_rows: Sequence[Mapping[str, Any]],
    as_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Score left/right response pairs using independently extracted AS motion."""
    as_by_id = {str(row.get("sample_id")): row for row in as_rows}
    if len(as_by_id) != len(as_rows):
        raise ValueError("AS rows contain duplicate sample_id values")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in manifest_rows:
        group = str(row.get("counterfactual_group_id") or "")
        role = str(row.get("branch_role") or "")
        if not group or not role:
            raise ValueError("manifest row lacks counterfactual_group_id or branch_role")
        if role in grouped[group]:
            raise ValueError(f"duplicate role {role!r} in group {group!r}")
        grouped[group][role] = row

    required_roles = set(config["intervention"]["required_roles"])
    yaw_config = config["yaw"]
    pair_rows: list[dict[str, Any]] = []
    invalid_reasons: Counter[str] = Counter()
    for group, roles in sorted(grouped.items()):
        if set(roles) != required_roles:
            invalid_reasons["required_roles_missing"] += 1
            pair_rows.append({"counterfactual_group_id": group, "status": "invalid_contract", "issues": ["required_roles_missing"]})
            continue
        left_meta, right_meta = roles["left"], roles["right"]
        issues = _pair_issues(left_meta, right_meta, config)
        if issues:
            invalid_reasons.update(issues)
            pair_rows.append({"counterfactual_group_id": group, "status": "invalid_contract", "issues": issues})
            continue
        left = as_by_id.get(str(left_meta.get("sample_id")))
        right = as_by_id.get(str(right_meta.get("sample_id")))
        if left is None or right is None:
            pair_rows.append({"counterfactual_group_id": group, "status": "missing_visual_result", "yaw_match": False})
            continue
        action_delta = float(left["yaw"]["action_yaw_rad"] - right["yaw"]["action_yaw_rad"])
        left_visual = left["yaw"].get("visual_yaw_rad")
        right_visual = right["yaw"].get("visual_yaw_rad")
        visual_delta = None if left_visual is None or right_visual is None else float(left_visual - right_visual)
        left_future_only = _future_only_visual_yaw(left)
        right_future_only = _future_only_visual_yaw(right)
        future_only_delta = (
            None
            if left_future_only is None or right_future_only is None
            else float(left_future_only - right_future_only)
        )
        left_boundary = (left.get("history_future_boundary") or {}).get("visual_yaw_rad")
        right_boundary = (right.get("history_future_boundary") or {}).get("visual_yaw_rad")
        boundary_delta = (
            None
            if left_boundary is None or right_boundary is None
            else float(left_boundary - right_boundary)
        )
        action_state = delta_state(action_delta, float(yaw_config["action_delta_threshold_rad"]))
        visual_state = delta_state(visual_delta, float(yaw_config["visual_delta_threshold_rad"]))
        pair_rows.append(
            {
                "counterfactual_group_id": group,
                "status": "scored" if visual_state is not None else "missing_visual_result",
                "action_yaw_delta_rad": action_delta,
                "visual_yaw_delta_rad": visual_delta,
                "future_only_visual_yaw_delta_rad": future_only_delta,
                "history_future_boundary_visual_yaw_delta_rad": boundary_delta,
                "action_yaw_state": action_state,
                "visual_yaw_state": visual_state,
                "yaw_match": visual_state is not None and action_state == visual_state,
                "material_action_response": action_state != "zero",
                "visual_response_detected": visual_state not in (None, "zero"),
                "progress_diagnostic": _progress_diagnostic(left, right, config),
            }
        )

    contract_valid = [row for row in pair_rows if row["status"] != "invalid_contract"]
    material = [row for row in contract_valid if row.get("material_action_response")]
    zero = [row for row in contract_valid if row.get("material_action_response") is False]
    detected_material = [row for row in material if row.get("visual_response_detected")]
    progress = [
        row["progress_diagnostic"]
        for row in contract_valid
        if row.get("progress_diagnostic", {}).get("status") == "scored"
    ]
    # A declared intervention only earns formal credit when it creates a
    # material action response and the visual response is non-zero and aligned.
    # Zero/zero agreement is useful as a specificity diagnostic, but awarding it
    # formal credit would let a model that ignores every intervention score 1.0.
    formal_hits = [
        bool(row.get("material_action_response") and row.get("yaw_match"))
        for row in contract_valid
    ]
    ternary_hits = [bool(row.get("yaw_match")) for row in contract_valid]
    material_hits = [bool(row.get("yaw_match")) for row in material]
    zero_hits = [bool(row.get("yaw_match")) for row in zero]
    stats = config["statistics"]
    swapped_material = [
        delta_state(
            -float(row["visual_yaw_delta_rad"]),
            float(yaw_config["visual_delta_threshold_rad"]),
        )
        == row["action_yaw_state"]
        for row in material
        if row.get("visual_yaw_delta_rad") is not None
    ]
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in contract_valid:
        action_state = str(row.get("action_yaw_state"))
        visual_state = str(row.get("visual_yaw_state"))
        confusion[action_state][visual_state] += 1
    material_strength: dict[str, dict[str, Any]] = {}
    for label in ("material_0.0128_0.03", "material_0.03_0.06", "material_0.06_0.12", "material_ge_0.12"):
        subset = [row for row in material if _action_magnitude_bin(row["action_yaw_delta_rad"], float(yaw_config["action_delta_threshold_rad"])) == label]
        detected = [row for row in subset if row.get("visual_response_detected")]
        material_strength[label] = {
            "pairs": len(subset),
            "formal_hits": sum(bool(row.get("yaw_match")) for row in subset),
            "formal_rate": float(np.mean([bool(row.get("yaw_match")) for row in subset])) if subset else None,
            "visual_response_rate": float(np.mean([bool(row.get("visual_response_detected")) for row in subset])) if subset else None,
            "conditional_direction": float(np.mean([bool(row.get("yaw_match")) for row in detected])) if detected else None,
        }
    material_with_visual = [row for row in material if row.get("visual_yaw_delta_rad") is not None]
    magnitude_spearman = _spearman(
        [abs(float(row["action_yaw_delta_rad"])) for row in material_with_visual],
        [abs(float(row["visual_yaw_delta_rad"])) for row in material_with_visual],
    )
    boundary_deltas = [
        abs(float(row["history_future_boundary_visual_yaw_delta_rad"]))
        for row in contract_valid
        if row.get("history_future_boundary_visual_yaw_delta_rad") is not None
    ]
    future_only_available = [
        row for row in contract_valid if row.get("future_only_visual_yaw_delta_rad") is not None
    ]
    future_only_hits = [
        bool(
            row.get("material_action_response")
            and delta_state(
                row["future_only_visual_yaw_delta_rad"],
                float(yaw_config["visual_delta_threshold_rad"]),
            )
            == row.get("action_yaw_state")
        )
        for row in future_only_available
    ]
    eligibility_reasons: list[str] = []
    if invalid_reasons:
        eligibility_reasons.append("pair_contract_violation")
    minimum_pairs = int(stats.get("minimum_declared_pairs_for_formal_report", 1))
    if len(contract_valid) < minimum_pairs:
        eligibility_reasons.append("insufficient_declared_pairs")
    if not contract_valid:
        eligibility_reasons.append("no_contract_valid_pairs")
    return {
        "protocol": config["protocol"],
        "metric_definition": "100 * directionally_correct_nonzero_response_pairs / contract_valid_declared_pairs",
        "yaw_thresholds_rad": {
            "action_delta": float(yaw_config["action_delta_threshold_rad"]),
            "visual_delta": float(yaw_config["visual_delta_threshold_rad"]),
        },
        "wam_model_ids": sorted(
            {str(row.get("wam_model_id")) for row in manifest_rows if row.get("wam_model_id")}
        ),
        "declared_pairs": len(grouped),
        "contract_valid_pairs": len(contract_valid),
        "contract_coverage": len(contract_valid) / len(grouped) if grouped else 0.0,
        "invalid_contract_reasons": dict(invalid_reasons),
        "RCS_yaw": float(np.mean(formal_hits)) if formal_hits else None,
        "RCS_yaw_100": 100.0 * float(np.mean(formal_hits)) if formal_hits else None,
        "RCS_yaw_bootstrap_95ci": _bootstrap_interval(
            formal_hits, draws=int(stats["bootstrap_draws"]), seed=int(stats["bootstrap_seed"])
        ),
        "ternary_state_agreement_diagnostic": float(np.mean(ternary_hits)) if ternary_hits else None,
        "material_action_pairs": len(material),
        "material_action_pair_coverage": len(material) / len(contract_valid) if contract_valid else 0.0,
        "visual_response_coverage_on_material_pairs": len(detected_material) / len(material) if material else None,
        "material_response_effective_score": float(np.mean(material_hits)) if material_hits else None,
        "material_response_conditional_direction": (
            float(np.mean([row["yaw_match"] for row in detected_material])) if detected_material else None
        ),
        "zero_action_pairs": len(zero),
        "zero_response_specificity": float(np.mean(zero_hits)) if zero_hits else None,
        "state_confusion": {key: dict(value) for key, value in confusion.items()},
        "intervention_strength": {
            "action_delta_abs_median_rad": float(np.median([abs(float(row["action_yaw_delta_rad"])) for row in material])) if material else None,
            "visual_delta_abs_median_rad_on_material": float(np.median([abs(float(row["visual_yaw_delta_rad"])) for row in material_with_visual])) if material_with_visual else None,
            "abs_delta_spearman_on_material_with_visual": magnitude_spearman,
            "bins": material_strength,
        },
        "common_history_boundary_diagnostic": {
            "available_pairs": len(boundary_deltas),
            "abs_delta_median_rad": float(np.median(boundary_deltas)) if boundary_deltas else None,
            "abs_delta_p95_rad": float(np.quantile(boundary_deltas, 0.95)) if boundary_deltas else None,
            "note": "This is diagnostic only; RCS is based on the full paired visual response.",
        },
        "future_only_sensitivity": {
            "status": "diagnostic_only",
            "definition": "exclude history-to-first-future interval and score future-to-future intervals only",
            "available_pairs": len(future_only_available),
            "RCS_yaw_100": 100.0 * float(np.mean(future_only_hits)) if future_only_hits else None,
            "formal_hit_count": int(sum(future_only_hits)),
        },
        "progress_diagnostic": {
            "status": config["progress"]["status"],
            "pair_coverage": len(progress) / len(contract_valid) if contract_valid else 0.0,
            "ternary_match": float(np.mean([row["match"] for row in progress])) if progress else None,
            "scored_pairs": len(progress),
        },
        "controls": {
            "visual_branch_swap_material_score": (
                float(np.mean(swapped_material)) if swapped_material else None
            ),
        },
        "formal_metric_eligible": not eligibility_reasons,
        "formal_metric_eligibility_reasons": eligibility_reasons,
        "pairs": pair_rows,
        "claim_boundary": config["claim_boundary"],
    }

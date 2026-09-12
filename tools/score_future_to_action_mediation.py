#!/usr/bin/env python3
"""Score a preregistered future-to-action pathway intervention.

Input is JSONL with one row per source and condition.  Each row must contain
``source_key``, ``condition`` and a finite ``native_action`` vector.  The four
conditions are baseline, future_perturbed, future_perturbed_pathway_blocked,
and future_fixed_action_pathway_control.  The script never reads candidate
trajectories or GT and fails closed when invariance metadata do not match.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = (
    "baseline",
    "future_perturbed",
    "future_perturbed_pathway_blocked",
    "future_fixed_action_pathway_control",
)

# These are the preregistered promotion gates in
# configs/future_to_action_mediation_v1.json.  Keeping the values here makes
# the command-line report self-contained; the report still records the config
# path and the exact values used.
PROMOTION_CRITERIA = {
    "minimum_sources": 30,
    "future_effect_ci95_lower_min": 0.05,
    "pathway_suppression_ci95_lower_min": 0.5,
    "specificity_control_ci95_upper_max": 0.25,
    "source_disjoint_confirmation": True,
}

EXPECTED_PATHWAY_STATE = {
    "baseline": "normal",
    "future_perturbed": "normal",
    "future_perturbed_pathway_blocked": "blocked",
    "future_fixed_action_pathway_control": "fixed_action",
}


def _vector(value: Any, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a non-empty finite vector")
    return array


def _normalization(row: dict[str, Any]) -> tuple[str, np.ndarray]:
    """Read the frozen, calibration-only action normalization contract.

    The scorer must not silently compare raw action coordinates when the
    protocol declares a calibration-fitted normalization.  Every condition
    therefore carries the same positive scale vector and fingerprint.
    """
    fingerprint = str(row.get("action_normalization_fingerprint") or "")
    if not fingerprint:
        raise ValueError("action_normalization_fingerprint_required")
    scale = np.asarray(row.get("action_normalization_scale"), dtype=np.float64)
    if scale.ndim != 1 or not len(scale) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("action_normalization_scale_invalid")
    return fingerprint, scale


def _percentile_ci(values: np.ndarray, draws: int, seed: int) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        means[index] = float(np.mean(values[rng.integers(0, len(values), len(values))]))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def score(
    rows: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
    calibration_source_keys: set[str] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        source = str(row.get("source_key") or "")
        condition = str(row.get("condition") or "")
        if not source or condition not in CONDITIONS:
            raise ValueError("each row requires source_key and a known condition")
        if condition in grouped[source]:
            raise ValueError(f"{source}: duplicate condition {condition}")
        grouped[source][condition] = row

    pairs: list[dict[str, Any]] = []
    future_effects: list[float] = []
    blocked_effects: list[float] = []
    suppressions: list[float] = []
    specificity: list[float] = []
    calibration_source_keys = None if calibration_source_keys is None else set(calibration_source_keys)
    for source in sorted(grouped):
        group = grouped[source]
        missing = [condition for condition in CONDITIONS if condition not in group]
        pair: dict[str, Any] = {"source_key": source, "status": "scored"}
        if calibration_source_keys is not None and source in calibration_source_keys:
            pair.update({"status": "unavailable", "reason": "source_in_calibration_split"})
            pairs.append(pair)
            continue
        if missing:
            pair.update({"status": "unavailable", "reason": "missing_conditions", "missing": missing})
            pairs.append(pair)
            continue
        metadata = {}
        for condition in CONDITIONS:
            row = group[condition]
            current = {
                "history_fingerprint": row.get("history_fingerprint"),
                "command_fingerprint": row.get("command_fingerprint"),
                "nuisance_seed": row.get("nuisance_seed"),
                "model_revision": row.get("model_revision"),
            }
            if any(value is None or value == "" for value in current.values()):
                pair.update({"status": "unavailable", "reason": "invariance_metadata_required"})
                pairs.append(pair)
                break
            if not metadata:
                metadata = current
            elif current != metadata:
                pair.update({"status": "unavailable", "reason": "invariance_mismatch"})
                pairs.append(pair)
                break
        if pair["status"] != "scored":
            continue
        future_fingerprints = {
            condition: group[condition].get("future_fingerprint")
            for condition in CONDITIONS
        }
        if any(value is None or value == "" for value in future_fingerprints.values()):
            pair.update({"status": "unavailable", "reason": "future_fingerprint_required"})
            pairs.append(pair)
            continue
        if future_fingerprints["baseline"] == future_fingerprints["future_perturbed"]:
            pair.update({"status": "unavailable", "reason": "future_perturbation_not_verified"})
            pairs.append(pair)
            continue
        if future_fingerprints["future_perturbed"] != future_fingerprints["future_perturbed_pathway_blocked"]:
            pair.update({"status": "unavailable", "reason": "blocked_condition_future_mismatch"})
            pairs.append(pair)
            continue
        if future_fingerprints["baseline"] != future_fingerprints["future_fixed_action_pathway_control"]:
            pair.update({"status": "unavailable", "reason": "fixed_action_control_future_mismatch"})
            pairs.append(pair)
            continue
        pathway_states = {condition: group[condition].get("pathway_state") for condition in CONDITIONS}
        if pathway_states != EXPECTED_PATHWAY_STATE:
            pair.update({"status": "unavailable", "reason": "pathway_state_contract_mismatch", "pathway_states": pathway_states})
            pairs.append(pair)
            continue
        try:
            normalizations = {
                condition: _normalization(group[condition]) for condition in CONDITIONS
            }
            normalization_fingerprints = {
                value[0] for value in normalizations.values()
            }
            if len(normalization_fingerprints) != 1:
                raise ValueError("action_normalization_mismatch")
            scales = [value[1] for value in normalizations.values()]
            if any(not np.array_equal(scales[0], scale) for scale in scales[1:]):
                raise ValueError("action_normalization_mismatch")
            scale = scales[0]
            baseline = _vector(group["baseline"].get("native_action"), "native_action")
            future = _vector(group["future_perturbed"].get("native_action"), "native_action")
            blocked = _vector(group["future_perturbed_pathway_blocked"].get("native_action"), "native_action")
            control = _vector(group["future_fixed_action_pathway_control"].get("native_action"), "native_action")
            if not (len(baseline) == len(future) == len(blocked) == len(control)):
                raise ValueError("native_action vector lengths differ")
            if len(scale) != len(baseline):
                raise ValueError("action_normalization_scale_length_mismatch")
            baseline = baseline / scale
            future = future / scale
            blocked = blocked / scale
            control = control / scale
        except ValueError as error:
            pair.update({"status": "unavailable", "reason": str(error)})
            pairs.append(pair)
            continue
        future_effect = float(np.linalg.norm(future - baseline))
        blocked_effect = float(np.linalg.norm(blocked - baseline))
        control_effect = float(np.linalg.norm(control - baseline))
        suppression = None if future_effect <= 1e-12 else float(1.0 - blocked_effect / future_effect)
        pair.update({
            "future_effect_on_action": future_effect,
            "blocked_effect_on_action": blocked_effect,
            "specificity_control_effect": control_effect,
            "pathway_suppression": suppression,
            "invariance": metadata,
            "future_fingerprints": future_fingerprints,
            "pathway_states": pathway_states,
            "action_normalization": {
                "fingerprint": next(iter(normalization_fingerprints)),
                "scale": [float(value) for value in scale],
            },
        })
        future_effects.append(future_effect)
        blocked_effects.append(blocked_effect)
        specificity.append(control_effect)
        if suppression is not None:
            suppressions.append(suppression)
        pairs.append(pair)

    scored = [pair for pair in pairs if pair["status"] == "scored"]
    future_array = np.asarray(future_effects, dtype=np.float64)
    blocked_array = np.asarray(blocked_effects, dtype=np.float64)
    suppression_array = np.asarray(suppressions, dtype=np.float64)
    specificity_array = np.asarray(specificity, dtype=np.float64)
    future_ci = _percentile_ci(future_array, draws, seed)
    suppression_ci = _percentile_ci(suppression_array, draws, seed + 2)
    specificity_ci = _percentile_ci(specificity_array, draws, seed + 3)
    promotion_checks = {
        "minimum_sources": len(scored) >= PROMOTION_CRITERIA["minimum_sources"],
        "future_effect_ci95_lower": (
            future_ci is not None
            and future_ci[0] >= PROMOTION_CRITERIA["future_effect_ci95_lower_min"]
        ),
        "pathway_suppression_ci95_lower": (
            suppression_ci is not None
            and suppression_ci[0] >= PROMOTION_CRITERIA["pathway_suppression_ci95_lower_min"]
        ),
        "specificity_control_ci95_upper": (
            specificity_ci is not None
            and specificity_ci[1] <= PROMOTION_CRITERIA["specificity_control_ci95_upper_max"]
        ),
        "source_disjoint_confirmation": calibration_source_keys is not None,
    }
    promotion_status = (
        "passed"
        if all(promotion_checks.values())
        else "insufficient_evidence"
        if len(scored) < PROMOTION_CRITERIA["minimum_sources"] or calibration_source_keys is None
        else "failed"
    )
    return {
        "protocol": "iac-future-to-action-mediation-v1",
        "status": "scored" if scored else "unavailable",
        "source_count": len(pairs),
        "scored_source_count": len(scored),
        "coverage": len(scored) / len(pairs) if pairs else None,
        "future_effect_on_action": {
            "median": float(np.median(future_array)) if len(future_array) else None,
            "mean": float(np.mean(future_array)) if len(future_array) else None,
            "bootstrap_ci95": future_ci,
        },
        "blocked_effect_on_action": {
            "median": float(np.median(blocked_array)) if len(blocked_array) else None,
            "mean": float(np.mean(blocked_array)) if len(blocked_array) else None,
            "bootstrap_ci95": _percentile_ci(blocked_array, draws, seed + 1),
        },
        "pathway_suppression": {
            "median": float(np.median(suppression_array)) if len(suppression_array) else None,
            "mean": float(np.mean(suppression_array)) if len(suppression_array) else None,
            "bootstrap_ci95": suppression_ci,
        },
        "specificity_control_effect": {
            "median": float(np.median(specificity_array)) if len(specificity_array) else None,
            "mean": float(np.mean(specificity_array)) if len(specificity_array) else None,
            "bootstrap_ci95": specificity_ci,
        },
        "promotion": {
            "status": promotion_status,
            "criteria": dict(PROMOTION_CRITERIA),
            "checks": promotion_checks,
            "claim_enabled": promotion_status == "passed",
        },
        "source_disjoint_confirmation": {
            "verified": calibration_source_keys is not None,
            "calibration_source_count": (
                len(calibration_source_keys) if calibration_source_keys is not None else None
            ),
            "overlap_count": (
                len(set(grouped).intersection(calibration_source_keys))
                if calibration_source_keys is not None else None
            ),
        },
        "pairs": pairs,
        "missing_value_policy": "unavailable_never_zero_fill",
        "claim_boundary": "This is a pathway intervention result, not a replacement for MAS, RCS or GS; passing requires the preregistered thresholds and source-disjoint confirmation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    parser.add_argument(
        "--calibration-sources",
        type=Path,
        help="one source_key per line; supplying this is required for promotion evidence",
    )
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help="return non-zero unless the preregistered mediation gates pass",
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    calibration_sources = None
    if args.calibration_sources:
        calibration_sources = {
            line.strip()
            for line in args.calibration_sources.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    result = score(
        rows,
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
        calibration_source_keys=calibration_sources,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "coverage": result["coverage"],
        "promotion": result["promotion"]["status"],
        "output": str(args.output),
    }))
    if result["status"] == "unavailable" or (
        args.require_promotion and not result["promotion"]["claim_enabled"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

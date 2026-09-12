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


def _percentile_ci(values: np.ndarray, draws: int, seed: int) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        means[index] = float(np.mean(values[rng.integers(0, len(values), len(values))]))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def score(rows: list[dict[str, Any]], *, draws: int, seed: int) -> dict[str, Any]:
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
    for source in sorted(grouped):
        group = grouped[source]
        missing = [condition for condition in CONDITIONS if condition not in group]
        pair: dict[str, Any] = {"source_key": source, "status": "scored"}
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
            baseline = _vector(group["baseline"].get("native_action"), "native_action")
            future = _vector(group["future_perturbed"].get("native_action"), "native_action")
            blocked = _vector(group["future_perturbed_pathway_blocked"].get("native_action"), "native_action")
            control = _vector(group["future_fixed_action_pathway_control"].get("native_action"), "native_action")
            if not (len(baseline) == len(future) == len(blocked) == len(control)):
                raise ValueError("native_action vector lengths differ")
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
    return {
        "protocol": "iac-future-to-action-mediation-v1",
        "status": "scored" if scored else "unavailable",
        "source_count": len(pairs),
        "scored_source_count": len(scored),
        "coverage": len(scored) / len(pairs) if pairs else None,
        "future_effect_on_action": {
            "median": float(np.median(future_array)) if len(future_array) else None,
            "mean": float(np.mean(future_array)) if len(future_array) else None,
            "bootstrap_ci95": _percentile_ci(future_array, draws, seed),
        },
        "blocked_effect_on_action": {
            "median": float(np.median(blocked_array)) if len(blocked_array) else None,
            "mean": float(np.mean(blocked_array)) if len(blocked_array) else None,
            "bootstrap_ci95": _percentile_ci(blocked_array, draws, seed + 1),
        },
        "pathway_suppression": {
            "median": float(np.median(suppression_array)) if len(suppression_array) else None,
            "mean": float(np.mean(suppression_array)) if len(suppression_array) else None,
            "bootstrap_ci95": _percentile_ci(suppression_array, draws, seed + 2),
        },
        "specificity_control_effect": {
            "median": float(np.median(specificity_array)) if len(specificity_array) else None,
            "mean": float(np.mean(specificity_array)) if len(specificity_array) else None,
            "bootstrap_ci95": _percentile_ci(specificity_array, draws, seed + 3),
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
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = score(rows, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "coverage": result["coverage"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

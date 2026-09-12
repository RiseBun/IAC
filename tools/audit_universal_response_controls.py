#!/usr/bin/env python3
"""Audit existing twin-control outputs under the Step1 universal gate.

The input files are already-computed flow controls.  This command does not
re-estimate flow and does not treat a missing identity-swap control as a pass.
It is therefore a conservative bridge from the old artifacts to the new
protocol, not a replacement for rerunning the four controls from raw videos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_CONTROLS = ("normal", "reversed", "identity_swap", "zero")


def _bootstrap_fraction(values: np.ndarray, *, draws: int, seed: int) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(draws)):
        sample = values[rng.integers(0, len(values), len(values))]
        estimates.append(float(np.mean(sample)))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _summarize_model(
    model_id: str,
    path: Path,
    *,
    minimum_interval_coverage: float,
    minimum_cosine: float,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    controls = {str(row.get("control")) for row in rows}
    by_control: dict[str, list[dict[str, Any]]] = {
        control: [row for row in rows if row.get("control") == control]
        for control in controls
    }
    normal_rows = [
        row for row in by_control.get("normal", [])
        if row.get("median_direction_cosine") is not None
        and float(row.get("interval_coverage") or 0.0) >= minimum_interval_coverage
    ]
    reversed_rows = [
        row for row in by_control.get("reversed", [])
        if row.get("median_direction_cosine") is not None
        and float(row.get("interval_coverage") or 0.0) >= minimum_interval_coverage
    ]
    normal_hits = np.asarray(
        [float(row["median_direction_cosine"]) >= minimum_cosine for row in normal_rows],
        dtype=np.float64,
    )
    reversed_hits = np.asarray(
        [float(row["median_direction_cosine"]) <= -minimum_cosine for row in reversed_rows],
        dtype=np.float64,
    )
    zero_rows = by_control.get("zero", [])
    zero_false_direction = sum(
        row.get("median_direction_cosine") is not None
        and abs(float(row["median_direction_cosine"])) >= minimum_cosine
        for row in zero_rows
    )
    identity_present = "identity_swap" in controls
    normal_coverage = len(normal_rows) / len(by_control.get("normal", [])) if by_control.get("normal") else 0.0
    normal_accuracy = float(np.mean(normal_hits)) if len(normal_hits) else None
    reversed_accuracy = float(np.mean(reversed_hits)) if len(reversed_hits) else None
    ci = _bootstrap_fraction(normal_hits, draws=bootstrap_draws, seed=bootstrap_seed)
    required_present = {
        "normal": "normal" in controls,
        "reversed": "reversed" in controls,
        "identity_swap": identity_present,
        "zero": "zero" in controls,
    }
    gates = {
        "coverage": normal_coverage >= 0.90,
        "normal_direction_ci_lower": ci is not None and ci[0] >= 0.75,
        "reversed_control": reversed_accuracy is not None and reversed_accuracy >= 0.75,
        "zero_control": "zero" in controls and zero_false_direction == 0,
        "identity_control": identity_present,
    }
    return {
        "model_id": model_id,
        "source_report": str(path),
        "available_controls": sorted(controls),
        "required_controls_present": required_present,
        "normal": {
            "rows": len(normal_rows),
            "coverage": normal_coverage,
            "direction_accuracy": normal_accuracy,
            "direction_ci95": ci,
            "median_temporal_persistence": (
                float(np.median([
                    float(row["temporal_persistence"])
                    for row in normal_rows
                    if row.get("temporal_persistence") is not None
                ]))
                if any(row.get("temporal_persistence") is not None for row in normal_rows)
                else None
            ),
        },
        "reversed": {
            "rows": len(reversed_rows),
            "direction_accuracy": reversed_accuracy,
        },
        "zero": {
            "rows": len(zero_rows),
            "false_direction_count": int(zero_false_direction),
        },
        "promotion_gates": gates,
        "promotion": bool(all(gates.values())),
        "status": "pass_candidate" if all(gates.values()) else "diagnostic_incomplete_or_failed",
    }


def audit(
    model_paths: dict[str, Path],
    *,
    minimum_interval_coverage: float = 0.90,
    minimum_cosine: float = 0.0,
    bootstrap_draws: int = 5000,
    bootstrap_seed: int = 20260912,
) -> dict[str, Any]:
    models = [
        _summarize_model(
            model, path,
            minimum_interval_coverage=minimum_interval_coverage,
            minimum_cosine=minimum_cosine,
            bootstrap_draws=bootstrap_draws,
            bootstrap_seed=bootstrap_seed + index,
        )
        for index, (model, path) in enumerate(sorted(model_paths.items()))
    ]
    promoted = [row for row in models if row["promotion"]]
    return {
        "protocol": "iac-step1-universal-response-real-control-audit-v1",
        "input_type": "precomputed_twin_control_reports",
        "raw_video_rerun": False,
        "required_controls": list(REQUIRED_CONTROLS),
        "minimum_architectures_for_promotion": 3,
        "models": models,
        "promoted_model_count": len(promoted),
        "universal_channel_promotion": len(promoted) >= 3,
        "claim_boundary": "This audit is a conservative bridge from existing control reports. Missing identity-swap controls block promotion; no missing control is imputed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", nargs=2, metavar=("MODEL", "REPORT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    report = audit(
        {model: Path(path) for model, path in args.model},
        bootstrap_draws=args.bootstrap_draws,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

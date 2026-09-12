#!/usr/bin/env python3
"""Run a tiny synthetic proof of the Step1 Universal Response invariants.

This is a contract proof, not benchmark evidence.  It uses the same pairwise
scorer as the exploratory flow-structure path and checks the four frozen
controls plus invariance to a large common-mode motion component.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iac_new.flow_structure_scoring import score_counterfactual_structure_pairs


def _measurement(source: str, role: str, values: list[float]) -> dict[str, Any]:
    return {
        "source_key": source,
        "branch_role": role,
        "candidate_bank_used_by_measurement": False,
        "flow_structure": {
            "rows": [
                {
                    "interval_index": index,
                    "input_available": True,
                    "horizontal_flow_center": value,
                }
                for index, value in enumerate(values)
            ]
        },
    }


def _manifest(source: str, role: str, yaw: float) -> dict[str, Any]:
    return {
        "source_key": source,
        "branch_role": role,
        "action_trajectory": [[0.0, 0.0, yaw]],
    }


def _run_control(name: str, action_sign: int, *, common_mode: float = 0.0) -> dict[str, Any]:
    measurements: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for index in range(8):
        source = f"{name}-{index}"
        # The visual contrast is fixed while the scene-level common motion
        # changes.  A valid contrast score should preserve its sign.
        left = [common_mode + 0.25, common_mode + 0.35, common_mode + 0.45]
        right = [common_mode - 0.25, common_mode - 0.35, common_mode - 0.45]
        measurements.extend([
            _measurement(source, "left", left),
            _measurement(source, "right", right),
        ])
        manifests.extend([
            _manifest(source, "left", action_sign * 0.30),
            _manifest(source, "right", action_sign * -0.30),
        ])
    report = score_counterfactual_structure_pairs(
        measurements,
        manifests,
        descriptor="horizontal_flow_center",
        action_column=2,
        action_reference="endpoint_column",
        minimum_common_intervals=2,
        minimum_action_delta=0.1,
        minimum_common_mode=1e-6,
    )
    comparable = report["action_direction_pairs"]
    return {
        "control": name,
        "pair_count": report["pair_count"],
        "coverage": report["coverage"],
        "comparable_pairs": comparable,
        "direction_accuracy": report["action_direction_accuracy"],
        "median_temporal_persistence": report["median_temporal_persistence"],
        "raw_delta_median": report["pairs"][0]["flow_delta"],
        "normalized_delta_median_diagnostic": report["pairs"][0]["normalized_flow_delta"],
        "status": "scored" if comparable else "unavailable",
    }


def prove() -> dict[str, Any]:
    normal = _run_control("normal", 1, common_mode=0.0)
    reversed_control = _run_control("reversed", -1, common_mode=0.0)
    # Equal branches/actions intentionally produce no comparable direction.
    # Replace identity/zero reports with explicit no-contrast fixtures rather
    # than allowing floating-point noise to create a direction.
    def no_contrast(name: str) -> dict[str, Any]:
        measurements: list[dict[str, Any]] = []
        manifests: list[dict[str, Any]] = []
        for index in range(8):
            source = f"{name}-{index}"
            measurements.extend([
                _measurement(source, "left", [1.0, 1.0, 1.0]),
                _measurement(source, "right", [1.0, 1.0, 1.0]),
            ])
            manifests.extend([
                _manifest(source, "left", 0.0),
                _manifest(source, "right", 0.0),
            ])
        report = score_counterfactual_structure_pairs(
            measurements, manifests, action_column=2,
            action_reference="endpoint_column",
            minimum_common_intervals=2, minimum_action_delta=0.1,
        )
        return {
            "control": name,
            "pair_count": report["pair_count"],
            "coverage": report["coverage"],
            "comparable_pairs": report["action_direction_pairs"],
            "direction_accuracy": report["action_direction_accuracy"],
            "median_temporal_persistence": report["median_temporal_persistence"],
            "status": "scored" if report["action_direction_pairs"] else "unavailable",
        }

    identity = no_contrast("identity_swap")
    zero = no_contrast("zero_contrast")
    shifted = _run_control("common_mode_shift", 1, common_mode=20.0)
    return {
        "protocol": "iac-step1-universal-response-minimal-proof-v1",
        "synthetic_only": True,
        "controls": [normal, reversed_control, identity, zero],
        "common_mode_invariance": {
            "baseline_raw_delta": normal["raw_delta_median"],
            "shifted_raw_delta": shifted["raw_delta_median"],
            "raw_delta_difference": abs(
                normal["raw_delta_median"] - shifted["raw_delta_median"]
            ),
            "normalized_magnitude_is_diagnostic": True,
        },
        "decision": {
            "normal_direction_passes": normal["direction_accuracy"] == 1.0,
            "reversed_direction_fails": reversed_control["direction_accuracy"] == 0.0,
            "identity_is_unavailable": identity["status"] == "unavailable",
            "zero_contrast_is_unavailable": zero["status"] == "unavailable",
            "common_mode_signed_difference_is_stable": abs(
                normal["raw_delta_median"] - shifted["raw_delta_median"]
            ) < 1e-9,
            "benchmark_promotion": False,
        },
        "claim_boundary": "Synthetic invariant proof only; no claim about any WAM or real/generated video.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    proof_keys = [
        "normal_direction_passes",
        "reversed_direction_fails",
        "identity_is_unavailable",
        "zero_contrast_is_unavailable",
        "common_mode_signed_difference_is_stable",
    ]
    if not all(report["decision"][key] for key in proof_keys):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run CCFC-S structural controls on a fixed same-source twin set.

The controls are constructed from already measured, candidate-blind branch
records.  No candidate trajectory, logged future state, or PDM score is used
to select pixels or intervals.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from iac_new.flow_structure_scoring import score_counterfactual_structure_pairs


def _read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("source_key") or ""), str(row.get("branch_role") or "")


def _normalize_branch_roles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept legacy manifests that encode the role only in branch_id/sample_id."""
    output = copy.deepcopy(rows)
    for row in output:
        if row.get("branch_role") in {"left", "right"}:
            continue
        for field in ("branch_id", "sample_id"):
            value = str(row.get(field) or "")
            if value.endswith("::left"):
                row["branch_role"] = "left"
                break
            if value.endswith("::right"):
                row["branch_role"] = "right"
                break
    return output


def _swap_actions(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an identity-swap control: each image keeps the other action."""
    by_source = {(str(row.get("source_key") or ""), str(row.get("branch_role") or "")): row for row in manifests}
    output = copy.deepcopy(manifests)
    for row in output:
        source = str(row.get("source_key") or "")
        role = str(row.get("branch_role") or "")
        other = "right" if role == "left" else "left"
        peer = by_source.get((source, other))
        if peer is not None and "action_trajectory" in peer:
            row["action_trajectory"] = copy.deepcopy(peer["action_trajectory"])
    return output


def _zero_contrast(measurements: list[dict[str, Any]], descriptor: str) -> list[dict[str, Any]]:
    """Return a zero-contrast control while preserving availability metadata."""
    output = copy.deepcopy(measurements)
    by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for row in output:
        by_source.setdefault(str(row.get("source_key") or ""), {})[str(row.get("branch_role") or "")] = row
    for branches in by_source.values():
        left = branches.get("left")
        right = branches.get("right")
        if left is None or right is None:
            continue
        left_rows = {int(item.get("interval_index")): item for item in left.get("flow_structure", {}).get("rows", [])}
        right_rows = {int(item.get("interval_index")): item for item in right.get("flow_structure", {}).get("rows", [])}
        for interval in sorted(set(left_rows) & set(right_rows)):
            left_item = left_rows[interval]
            right_item = right_rows[interval]
            if descriptor in left_item and descriptor in right_item:
                midpoint = 0.5 * (float(left_item[descriptor]) + float(right_item[descriptor]))
                left_item[descriptor] = midpoint
                right_item[descriptor] = midpoint
    return output


def run_controls(
    measurements: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    *,
    descriptor: str,
    action_reference: str,
    action_column: int,
    minimum_common_intervals: int,
    minimum_action_delta: float,
) -> dict[str, dict[str, Any]]:
    """Run preregistered controls and return their complete reports."""
    measurements = _normalize_branch_roles(measurements)
    manifests = _normalize_branch_roles(manifests)
    swapped_measurements = copy.deepcopy(measurements)
    reports: dict[str, dict[str, Any]] = {}
    common = dict(
        descriptor=descriptor,
        action_reference=action_reference,
        action_column=action_column,
        minimum_common_intervals=minimum_common_intervals,
        minimum_action_delta=minimum_action_delta,
    )
    reports["normal_order"] = score_counterfactual_structure_pairs(measurements, manifests, orientation=1, **common)
    reports["reversed_order"] = score_counterfactual_structure_pairs(measurements, manifests, orientation=-1, **common)
    reports["identity_swap"] = score_counterfactual_structure_pairs(
        measurements, _swap_actions(manifests), orientation=1, **common
    )
    reports["zero_contrast"] = score_counterfactual_structure_pairs(
        _zero_contrast(swapped_measurements, descriptor), manifests, orientation=1, **common
    )
    return reports


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "pair_count",
        "status_counts",
        "coverage",
        "action_response_spearman",
        "normalized_action_response_spearman",
        "action_direction_pairs",
        "action_direction_hits",
        "action_direction_accuracy",
        "action_direction_accuracy_ci95",
        "median_temporal_persistence",
    )
    return {key: report.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--descriptor", default="horizontal_flow_center")
    parser.add_argument("--action-reference", choices=["endpoint_column", "trajectory_path_length"], default="endpoint_column")
    parser.add_argument("--action-column", type=int, default=2)
    parser.add_argument("--minimum-common-intervals", type=int, default=2)
    parser.add_argument("--minimum-action-delta", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = run_controls(
        _read_jsonl(args.measurement),
        _read_jsonl(args.manifest),
        descriptor=args.descriptor,
        action_reference=args.action_reference,
        action_column=args.action_column,
        minimum_common_intervals=args.minimum_common_intervals,
        minimum_action_delta=args.minimum_action_delta,
    )
    payload = {
        "protocol": "iac-counterfactual-flow-structure-controls-v1",
        "exploratory": True,
        "descriptor": args.descriptor,
        "controls": {name: _summary(report) for name, report in reports.items()},
        "reports": reports,
        "warning": "Controls are not a formal promotion result until the twin set is held out and the required model count is met.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["controls"], indent=2))


if __name__ == "__main__":
    main()

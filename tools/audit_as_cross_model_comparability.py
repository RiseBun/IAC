#!/usr/bin/env python3
"""Audit whether AS reports can support a strict cross-model comparison."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_HISTORY_TIMES = (-1.5, -1.0, -0.5, 0.0)
EXPECTED_FUTURE_TIMES = (1.0, 2.0, 3.0, 4.0)
NATIVE_ACTION_SOURCES = {"wam_action_head", "native_action_head"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _times(value: Any) -> tuple[float, ...]:
    return tuple(round(float(item), 4) for item in (value or []))


def _row_candidate_blind(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    if metadata.get("candidate_blind_image_branch") is True:
        return True
    return (
        row.get("candidate_bank_used_by_decoder") is False
        and metadata.get("action_waypoint_used_by_image_branch") is False
    )


def _audit_model(root: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    manifest_path = root / spec["manifest"]
    report_path = root / spec["report"]
    rows = _read_jsonl(manifest_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    aggregate = report.get("aggregate") or {}
    sources = {str(row.get("source_key")) for row in rows if row.get("source_key")}
    action_sources = Counter(str(row.get("action_trajectory_source")) for row in rows)

    row_gates = {
        "four_history_frames": all(len(row.get("history_frame_paths") or []) == 4 for row in rows),
        "four_future_frames": all(len(row.get("future_frame_paths") or row.get("future_images") or []) == 4 for row in rows),
        "four_action_states": all(len(row.get("action_trajectory") or []) == 4 for row in rows),
        "history_timeline": all(_times(row.get("history_times_s")) == EXPECTED_HISTORY_TIMES for row in rows),
        "future_timeline": all(_times(row.get("future_times_s")) == EXPECTED_FUTURE_TIMES for row in rows),
        "camera_intrinsics": all(bool(row.get("intrinsics")) for row in rows),
        "camera_to_ego": all(bool(row.get("camera_to_ego")) for row in rows),
        "candidate_blind_visual_measurement": all(_row_candidate_blind(row) for row in rows),
        "generated_future": all(bool(row.get("future_images_source")) for row in rows),
    }
    own_native_action = bool(rows) and set(action_sources).issubset(NATIVE_ACTION_SOURCES)
    lineage = aggregate.get("input_lineage_audit") or {}
    lineage_passed = lineage.get("formal_metric_eligible") is True and lineage.get("status") == "passed"
    declared_contract = str(spec.get("action_contract") or "unknown")
    selection_policy = str(spec.get("selection_policy") or "unknown")
    individual_gates = {
        **row_gates,
        "own_native_action": own_native_action and declared_contract == "own_native",
        "lineage_passed": lineage_passed,
        "selection_independent_of_evaluated_action": selection_policy not in {
            "action_materiality_prefilter",
            "score_prefilter",
        },
    }
    return {
        "label": spec["label"],
        "wam_model_ids": sorted({str(row.get("wam_model_id")) for row in rows}),
        "row_count": len(rows),
        "source_count": len(sources),
        "action_sources": dict(action_sources),
        "action_contract": declared_contract,
        "selection_policy": selection_policy,
        "as_protocol": report.get("protocol"),
        "visual_protocol": report.get("visual_protocol"),
        "AS_conditional_100": aggregate.get("AS_conditional_100"),
        "AS_overall_100": aggregate.get("AS_overall_100"),
        "legacy_AS_composite_100": (
            100.0 * float(aggregate["AS_composite_mean"])
            if aggregate.get("AS_composite_mean") is not None
            else None
        ),
        "gates": individual_gates,
        "individually_eligible": bool(rows) and all(individual_gates.values()),
    }, sources


def audit(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[1]
    model_results: list[dict[str, Any]] = []
    source_sets: dict[str, set[str]] = {}
    for spec in config["models"]:
        result, sources = _audit_model(root, spec)
        model_results.append(result)
        source_sets[spec["label"]] = sources

    labels = [item["label"] for item in model_results]
    pairwise = []
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            common = source_sets[left] & source_sets[right]
            pairwise.append({
                "left": left,
                "right": right,
                "common_source_count": len(common),
                "same_declared_source_pool": source_sets[left] == source_sets[right],
            })

    protocol_values = {item["as_protocol"] for item in model_results}
    visual_values = {item["visual_protocol"] for item in model_results}
    all_common = set.intersection(*(source_sets[label] for label in labels)) if labels else set()
    minimum_common_sources = int(config.get("minimum_common_sources_for_formal_comparison", 30))
    joint_gates = {
        "at_least_two_models": len(model_results) >= 2,
        "all_models_individually_eligible": all(item["individually_eligible"] for item in model_results),
        "same_as_protocol": len(protocol_values) == 1 and None not in protocol_values,
        "same_visual_protocol": len(visual_values) == 1 and None not in visual_values,
        "nonempty_all_model_source_intersection": bool(all_common),
        "same_declared_source_pool": bool(source_sets) and len({frozenset(value) for value in source_sets.values()}) == 1,
        "minimum_common_sources": len(all_common) >= minimum_common_sources,
        "source_is_comparison_unit": config.get("comparison_unit") == "source",
    }
    eligible = all(joint_gates.values())
    return {
        "protocol": "iac-as-cross-model-comparability-audit-v1",
        "config": str(config_path),
        "models": model_results,
        "pairwise_source_overlap": pairwise,
        "all_model_common_source_count": len(all_common),
        "minimum_common_sources_for_formal_comparison": minimum_common_sources,
        "comparison_unit": config.get("comparison_unit"),
        "joint_gates": joint_gates,
        "strict_cross_model_leaderboard_eligible": eligible,
        "decision": (
            "strict comparison is allowed on the declared common pool"
            if eligible
            else "report model results separately; no cross-model AS ranking"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "eligible": result["strict_cross_model_leaderboard_eligible"],
        "all_model_common_source_count": result["all_model_common_source_count"],
    }, indent=2))


if __name__ == "__main__":
    main()

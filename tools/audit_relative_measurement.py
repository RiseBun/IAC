#!/usr/bin/env python3
"""Validate Step 1 in its declared relative and arc-relative coordinates."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from iac_new.continuous_motion import (
    compare_distance_profiles,
    compare_pose_profiles,
    history_only_motion_profile,
    trajectory_to_motion_profile,
)


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(line) for path in paths for line in path.open() if line.strip()]


def reference_trajectory(record: dict[str, Any]) -> np.ndarray:
    candidate_id = record.get("gt_candidate_id")
    for candidate in record.get("candidates", []):
        if str(candidate.get("candidate_id")) == str(candidate_id):
            return np.asarray(candidate["trajectory"], dtype=np.float64)
    raise KeyError(f"missing gt candidate for {record.get('sample_id')}")


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def paired_improvement(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    values = np.asarray([
        row[f"history_{name}"] - row[name] for row in rows
        if name in row and f"history_{name}" in row
    ], dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "bootstrap_mean_ci95": None}
    rng = np.random.default_rng(17)
    draws = np.asarray([
        np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(2000)
    ])
    return {
        "n": int(len(values)),
        "definition": "history_error_minus_step1_error; positive favors Step 1",
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "bootstrap_mean_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "relative_progress_mae",
        "relative_progress_increment_mae",
        "arc_translation_mae",
        "arc_forward_mae",
        "arc_lateral_mae",
        "heading_mae_rad",
    )
    return {
        "n": len(rows),
        "metrics": {
            name: describe([row[name] for row in rows if name in row]) for name in metrics
        },
        "history_metrics_on_common_support": {
            name: describe([
                row[f"history_{name}"] for row in rows
                if name in row and f"history_{name}" in row
            ])
            for name in metrics
        },
        "beats_history_fraction": {
            name: float(np.mean([
                row[name] < row[f"history_{name}"] for row in rows
                if name in row and f"history_{name}" in row
            ]))
            if any(name in row and f"history_{name}" in row for row in rows) else None
            for name in metrics
        },
        "paired_improvement_over_history": {
            name: paired_improvement(rows, name) for name in metrics
        },
    }


def direction_and_rank(rows: list[dict[str, Any]], field: str, threshold: float) -> dict[str, Any]:
    selected = [
        row for row in rows
        if "arc_translation_mae" in row and abs(row[f"reference_{field}"]) >= threshold
    ]
    if len(selected) < 3:
        return {"n": len(selected), "direction_accuracy": None, "spearman": None}
    predicted = np.asarray([row[f"predicted_{field}"] for row in selected])
    reference = np.asarray([row[f"reference_{field}"] for row in selected])
    return {
        "n": len(selected),
        "threshold": threshold,
        "direction_accuracy": float(np.mean(np.sign(predicted) == np.sign(reference))),
        "spearman": float(spearmanr(predicted, reference).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_paths = [Path(path) for pattern in args.outputs for path in glob.glob(pattern)]
    scores = read_jsonl(output_paths)
    references = {row["sample_id"]: row for row in read_jsonl([args.reference_manifest])}
    audited = []
    missing = []
    unavailable = defaultdict(int)
    for score in scores:
        reference_record = references.get(score.get("sample_id"))
        if reference_record is None:
            missing.append(score.get("sample_id"))
            continue
        status = str(score.get("motion_explanation_status", score.get("geometry_fit_status")))
        if status not in {"explained", "weak"} or not score.get("measurement_available"):
            unavailable[status] += 1
            continue
        times = np.asarray(reference_record["future_times_s"], dtype=np.float64)
        predicted_trajectory = np.asarray(score["decoder"]["trajectory"], dtype=np.float64)
        truth_trajectory = reference_trajectory(reference_record)
        predicted = trajectory_to_motion_profile(predicted_trajectory, times)
        predicted["source"] = "image_only_candidate_blind_decoder"
        truth = trajectory_to_motion_profile(truth_trajectory, times)
        history_state = reference_record.get("history_ego_state") or (
            reference_record.get("metadata") or {}
        ).get("history_ego_state")
        history = history_only_motion_profile(
            history_state,
            times,
            history_times_s=reference_record.get("history_times_s"),
            model="constant_acceleration_yaw_rate",
        )
        comparisons = {}
        for name, profile, allow_non_image in (
            ("fit", predicted, False),
            ("history", history, True),
        ):
            distance = compare_distance_profiles(
                profile,
                truth,
                scale_mode="relative",
                include_uncertain=True,
                allow_non_image_source=allow_non_image,
            )
            pose = compare_pose_profiles(
                profile,
                truth,
                scale_mode="arc_relative",
                include_uncertain=True,
                allow_non_image_source=allow_non_image,
            )
            comparisons[name] = {}
            if distance["status"] == "ok":
                comparisons[name]["distance"] = distance["metrics"]["forward_displacement_profile"]
            else:
                unavailable[f"{status}:{name}:distance:{distance.get('reason')}"] += 1
            if pose["status"] == "ok":
                comparisons[name]["pose"] = pose["metrics"]["se2_pose"]
            else:
                unavailable[f"{status}:{name}:pose:{pose.get('reason')}"] += 1
        row = {
            "sample_id": score["sample_id"],
            "status": status,
            "stratum": str((reference_record.get("metadata") or {}).get("stratum", "unknown")),
            "predicted_lateral": float(predicted_trajectory[-1, 1]),
            "reference_lateral": float(truth_trajectory[-1, 1]),
            "predicted_yaw": float(predicted_trajectory[-1, 2]),
            "reference_yaw": float(truth_trajectory[-1, 2]),
        }
        mapping = {
            "relative_progress_mae": ("distance", "mae"),
            "relative_progress_increment_mae": ("distance", "increment_mae"),
            "arc_translation_mae": ("pose", "translation_mae"),
            "arc_forward_mae": ("pose", "forward_mae"),
            "arc_lateral_mae": ("pose", "lateral_mae"),
            "heading_mae_rad": ("pose", "heading_mae_rad"),
        }
        for output_name, (section, metric) in mapping.items():
            if section in comparisons["fit"]:
                row[output_name] = float(comparisons["fit"][section][metric])
            if section in comparisons["history"]:
                row[f"history_{output_name}"] = float(comparisons["history"][section][metric])
        audited.append(row)

    if missing:
        raise SystemExit(f"{len(missing)} output ids are absent from the exact reference manifest")
    result = {
        "protocol": "step1-relative-measurement-audit-v1",
        "coordinates": {
            "distance": "relative terminal-progress normalized",
            "pose": "arc-relative translation with metric heading",
        },
        "num_input": len(scores),
        "num_audited": len(audited),
        "unavailable": dict(sorted(unavailable.items())),
        "by_status": {
            status: summarize([row for row in audited if row["status"] == status])
            for status in ("explained", "weak")
        },
        "explained_by_stratum": {
            stratum: summarize([
                row for row in audited
                if row["status"] == "explained" and row["stratum"] == stratum
            ])
            for stratum in sorted({row["stratum"] for row in audited})
        },
    }
    for status in ("explained", "weak"):
        subset = [row for row in audited if row["status"] == status]
        result["by_status"][status]["endpoint_direction_and_rank"] = {
            "lateral": direction_and_rank(subset, "lateral", 0.05),
            "yaw": direction_and_rank(subset, "yaw", 0.01),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

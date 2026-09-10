#!/usr/bin/env python3
"""Validate decoder-free flow structure on real and paired generated clips."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DESCRIPTORS = (
    "divergence",
    "curl",
    "horizontal_flow_center",
    "left_right_horizontal_contrast",
    "foe_x",
)
TARGETS = ("terminal_x", "terminal_y", "terminal_yaw")


def _read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _manifest_by_key(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(paths):
        key = (str(row.get("source_key") or ""), str(row.get("branch_role") or "real"))
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = row
    return result


def _logged_trajectory(record: dict[str, Any]) -> np.ndarray:
    source = Path(str(record["lineage"]["source_sample"]))
    with source.open("rb") as stream:
        payload = pickle.load(stream)
    future = payload.get("future_trajectory") or []
    source_times = np.asarray(
        (payload.get("metadata") or {}).get("future_times_s")
        or [0.5 * (index + 1) for index in range(len(future))],
        dtype=np.float64,
    )
    target_times = np.asarray(record["future_times_s"], dtype=np.float64)
    poses = np.asarray([item["pose"] for item in future], dtype=np.float64)
    indices = [int(np.argmin(np.abs(source_times - target))) for target in target_times]
    if any(abs(source_times[index] - target) > 0.08 for index, target in zip(indices, target_times)):
        raise ValueError(f"no logged pose near requested time for {source}")
    return poses[indices]


def _value(row: dict[str, Any], descriptor: str) -> float | None:
    if not row.get("input_available"):
        return None
    if descriptor == "foe_x":
        value = (row.get("foe") or {}).get("foe_normalized_xy")
        value = None if value is None else value[0]
    else:
        value = row.get(descriptor)
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _profile_values(record: dict[str, Any], descriptor: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in record["flow_structure"]["rows"]:
        value = _value(row, descriptor)
        if value is not None:
            values[int(row["interval_index"])] = value
    return values


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
    first: np.ndarray,
    second: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    estimate = _spearman(first, second)
    if estimate is None:
        return {"n": int(len(first)), "rho": None, "ci95": None}
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(draws):
        index = rng.integers(0, len(first), size=len(first))
        value = _spearman(first[index], second[index])
        if value is not None:
            samples.append(value)
    return {
        "n": int(len(first)),
        "rho": estimate,
        "ci95": (
            [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
            if samples else None
        ),
    }


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


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = [row["flow_structure"] for row in rows]
    intervals = [item for profile in profiles for item in profile["rows"]]
    statuses = Counter(str(profile["status"]) for profile in profiles)
    available = [item for item in intervals if item.get("input_available")]
    return {
        "records": len(rows),
        "status_counts": dict(statuses),
        "all_intervals_available_fraction": float(np.mean([
            profile["measurement_available"] for profile in profiles
        ])) if profiles else None,
        "available_interval_fraction": float(np.mean([
            item["input_available"] for item in intervals
        ])) if intervals else None,
        "median_roi_support_fraction": (
            float(np.median([item["roi_support_fraction"] for item in intervals]))
            if intervals else None
        ),
        "median_affine_explained_fraction": (
            float(np.median([item["affine_explained_fraction"] for item in available]))
            if available else None
        ),
        "median_structure_confidence": (
            float(np.median([item["structure_confidence"] for item in available]))
            if available else None
        ),
    }


def _real_validation(
    rows: list[dict[str, Any]],
    manifests: dict[tuple[str, str], dict[str, Any]],
    *,
    bootstrap_draws: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["source_key"]), "real")
        manifest = manifests[key]
        terminal = _logged_trajectory(manifest)[-1]
        sample: dict[str, Any] = {
            "stratum": str(row.get("stratum") or "unknown"),
            "terminal_x": float(terminal[0]),
            "terminal_y": float(terminal[1]),
            "terminal_yaw": float(terminal[2]),
        }
        for descriptor in DESCRIPTORS:
            values = list(_profile_values(row, descriptor).values())
            sample[descriptor] = float(np.median(values)) if values else None
        samples.append(sample)
    correlations: dict[str, Any] = {}
    orientations: dict[str, int] = {}
    for descriptor_index, descriptor in enumerate(DESCRIPTORS):
        correlations[descriptor] = {}
        for target_index, target in enumerate(TARGETS):
            paired = [(sample[descriptor], sample[target]) for sample in samples if sample[descriptor] is not None]
            first = np.asarray([item[0] for item in paired], dtype=np.float64)
            second = np.asarray([item[1] for item in paired], dtype=np.float64)
            correlations[descriptor][target] = _bootstrap_spearman(
                first,
                second,
                draws=bootstrap_draws,
                seed=1701 + descriptor_index * 10 + target_index,
            )
        preferred_target = {
            "divergence": "terminal_x",
            "curl": "terminal_yaw",
            "horizontal_flow_center": "terminal_y",
            "left_right_horizontal_contrast": "terminal_yaw",
            "foe_x": "terminal_yaw",
        }[descriptor]
        rho = correlations[descriptor][preferred_target]["rho"]
        orientations[descriptor] = 1 if rho is None or rho >= 0.0 else -1
    by_stratum: dict[str, Any] = {}
    for stratum_index, stratum in enumerate(sorted({sample["stratum"] for sample in samples})):
        subset = [sample for sample in samples if sample["stratum"] == stratum]
        by_stratum[stratum] = {}
        for descriptor_index, descriptor in enumerate(DESCRIPTORS):
            by_stratum[stratum][descriptor] = {}
            for target_index, target in enumerate(TARGETS):
                paired = [
                    (sample[descriptor], sample[target])
                    for sample in subset
                    if sample[descriptor] is not None
                ]
                first = np.asarray([item[0] for item in paired], dtype=np.float64)
                second = np.asarray([item[1] for item in paired], dtype=np.float64)
                by_stratum[stratum][descriptor][target] = _bootstrap_spearman(
                    first,
                    second,
                    draws=bootstrap_draws,
                    seed=3701 + stratum_index * 100 + descriptor_index * 10 + target_index,
                )
    coverage_by_stratum = {
        stratum: _coverage([row for row in rows if str(row.get("stratum") or "unknown") == stratum])
        for stratum in sorted({str(row.get("stratum") or "unknown") for row in rows})
    }
    return {
        "coverage": _coverage(rows),
        "coverage_by_stratum": coverage_by_stratum,
        "spearman": correlations,
        "spearman_by_stratum": by_stratum,
    }, orientations


def _generated_pairs(
    rows: list[dict[str, Any]],
    manifests: dict[tuple[str, str], dict[str, Any]],
    orientations: dict[str, int],
    *,
    bootstrap_draws: int,
) -> dict[str, Any]:
    by_key = {(str(row["source_key"]), str(row["branch_role"])): row for row in rows}
    sources = sorted({key for key, role in by_key if role == "left"} & {key for key, role in by_key if role == "right"})
    descriptor_results: dict[str, Any] = {}
    for descriptor_index, descriptor in enumerate(DESCRIPTORS):
        pairs: list[dict[str, Any]] = []
        direction_hits = 0
        direction_total = 0
        for source in sources:
            left_values = _profile_values(by_key[(source, "left")], descriptor)
            right_values = _profile_values(by_key[(source, "right")], descriptor)
            common = sorted(set(left_values) & set(right_values))
            if not common:
                continue
            measured_delta = float(np.median([left_values[index] - right_values[index] for index in common]))
            left_action = np.asarray(manifests[(source, "left")]["action_trajectory"], dtype=np.float64)
            right_action = np.asarray(manifests[(source, "right")]["action_trajectory"], dtype=np.float64)
            target_name = {
                "divergence": "terminal_x",
                "curl": "terminal_yaw",
                "horizontal_flow_center": "terminal_y",
                "left_right_horizontal_contrast": "terminal_yaw",
                "foe_x": "terminal_yaw",
            }[descriptor]
            target_column = {"terminal_x": 0, "terminal_y": 1, "terminal_yaw": 2}[target_name]
            action_delta = float(left_action[-1, target_column] - right_action[-1, target_column])
            pairs.append({
                "measured_delta": measured_delta,
                "action_delta": action_delta,
                "common_intervals": len(common),
                "stratum": str(by_key[(source, "left")].get("stratum") or "unknown"),
            })
            if abs(action_delta) > {0: 0.1, 1: 0.05, 2: 0.01}[target_column] and abs(measured_delta) > 1e-12:
                direction_total += 1
                direction_hits += int(
                    np.sign(orientations[descriptor] * measured_delta) == np.sign(action_delta)
                )
        measured = np.asarray([item["measured_delta"] for item in pairs], dtype=np.float64)
        action = np.asarray([item["action_delta"] for item in pairs], dtype=np.float64)
        result = _bootstrap_spearman(
            orientations[descriptor] * measured,
            action,
            draws=bootstrap_draws,
            seed=2701 + descriptor_index,
        )
        result.update({
            "real_calibrated_sign": orientations[descriptor],
            "comparable_pairs": len(pairs),
            "direction_pairs": direction_total,
            "direction_hits": direction_hits,
            "direction_accuracy": direction_hits / direction_total if direction_total else None,
            "direction_accuracy_ci95": _wilson_interval(direction_hits, direction_total),
            "command_label_direction_hits": int(np.sum(orientations[descriptor] * measured > 0.0)),
            "command_label_direction_accuracy": (
                float(np.mean(orientations[descriptor] * measured > 0.0)) if len(measured) else None
            ),
            "command_label_direction_accuracy_ci95": _wilson_interval(
                int(np.sum(orientations[descriptor] * measured > 0.0)), len(measured)
            ),
            "target": {
                "divergence": "terminal_x",
                "curl": "terminal_yaw",
                "horizontal_flow_center": "terminal_y",
                "left_right_horizontal_contrast": "terminal_yaw",
                "foe_x": "terminal_yaw",
            }[descriptor],
        })
        by_stratum: dict[str, Any] = {}
        for stratum_index, stratum in enumerate(sorted({item["stratum"] for item in pairs})):
            subset = [item for item in pairs if item["stratum"] == stratum]
            subset_measured = np.asarray(
                [item["measured_delta"] for item in subset], dtype=np.float64
            )
            subset_action = np.asarray(
                [item["action_delta"] for item in subset], dtype=np.float64
            )
            threshold = {"terminal_x": 0.1, "terminal_y": 0.05, "terminal_yaw": 0.01}[
                result["target"]
            ]
            directional = (
                (np.abs(subset_action) > threshold)
                & (np.abs(subset_measured) > 1e-12)
            )
            subset_hits = int(np.sum(
                np.sign(orientations[descriptor] * subset_measured[directional])
                == np.sign(subset_action[directional])
            ))
            by_stratum[stratum] = {
                **_bootstrap_spearman(
                    orientations[descriptor] * subset_measured,
                    subset_action,
                    draws=bootstrap_draws,
                    seed=4701 + descriptor_index * 10 + stratum_index,
                ),
                "direction_pairs": int(np.sum(directional)),
                "direction_hits": subset_hits,
                "direction_accuracy": (
                    subset_hits / int(np.sum(directional)) if np.sum(directional) else None
                ),
                "direction_accuracy_ci95": _wilson_interval(subset_hits, int(np.sum(directional))),
                "command_label_direction_hits": int(np.sum(
                    orientations[descriptor] * subset_measured > 0.0
                )),
                "command_label_direction_accuracy": float(np.mean(
                    orientations[descriptor] * subset_measured > 0.0
                )),
                "command_label_direction_accuracy_ci95": _wilson_interval(
                    int(np.sum(orientations[descriptor] * subset_measured > 0.0)),
                    len(subset_measured),
                ),
            }
        result["by_stratum"] = by_stratum
        result["by_minimum_common_intervals"] = {}
        for minimum in range(1, 5):
            subset = [item for item in pairs if item["common_intervals"] >= minimum]
            subset_measured = np.asarray(
                [item["measured_delta"] for item in subset], dtype=np.float64
            )
            subset_action = np.asarray(
                [item["action_delta"] for item in subset], dtype=np.float64
            )
            hits = int(np.sum(orientations[descriptor] * subset_measured > 0.0))
            result["by_minimum_common_intervals"][str(minimum)] = {
                **_bootstrap_spearman(
                    orientations[descriptor] * subset_measured,
                    subset_action,
                    draws=bootstrap_draws,
                    seed=5701 + descriptor_index * 10 + minimum,
                ),
                "command_label_direction_hits": hits,
                "command_label_direction_accuracy": hits / len(subset) if subset else None,
                "command_label_direction_accuracy_ci95": _wilson_interval(hits, len(subset)),
            }
        descriptor_results[descriptor] = result
    complete_pairs = sum(
        (source, "left") in by_key and (source, "right") in by_key
        and by_key[(source, "left")]["flow_structure"]["measurement_available"]
        and by_key[(source, "right")]["flow_structure"]["measurement_available"]
        for source in sources
    )
    return {
        "coverage": _coverage(rows),
        "coverage_by_stratum": {
            stratum: _coverage([
                row for row in rows if str(row.get("stratum") or "unknown") == stratum
            ])
            for stratum in sorted({str(row.get("stratum") or "unknown") for row in rows})
        },
        "complete_pairs": int(complete_pairs),
        "common_interval_histogram": {
            str(count): int(sum(
                len(
                    set(_profile_values(by_key[(source, "left")], "horizontal_flow_center"))
                    & set(_profile_values(by_key[(source, "right")], "horizontal_flow_center"))
                ) == count
                for source in sources
            ))
            for count in range(5)
        },
        "pairs_with_any_common_interval": int(max(
            (value["comparable_pairs"] for value in descriptor_results.values()), default=0
        )),
        "command_alignment_diagnostic": descriptor_results,
        "warning": "Command/action alignment is not ground-truth video correctness or policy ranking.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-output", type=Path, action="append", required=True)
    parser.add_argument("--real-manifest", type=Path, action="append", required=True)
    parser.add_argument("--generated-output", type=Path, action="append", required=True)
    parser.add_argument("--generated-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    real_rows = _read_jsonl(args.real_output)
    generated_rows = _read_jsonl(args.generated_output)
    real_manifests = _manifest_by_key(args.real_manifest)
    generated_manifests = _manifest_by_key(args.generated_manifest)
    real, orientations = _real_validation(
        real_rows,
        real_manifests,
        bootstrap_draws=args.bootstrap_draws,
    )
    report = {
        "protocol": "iac-flow-structure-validation-v1",
        "metric_reconstruction_used": False,
        "real": real,
        "generated": _generated_pairs(
            generated_rows,
            generated_manifests,
            orientations,
            bootstrap_draws=args.bootstrap_draws,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

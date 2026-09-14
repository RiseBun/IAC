#!/usr/bin/env python3
"""Source-disjoint calibration of a candidate-blind visual stop token.

The stop token means that the ego path remains within a small radius for the
whole four-second future.  It is intentionally separate from turn direction:
low-magnitude flow must not be removed by the direction reader's motion floor.
Generated-video results are visual/native-action alignment, not visual ground
truth accuracy.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FEATURES = (
    "max_q25",
    "median_q25",
    "max_median",
    "median_median",
)


def read_many(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        paths = sorted(glob.glob(pattern)) or [pattern]
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def sample_id(row: dict[str, Any], *, real: bool = False) -> str:
    if real:
        return f"{row['source_key']}::real"
    return str(row.get("sample_id") or f"{row['source_key']}::{row.get('branch_role', 'unknown')}")


def trajectory(row: dict[str, Any]) -> np.ndarray | None:
    value = row.get("action_trajectory") or row.get("trajectory")
    if not isinstance(value, list) or not value:
        return None
    if isinstance(value[0], dict):
        value = [item.get("pose", item) for item in value]
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] < 2:
        return None
    return result


def action_is_stop(row: dict[str, Any], radius_m: float) -> bool | None:
    path = trajectory(row)
    if path is None:
        return None
    extent = float(np.max(np.linalg.norm(path[:, :2], axis=1)))
    return extent <= radius_m


def motion_features(output: dict[str, Any]) -> dict[str, float] | None:
    intervals = output.get("intervals") or []
    if len(intervals) != 4:
        return None
    q25, median = [], []
    for item in intervals:
        if float(item.get("motion_support_fraction", 0.0)) < 0.9:
            return None
        a = item.get("flow_magnitude_q25_px")
        b = item.get("flow_magnitude_median_px")
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            return None
        q25.append(float(a))
        median.append(float(b))
    return {
        "max_q25": float(np.max(q25)),
        "median_q25": float(np.median(q25)),
        "max_median": float(np.max(median)),
        "median_median": float(np.median(median)),
    }


def join(manifests: list[dict[str, Any]], outputs: list[dict[str, Any]], *, real: bool, radius_m: float) -> list[dict[str, Any]]:
    index = {sample_id(row, real=real): row for row in manifests}
    rows = []
    for output in outputs:
        key = str(output.get("sample_id"))
        manifest = index.get(key)
        if manifest is None:
            continue
        label = action_is_stop(manifest, radius_m)
        if label is None:
            continue
        rows.append({
            "source_key": str(manifest["source_key"]),
            "sample_id": key,
            "label": bool(label),
            "features": motion_features(output),
        })
    return rows


def assignments(rows: list[dict[str, Any]], folds: int, seed: int) -> dict[str, int]:
    grouped: dict[bool, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row["source_key"])
    rng = np.random.default_rng(seed)
    result: dict[str, int] = {}
    for label in (False, True):
        sources = sorted(set(grouped[label]))
        rng.shuffle(sources)
        for index, source in enumerate(sources):
            result[source] = index % folds
    return result


def fit_feature(rows: list[dict[str, Any]], feature: str) -> dict[str, float | str] | None:
    usable = [row for row in rows if row["features"] is not None]
    labels = np.asarray([row["label"] for row in usable], dtype=bool)
    if not labels.any() or labels.all():
        return None
    values = np.asarray([row["features"][feature] for row in usable], dtype=np.float64)
    candidates = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 201)))
    best = None
    for threshold in candidates:
        predictions = values <= threshold
        stop_recall = float(np.mean(predictions[labels]))
        moving_recall = float(np.mean(~predictions[~labels]))
        balanced = 0.5 * (stop_recall + moving_recall)
        accuracy = float(np.mean(predictions == labels))
        objective = (balanced, accuracy, -float(threshold))
        if best is None or objective > best[0]:
            best = (objective, float(threshold))
    assert best is not None
    return {"feature": feature, "threshold": best[1], "balanced_accuracy": best[0][0], "accuracy": best[0][1]}


def evaluate(row: dict[str, Any], adapter: dict[str, float | str]) -> dict[str, Any]:
    if row["features"] is None:
        return {**row, "prediction": None, "hit": None, "confidence": None}
    value = float(row["features"][str(adapter["feature"])])
    threshold = float(adapter["threshold"])
    prediction = value <= threshold
    confidence = abs(math.log((value + 1e-6) / (threshold + 1e-6)))
    return {**row, "value": value, "prediction": prediction, "hit": prediction == row["label"], "confidence": confidence}


def wilson(hits: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = hits / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [centre - half, centre + half]


def summarize(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    scored = [row for row in rows if row.get("hit") is not None]
    result: dict[str, Any] = {
        "scored": len(scored),
        "coverage": len(scored) / total if total else 0.0,
        "accuracy": None,
        "balanced_accuracy": None,
        "classes": {},
        "wilson_ci95": None,
    }
    if not scored:
        return result
    hits = sum(bool(row["hit"]) for row in scored)
    recalls = []
    for label, name in ((True, "stop"), (False, "moving")):
        subset = [row for row in scored if row["label"] == label]
        class_hits = sum(bool(row["hit"]) for row in subset)
        recall = class_hits / len(subset) if subset else None
        if recall is not None:
            recalls.append(recall)
        result["classes"][name] = {
            "n": len(subset),
            "recall": recall,
            "wilson_ci95": wilson(class_hits, len(subset)),
        }
    result["accuracy"] = hits / len(scored)
    # A one-class submission can obtain a nearly perfect raw accuracy by
    # always predicting "moving".  Do not call that a balanced MAS.stop score.
    result["balanced_accuracy"] = float(np.mean(recalls)) if len(recalls) == 2 else None
    result["wilson_ci95"] = wilson(hits, len(scored))
    return result


def controls(rows: list[dict[str, Any]], draws: int = 20000) -> dict[str, Any]:
    scored = [row for row in rows if row.get("prediction") is not None]
    labels = np.asarray([row["label"] for row in scored], dtype=bool)
    predictions = np.asarray([row["prediction"] for row in scored], dtype=bool)

    def balanced(reference: np.ndarray, predicted: np.ndarray) -> float | None:
        recalls = [float(np.mean(predicted[reference == value] == value)) for value in (False, True) if np.any(reference == value)]
        return float(np.mean(recalls)) if len(recalls) == 2 else None

    rng = np.random.default_rng(20260914)
    shuffled = []
    for _ in range(draws):
        shuffled.append(balanced(rng.permutation(labels), predictions))
    values = np.asarray([value for value in shuffled if value is not None], dtype=np.float64)
    return {
        "always_moving_balanced_accuracy": balanced(labels, np.zeros_like(labels)),
        "always_stop_balanced_accuracy": balanced(labels, np.ones_like(labels)),
        "identity_shuffle_balanced_accuracy_mean": float(np.mean(values)),
        "identity_shuffle_q95": float(np.quantile(values, 0.95)),
    }


def feature_oof(rows: list[dict[str, Any]], feature: str, folds: int, seed: int) -> list[dict[str, Any]]:
    split = assignments(rows, folds, seed)
    output = []
    for held_out in range(folds):
        train = [row for row in rows if split[row["source_key"]] != held_out]
        test = [row for row in rows if split[row["source_key"]] == held_out]
        adapter = fit_feature(train, feature)
        if adapter is not None:
            output.extend(evaluate(row, adapter) for row in test)
    return output


def choose_feature(rows: list[dict[str, Any]], folds: int, seed: int) -> tuple[str, dict[str, Any]]:
    reports = {}
    for feature in FEATURES:
        evaluated = feature_oof(rows, feature, folds, seed)
        reports[feature] = summarize(evaluated, len(rows))
    chosen = max(FEATURES, key=lambda item: (reports[item]["balanced_accuracy"] or -1.0, reports[item]["accuracy"] or -1.0))
    return chosen, reports


def nested_oof(rows: list[dict[str, Any]], folds: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split = assignments(rows, folds, seed)
    output, choices = [], []
    for held_out in range(folds):
        train = [row for row in rows if split[row["source_key"]] != held_out]
        test = [row for row in rows if split[row["source_key"]] == held_out]
        feature, reports = choose_feature(train, max(3, folds - 1), seed + 1009 + held_out)
        adapter = fit_feature(train, feature)
        choices.append({"outer_fold": held_out, "feature": feature, "adapter": adapter, "inner_reports": reports})
        if adapter is not None:
            output.extend(evaluate(row, adapter) for row in test)
    return output, choices


def selective_curve(rows: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    scored = sorted((row for row in rows if row.get("hit") is not None), key=lambda row: row["confidence"], reverse=True)
    output = []
    for retain in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        count = max(1, int(math.floor(len(scored) * retain)))
        item = summarize(scored[:count], total)
        item["minimum_confidence"] = float(scored[count - 1]["confidence"])
        output.append(item)
    return output


def source_bootstrap(rows: list[dict[str, Any]], draws: int = 20000) -> list[float] | None:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row.get("hit") is not None:
            grouped[row["source_key"]].append(bool(row["hit"]))
    if not grouped:
        return None
    values = np.asarray([np.mean(items) for items in grouped.values()], dtype=np.float64)
    rng = np.random.default_rng(20260914)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--generated-manifest", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-radius-m", type=float, default=0.25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    measurements = read_many(args.outputs)
    real_manifest = read_many([args.benchmark_manifest])
    generated_manifest = read_many([args.generated_manifest])
    real = join(real_manifest, measurements, real=True, radius_m=args.stop_radius_m)
    generated = join(generated_manifest, measurements, real=False, radius_m=args.stop_radius_m)

    real_oof, choices = nested_oof(real, args.folds, args.seed)
    final_feature, feature_reports = choose_feature(real, args.folds, args.seed)
    final_adapter = fit_feature(real, final_feature)
    assert final_adapter is not None
    generated_eval = [evaluate(row, final_adapter) for row in generated]

    payload = {
        "protocol": "iac-mas-stop-token-pilot-v1",
        "definition": "stop iff the supplied trajectory remains within stop_radius_m of the origin over the four-second future",
        "stop_radius_m": args.stop_radius_m,
        "candidate_blind": True,
        "metric_reconstruction_used": False,
        "logged_real": {
            "records": len(real),
            "label_counts": dict(Counter("stop" if row["label"] else "moving" for row in real)),
            "nested_source_disjoint": summarize(real_oof, len(real)),
            "controls": controls(real_oof),
            "selective_curve_exploratory": selective_curve(real_oof, len(real)),
            "outer_choices": choices,
            "feature_oof_reports": feature_reports,
        },
        "frozen_adapter": final_adapter,
        "drivewam_generated": {
            "records": len(generated),
            "action_label_counts": dict(Counter("stop" if row["label"] else "moving" for row in generated)),
            "alignment": summarize(generated_eval, len(generated)),
            "score_status": (
                "scored"
                if min(Counter(row["label"] for row in generated_eval).values(), default=0) >= 20
                and len(Counter(row["label"] for row in generated_eval)) == 2
                else "diagnostic_one_action_class_missing"
            ),
            "source_bootstrap_ci95": source_bootstrap(generated_eval),
            "interpretation": "visual/native-action stop alignment; not visual ground-truth accuracy",
        },
        "claim_boundary": "whole-horizon stationary-vs-moving token only; braking-to-terminal-stop and absolute speed remain unavailable",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

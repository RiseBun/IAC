#!/usr/bin/env python3
"""Audit selective accuracy of the corrected SE(2)-yaw token reader.

The confidence is candidate/action blind: it is only the normalized distance
of decoded endpoint yaw from a straight/turn decision boundary learned on
logged-real video.  Generated-video results are alignment with the native
action, not visual ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CLASSES = ("left", "right", "straight")


def read_many(patterns: list[str]) -> list[dict]:
    rows: list[dict] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern)) if not Path(pattern).is_absolute() else []
        paths = matches or [Path(pattern)]
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def endpoint_yaw(value: object) -> float | None:
    if not isinstance(value, list) or not value:
        return None
    last = value[-1]
    if isinstance(last, dict):
        last = last.get("pose")
    if not isinstance(last, list) or len(last) < 3:
        return None
    yaw = float(last[2])
    return yaw if math.isfinite(yaw) else None


def path_extent(value: object) -> float | None:
    if not isinstance(value, list) or not value:
        return None
    points = [item.get("pose", item) if isinstance(item, dict) else item for item in value]
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2 or not np.isfinite(array[:, :2]).all():
        return None
    return float(np.max(np.linalg.norm(array[:, :2], axis=1)))


def token(yaw: float, deadband: float) -> str:
    if abs(yaw) < deadband:
        return "straight"
    return "left" if yaw > 0 else "right"


def decoded(row: dict) -> tuple[float | None, bool]:
    decoder = row.get("decoder") or {}
    yaw = endpoint_yaw(decoder.get("trajectory"))
    status = row.get("motion_explanation_status", decoder.get("motion_explanation_status"))
    return yaw, status == "explained"


def fit_adapter(rows: list[dict], action_deadband: float) -> dict:
    usable = [row for row in rows if row["available"]]
    magnitudes = np.asarray([abs(row["visual_yaw"]) for row in usable], dtype=float)
    candidates = np.unique(np.concatenate(([0.0], np.quantile(magnitudes, np.linspace(0, 0.95, 96)))))
    best = None
    for orientation in (-1, 1):
        for threshold in candidates:
            labels = [token(row["action_yaw"], action_deadband) for row in usable]
            predictions = [token(orientation * row["visual_yaw"], float(threshold)) for row in usable]
            recalls = []
            for name in CLASSES:
                hits = [predictions[i] == name for i, label in enumerate(labels) if label == name]
                if hits:
                    recalls.append(float(np.mean(hits)))
            macro = float(np.mean(recalls))
            accuracy = float(np.mean(np.asarray(labels) == np.asarray(predictions)))
            objective = (macro, accuracy, -float(threshold))
            if best is None or objective > best[0]:
                best = (objective, orientation, float(threshold))
    assert best is not None
    return {
        "orientation": best[1],
        "visual_deadband": best[2],
        "action_deadband": action_deadband,
        "fit_macro_accuracy": best[0][0],
        "fit_accuracy": best[0][1],
    }


def evaluate(row: dict, adapter: dict) -> dict:
    label = token(row["action_yaw"], adapter["action_deadband"])
    if not row["available"]:
        return {**row, "label": label, "prediction": None, "hit": None, "confidence": None}
    signed_yaw = adapter["orientation"] * row["visual_yaw"]
    prediction = token(signed_yaw, adapter["visual_deadband"])
    threshold = max(adapter["visual_deadband"], 1e-12)
    confidence = abs(abs(signed_yaw) - adapter["visual_deadband"]) / threshold
    return {
        **row,
        "label": label,
        "prediction": prediction,
        "hit": prediction == label,
        "confidence": float(confidence),
    }


def wilson(hits: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [centre - half, centre + half]


def summary(rows: list[dict], total: int) -> dict:
    if not rows:
        return {"scored": 0, "absolute_coverage": 0.0, "accuracy": None, "macro_accuracy": None, "per_class": {}, "wilson_ci95": None}
    hits = sum(bool(row["hit"]) for row in rows)
    per_class = {}
    for name in CLASSES:
        subset = [row for row in rows if row.get("label") == name]
        per_class[name] = {
            "n": len(subset),
            "accuracy": (sum(bool(row["hit"]) for row in subset) / len(subset)) if subset else None,
        }
    class_accuracies = [item["accuracy"] for item in per_class.values() if item["accuracy"] is not None]
    return {
        "scored": len(rows),
        "absolute_coverage": len(rows) / total,
        "accuracy": hits / len(rows),
        "macro_accuracy": float(np.mean(class_accuracies)),
        "per_class": per_class,
        "wilson_ci95": wilson(hits, len(rows)),
        "minimum_confidence": min(row["confidence"] for row in rows),
    }


def selective_curve(rows: list[dict], total: int) -> list[dict]:
    scored = sorted((row for row in rows if row["hit"] is not None), key=lambda row: row["confidence"], reverse=True)
    curve = []
    for retain in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        count = max(1, int(math.floor(retain * len(scored))))
        item = summary(scored[:count], total)
        item["retained_fraction_of_available"] = count / len(scored)
        curve.append(item)
    return curve


def largest_gate(rows: list[dict], total: int, lower_bound: float, minimum: int = 30) -> dict | None:
    scored = sorted((row for row in rows if row["hit"] is not None), key=lambda row: row["confidence"], reverse=True)
    best = None
    for count in range(minimum, len(scored) + 1):
        candidate = summary(scored[:count], total)
        if candidate["wilson_ci95"][0] >= lower_bound:
            best = candidate
    return best


def build_real(manifest: list[dict], outputs: list[dict]) -> list[dict]:
    by_source = {str(row["source_key"]): row for row in manifest}
    result = []
    for output in outputs:
        sample_id = str(output.get("sample_id", ""))
        if not sample_id.endswith("::real"):
            continue
        source = sample_id.rsplit("::", 1)[0]
        match = by_source.get(source)
        visual_yaw, available = decoded(output)
        action_path = (match or {}).get("trajectory")
        action_yaw = endpoint_yaw(action_path)
        action_extent = path_extent(action_path)
        if match is not None and action_yaw is not None:
            result.append({"source_key": source, "sample_id": sample_id, "visual_yaw": visual_yaw, "action_yaw": action_yaw, "action_extent_m": action_extent, "available": available})
    return result


def build_generated(manifests: list[dict], outputs: list[dict]) -> list[dict]:
    by_sample = {str(row["sample_id"]): row for row in manifests}
    result = []
    for output in outputs:
        sample_id = str(output.get("sample_id", ""))
        match = by_sample.get(sample_id)
        visual_yaw, available = decoded(output)
        action_path = (match or {}).get("action_trajectory")
        action_yaw = endpoint_yaw(action_path)
        action_extent = path_extent(action_path)
        if match is not None and action_yaw is not None:
            result.append({"source_key": str(match["source_key"]), "sample_id": sample_id, "visual_yaw": visual_yaw, "action_yaw": action_yaw, "action_extent_m": action_extent, "available": available})
    return result


def stratified_folds(rows: list[dict], count: int, seed: int, action_deadband: float) -> dict[str, int]:
    """Assign sources to balanced folds without leaking visual evidence."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[token(row["action_yaw"], action_deadband)].append(row["source_key"])
    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    for name in CLASSES:
        sources = sorted(set(grouped[name]))
        rng.shuffle(sources)
        for index, source in enumerate(sources):
            assignment[source] = index % count
    return assignment


def cross_validated(rows: list[dict], count: int, seed: int, action_deadband: float) -> tuple[list[dict], list[dict]]:
    assignment = stratified_folds(rows, count, seed, action_deadband)
    evaluated: list[dict] = []
    adapters: list[dict] = []
    for held_out in range(count):
        train = [row for row in rows if assignment[row["source_key"]] != held_out]
        test = [row for row in rows if assignment[row["source_key"]] == held_out]
        adapter = fit_adapter(train, action_deadband)
        adapters.append(adapter)
        evaluated.extend(evaluate(row, adapter) for row in test)
    return evaluated, adapters


def nested_selective(rows: list[dict], outer_folds: int, seed: int, action_deadband: float) -> list[dict]:
    """Evaluate a confidence gate selected without seeing each outer test fold."""
    assignment = stratified_folds(rows, outer_folds, seed, action_deadband)
    selected: list[dict] = []
    for held_out in range(outer_folds):
        train = [row for row in rows if assignment[row["source_key"]] != held_out]
        test = [row for row in rows if assignment[row["source_key"]] == held_out]
        inner, _ = cross_validated(train, max(3, outer_folds - 1), seed + 1009 + held_out, action_deadband)
        gate = largest_gate(inner, len(train), 0.75, minimum=25)
        if gate is None:
            continue
        adapter = fit_adapter(train, action_deadband)
        threshold = gate["minimum_confidence"]
        selected.extend(row for row in (evaluate(item, adapter) for item in test) if row["hit"] is not None and row["confidence"] >= threshold)
    return selected


def source_bootstrap(rows: list[dict], draws: int = 20000) -> list[float] | None:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        groups[row["source_key"]].append(bool(row["hit"]))
    if not groups:
        return None
    values = np.asarray([np.mean(value) for value in groups.values()])
    rng = np.random.default_rng(20260914)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--real-outputs", nargs="+", required=True)
    parser.add_argument("--generated-manifests", nargs="+", required=True)
    parser.add_argument("--generated-outputs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-yaw-deadband", type=float, default=0.012)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--repeat-splits", type=int, default=30)
    parser.add_argument("--backend-name", default="corrected_se2_endpoint_yaw")
    parser.add_argument("--direction-min-path-extent-m", type=float, default=0.0)
    args = parser.parse_args()

    real_all = build_real(read_many([args.benchmark_manifest]), read_many(args.real_outputs))
    generated_all = build_generated(read_many(args.generated_manifests), read_many(args.generated_outputs))
    real = [row for row in real_all if row.get("action_extent_m") is not None and row["action_extent_m"] > args.direction_min_path_extent_m]
    generated = [row for row in generated_all if row.get("action_extent_m") is not None and row["action_extent_m"] > args.direction_min_path_extent_m]
    real_oof, adapters = cross_validated(real, args.folds, args.seed, args.action_yaw_deadband)
    nested = nested_selective(real, args.folds, args.seed, args.action_yaw_deadband)
    repeated = []
    for offset in range(args.repeat_splits):
        seed = args.seed + offset
        oof, _ = cross_validated(real, args.folds, seed, args.action_yaw_deadband)
        base = summary([row for row in oof if row["hit"] is not None], len(real))
        selected = nested_selective(real, args.folds, seed, args.action_yaw_deadband)
        selective = summary(selected, len(real))
        repeated.append({"seed": seed, "base": base, "nested_selective": selective})
    final_adapter = fit_adapter(real, args.action_yaw_deadband)
    generated_eval = [evaluate(row, final_adapter) for row in generated]

    real_scored = [row for row in real_oof if row["hit"] is not None]
    generated_scored = [row for row in generated_eval if row["hit"] is not None]
    promotion_gate = largest_gate(real_oof, len(real), 0.75)
    frozen_threshold = promotion_gate["minimum_confidence"] if promotion_gate else None
    generated_at_gate = [] if frozen_threshold is None else [row for row in generated_scored if row["confidence"] >= frozen_threshold]

    nested_accuracy_values = [row["nested_selective"]["accuracy"] for row in repeated if row["nested_selective"]["accuracy"] is not None]
    payload = {
        "protocol": "iac-whole-clip-yaw-confidence-audit-v1",
        "visual_backend": args.backend_name,
        "confidence_definition": "abs(abs(oriented_endpoint_yaw)-real_fitted_deadband)/real_fitted_deadband",
        "confidence_is_action_blind": True,
        "logged_real": {
            "records": len(real),
            "declared_records_before_motion_eligibility": len(real_all),
            "available": len(real_scored),
            "base": summary(real_scored, len(real)),
            "selective_curve": selective_curve(real_oof, len(real)),
            "largest_gate_with_wilson_lower_at_least_0_75": promotion_gate,
            "nested_selective_validation": summary(nested, len(real)),
            "split_sensitivity": {
                "repeats": args.repeat_splits,
                "base_accuracy_q025_q50_q975": [float(value) for value in np.quantile([row["base"]["accuracy"] for row in repeated], [0.025, 0.5, 0.975])],
                "nested_coverage_q025_q50_q975": [float(value) for value in np.quantile([row["nested_selective"]["absolute_coverage"] for row in repeated], [0.025, 0.5, 0.975])],
                "nested_gate_success_fraction": len(nested_accuracy_values) / max(len(repeated), 1),
                "nested_accuracy_q025_q50_q975": (
                    [float(value) for value in np.quantile(nested_accuracy_values, [0.025, 0.5, 0.975])]
                    if nested_accuracy_values else None
                ),
            },
        },
        "frozen_adapter": final_adapter,
        "drivewam_generated": {
            "records": len(generated),
            "declared_records_before_motion_eligibility": len(generated_all),
            "available": len(generated_scored),
            "base": summary(generated_scored, len(generated)),
            "base_source_bootstrap_ci95": source_bootstrap(generated_scored),
            "at_real_frozen_confidence_gate": summary(generated_at_gate, len(generated)) if frozen_threshold is not None else None,
            "at_gate_source_bootstrap_ci95": source_bootstrap(generated_at_gate),
            "interpretation": "visual/native-action alignment, not visual ground-truth accuracy",
        },
        "claim_boundary": "diagnostic selective semantic-yaw reader; stop and complete trajectory remain unavailable",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

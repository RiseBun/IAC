#!/usr/bin/env python3
"""Audit the coarse MAS token: stop or moving direction (left/right/straight)."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import audit_mas_stop_token as motion
import audit_se2_yaw_confidence as direction


CLASSES = ("stop", "left", "right", "straight")


def wilson(hits: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = hits / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [centre - half, centre + half]


def combine(
    motion_rows: list[dict[str, Any]],
    direction_rows: list[dict[str, Any]],
    total: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    yaw = {row["sample_id"]: row for row in direction_rows}
    combined = []
    for row in motion_rows:
        motion_prediction = row.get("prediction")
        if motion_prediction is None:
            prediction = None
        elif motion_prediction:
            prediction = "stop"
        else:
            prediction = (yaw.get(row["sample_id"]) or {}).get("prediction")
        if row["label"]:
            label = "stop"
        else:
            yaw_row = yaw.get(row["sample_id"])
            label = yaw_row.get("label") if yaw_row is not None else None
        hit = None if prediction is None or label is None else prediction == label
        combined.append({
            "source_key": row["source_key"],
            "sample_id": row["sample_id"],
            "label": label,
            "prediction": prediction,
            "hit": hit,
        })
    scored = [row for row in combined if row["hit"] is not None]
    hits = sum(bool(row["hit"]) for row in scored)
    per_class = {}
    recalls = []
    for name in CLASSES:
        subset = [row for row in scored if row["label"] == name]
        class_hits = sum(bool(row["hit"]) for row in subset)
        recall = class_hits / len(subset) if subset else None
        if recall is not None:
            recalls.append(recall)
        per_class[name] = {"n": len(subset), "recall": recall, "wilson_ci95": wilson(class_hits, len(subset))}
    return ({
        "scored": len(scored),
        "coverage": len(scored) / total if total else 0.0,
        "accuracy": hits / len(scored) if scored else None,
        "macro_accuracy": float(np.mean(recalls)) if len(recalls) == len(CLASSES) else None,
        "per_class": per_class,
        "wilson_ci95": wilson(hits, len(scored)),
    }, combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--generated-manifest", required=True)
    parser.add_argument("--real-outputs", nargs="+", required=True)
    parser.add_argument("--generated-outputs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-radius-m", type=float, default=0.25)
    parser.add_argument("--yaw-deadband-rad", type=float, default=0.012)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    benchmark = direction.read_many([args.benchmark_manifest])
    generated_manifest = direction.read_many([args.generated_manifest])
    real_outputs = direction.read_many(args.real_outputs)
    generated_outputs = direction.read_many(args.generated_outputs)

    real_yaw_all = direction.build_real(benchmark, real_outputs)
    real_yaw = [row for row in real_yaw_all if row.get("action_extent_m") is not None and row["action_extent_m"] > args.stop_radius_m]
    real_yaw_oof, _ = direction.cross_validated(real_yaw, args.folds, args.seed, args.yaw_deadband_rad)
    real_motion = motion.join(benchmark, real_outputs, real=True, radius_m=args.stop_radius_m)
    real_motion_oof, motion_choices = motion.nested_oof(real_motion, args.folds, args.seed)
    real_summary, _ = combine(real_motion_oof, real_yaw_oof, len(benchmark))

    yaw_adapter = direction.fit_adapter(real_yaw, args.yaw_deadband_rad)
    motion_feature, _ = motion.choose_feature(real_motion, args.folds, args.seed)
    motion_adapter = motion.fit_feature(real_motion, motion_feature)
    assert motion_adapter is not None

    generated_yaw_all = direction.build_generated(generated_manifest, generated_outputs)
    generated_yaw = [row for row in generated_yaw_all if row.get("action_extent_m") is not None and row["action_extent_m"] > args.stop_radius_m]
    generated_yaw_eval = [direction.evaluate(row, yaw_adapter) for row in generated_yaw]
    generated_motion = motion.join(generated_manifest, generated_outputs, real=False, radius_m=args.stop_radius_m)
    generated_motion_eval = [motion.evaluate(row, motion_adapter) for row in generated_motion]
    generated_summary, generated_rows = combine(generated_motion_eval, generated_yaw_eval, len(generated_manifest))
    action_counts = Counter(row["label"] for row in generated_rows)
    stop_count = action_counts.get("stop", 0)

    payload = {
        "protocol": "iac-mas-direction-stop-pilot-v1",
        "token_space": list(CLASSES),
        "candidate_blind": True,
        "calibration_domain": "NAVSIM logged-real only",
        "logged_real_source_disjoint": real_summary,
        "frozen_adapters": {"direction": yaw_adapter, "motion": motion_adapter},
        "drivewam_native": {
            "action_class_counts": dict(action_counts),
            "alignment": generated_summary,
            "score_status": "scored" if stop_count >= 20 else "diagnostic_insufficient_native_stop_actions",
            "interpretation": "visual/native-action coarse-token alignment; not reality grounding or causality",
        },
        "motion_feature_choices_by_outer_fold": [row["feature"] for row in motion_choices],
        "claim_boundary": "coarse stop/left/right/straight compatibility only; no metric trajectory, speed, lateral distance, curvature, or causal mediation claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

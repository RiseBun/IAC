#!/usr/bin/env python3
"""Score frozen coarse flow-token MAS and same-source differential RCS."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:  # Package import for tests and library use.
    from tools import audit_mas_stop_token as motion
    from tools import audit_se2_yaw_confidence as direction
except ImportError:  # Direct ``python tools/score_*.py`` execution.
    import audit_mas_stop_token as motion
    import audit_se2_yaw_confidence as direction


def wilson(hits: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = hits / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [centre - half, centre + half]


def bootstrap(values: list[float], draws: int = 20000) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260914)
    sampled = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def source_bootstrap(rows: list[dict[str, Any]]) -> list[float] | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("hit") is not None:
            grouped[str(row["source_key"])].append(float(row["hit"]))
    return bootstrap([float(np.mean(values)) for values in grouped.values()])


def binary_summary(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    scored = [row for row in rows if row.get("hit") is not None]
    hits = sum(bool(row["hit"]) for row in scored)
    return {
        "scored": len(scored),
        "declared": total,
        "coverage": len(scored) / total if total else 0.0,
        "score": hits / len(scored) if scored else None,
        "wilson_ci95": wilson(hits, len(scored)),
        "source_bootstrap_ci95": source_bootstrap(scored),
    }


def load_frozen(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    yaw = config["direction"]
    stop = config["stop"]
    return (
        {
            "orientation": int(yaw["visual_orientation"]),
            "visual_deadband": float(yaw["visual_deadband"]),
            "action_deadband": float(yaw["action_yaw_deadband_rad"]),
        },
        {
            "feature": "median_q25",
            "threshold": float(stop["visual_threshold_px"]),
            "response_min_delta_px": float(stop.get("response_min_delta_px", 0.0)),
        },
    )


def joined(manifest: list[dict[str, Any]], outputs: list[dict[str, Any]], stop_radius: float) -> list[dict[str, Any]]:
    output_index = {str(row.get("sample_id")): row for row in outputs}
    rows = []
    for item in manifest:
        sid = str(item.get("sample_id") or f"{item['source_key']}::{item.get('branch_role', 'unknown')}")
        measured = output_index.get(sid)
        path = item.get("action_trajectory") or item.get("trajectory")
        yaw = direction.endpoint_yaw(path)
        extent = direction.path_extent(path)
        if measured is None or yaw is None or extent is None:
            continue
        visual_yaw, yaw_available = direction.decoded(measured)
        rows.append({
            "source_key": str(item["source_key"]),
            "sample_id": sid,
            "counterfactual_group_id": str(item.get("counterfactual_group_id") or item["source_key"]),
            "branch_role": str(item.get("branch_role") or item.get("branch_mode") or "native"),
            "action_yaw": yaw,
            "action_extent_m": extent,
            "action_stop": extent <= stop_radius,
            "visual_yaw": visual_yaw,
            "yaw_available": yaw_available,
            "motion_features": motion.motion_features(measured),
        })
    return rows


def score_mas(rows: list[dict[str, Any]], yaw_adapter: dict[str, Any], motion_adapter: dict[str, Any]) -> dict[str, Any]:
    direction_rows, stop_rows, joint_rows = [], [], []
    for row in rows:
        motion_eval = motion.evaluate({
            "source_key": row["source_key"], "sample_id": row["sample_id"],
            "label": row["action_stop"], "features": row["motion_features"],
        }, motion_adapter)
        yaw_eval = direction.evaluate({
            "source_key": row["source_key"], "sample_id": row["sample_id"],
            "action_yaw": row["action_yaw"], "visual_yaw": row["visual_yaw"],
            "available": row["yaw_available"],
        }, yaw_adapter)
        if not row["action_stop"]:
            direction_rows.append(yaw_eval)
        stop_rows.append(motion_eval)
        if motion_eval.get("prediction") is None:
            joint_prediction = None
        elif motion_eval["prediction"]:
            joint_prediction = "stop"
        else:
            joint_prediction = yaw_eval.get("prediction")
        joint_label = "stop" if row["action_stop"] else yaw_eval["label"]
        joint_rows.append({**row, "label": joint_label, "prediction": joint_prediction, "hit": None if joint_prediction is None else joint_prediction == joint_label})

    direction_result = binary_summary(direction_rows, len(direction_rows))
    motion_scored = [row for row in stop_rows if row.get("hit") is not None]
    labels = Counter("stop" if row["label"] else "moving" for row in stop_rows)
    recalls = {}
    for label, name in ((True, "stop"), (False, "moving")):
        subset = [row for row in motion_scored if row["label"] == label]
        hits = sum(bool(row["hit"]) for row in subset)
        recalls[name] = {"n": len(subset), "recall": hits / len(subset) if subset else None, "wilson_ci95": wilson(hits, len(subset))}
    valid_recalls = [value["recall"] for value in recalls.values() if value["recall"] is not None]
    stop_result = {
        **binary_summary(stop_rows, len(stop_rows)),
        "action_class_counts": dict(labels),
        "balanced_accuracy": float(np.mean(valid_recalls)) if len(valid_recalls) == 2 else None,
        "per_class": recalls,
        "status": "scored" if min(labels.values(), default=0) >= 20 and len(labels) == 2 else "diagnostic_insufficient_action_class_diversity",
    }
    stop_ci = stop_result.get("source_bootstrap_ci95")
    stop_result["promotion_status"] = (
        "passed" if stop_result["status"] == "scored" and stop_result["coverage"] >= 0.90
        and stop_ci is not None and stop_ci[0] >= 0.75
        else "failed_performance" if stop_result["status"] == "scored"
        else "not_eligible"
    )
    classes = ("stop", "left", "right", "straight")
    joint_scored = [row for row in joint_rows if row["hit"] is not None]
    per_class = {}
    class_recalls = []
    for name in classes:
        subset = [row for row in joint_scored if row["label"] == name]
        hits = sum(bool(row["hit"]) for row in subset)
        recall = hits / len(subset) if subset else None
        if recall is not None:
            class_recalls.append(recall)
        per_class[name] = {"n": len(subset), "recall": recall, "wilson_ci95": wilson(hits, len(subset))}
    joint_result = {
        **binary_summary(joint_rows, len(joint_rows)),
        "macro_accuracy": float(np.mean(class_recalls)) if len(class_recalls) == len(classes) else None,
        "per_class": per_class,
        "status": "scored" if all(per_class[name]["n"] >= 20 for name in classes) else "diagnostic_incomplete_class_support",
    }
    return {"direction": direction_result, "stop": stop_result, "joint": joint_result}


def score_rcs(
    rows: list[dict[str, Any]],
    *,
    orientation: int,
    action_yaw_delta: float,
    visual_yaw_delta: float,
    motion_adapter: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["counterfactual_group_id"]].append(row)
    declared_groups = 0
    direction_declared_pairs = 0
    stop_declared_pairs = 0
    yaw_rows, stop_rows = [], []
    pair_reasons = Counter()
    yaw_measurement_available = 0
    for group, items in grouped.items():
        if len(items) == 2:
            declared_groups += 1
        roles = {row["branch_role"]: row for row in items}
        if "left" in roles and "right" in roles:
            direction_declared_pairs += 1
            left, right = roles["left"], roles["right"]
            if left["yaw_available"] and right["yaw_available"]:
                yaw_measurement_available += 1
        else:
            left = right = None
        if left is not None and right is not None and not left["action_stop"] and not right["action_stop"]:
            action_delta = left["action_yaw"] - right["action_yaw"]
            if abs(action_delta) < action_yaw_delta:
                pair_reasons["nonmaterial_action_yaw_delta"] += 1
            elif not left["yaw_available"] or not right["yaw_available"]:
                yaw_rows.append({"source_key": left["source_key"], "hit": None, "reason": "visual_yaw_unavailable"})
            else:
                visual_delta = orientation * (float(left["visual_yaw"]) - float(right["visual_yaw"]))
                hit = abs(visual_delta) >= visual_yaw_delta and np.sign(visual_delta) == np.sign(action_delta)
                yaw_rows.append({"source_key": left["source_key"], "hit": bool(hit), "visual_delta": visual_delta, "action_delta": action_delta})

        # Stop response pairs are intentionally role-agnostic.  A valid pair
        # contains exactly one action-stop and one action-moving branch; names
        # such as ``stop/move`` must not be disguised as ``left/right`` merely
        # to pass a direction-specific schema.
        if len(items) == 2 and bool(items[0]["action_stop"]) != bool(items[1]["action_stop"]):
            stop_declared_pairs += 1
            evaluated = []
            for item in items:
                evaluated.append(motion.evaluate({
                    "source_key": item["source_key"],
                    "sample_id": item["sample_id"],
                    "label": item["action_stop"],
                    "features": item["motion_features"],
                }, motion_adapter))
            stop_item = items[0] if items[0]["action_stop"] else items[1]
            move_item = items[1] if items[0]["action_stop"] else items[0]
            if (
                any(item.get("prediction") is None for item in evaluated)
                or stop_item["motion_features"] is None
                or move_item["motion_features"] is None
            ):
                stop_rows.append({"source_key": items[0]["source_key"], "hit": None, "reason": "visual_motion_unavailable"})
            else:
                absolute_hits = [bool(item["prediction"] == item["label"]) for item in evaluated]
                visual_delta = move_item["motion_features"]["median_q25"] - stop_item["motion_features"]["median_q25"]
                response_margin = float(motion_adapter.get("response_min_delta_px", 0.0))
                stop_rows.append({
                    "source_key": items[0]["source_key"],
                    # RCS is an intervention-response ordering test.  Whether
                    # the stop branch reaches the absolute stop state is MAS,
                    # retained below as a separate semantic endpoint audit.
                    "hit": bool(visual_delta > response_margin),
                    "reversed_hit": bool(visual_delta < -response_margin),
                    "absolute_branch_hits": absolute_hits,
                    "absolute_endpoint_match": bool(all(absolute_hits)),
                    "visual_move_minus_stop": visual_delta,
                    "action_move_minus_stop_m": move_item["action_extent_m"] - stop_item["action_extent_m"],
                })
        elif len(items) == 2 and left is None:
            pair_reasons["nonmaterial_stop_move_delta"] += 1

    yaw_result = binary_summary(yaw_rows, direction_declared_pairs)
    yaw_scored = [row for row in yaw_rows if row.get("hit") is not None]
    yaw_result.update({
        "declared_pairs": direction_declared_pairs,
        "measurement_pair_coverage": yaw_measurement_available / direction_declared_pairs if direction_declared_pairs else 0.0,
        "material_action_pairs": len(yaw_rows),
        "score_coverage_within_material_pairs": len(yaw_scored) / len(yaw_rows) if yaw_rows else 0.0,
        "effective_score_coverage_over_declared_pairs": len(yaw_scored) / direction_declared_pairs if direction_declared_pairs else 0.0,
        "reversed_action_score": (sum(not row["hit"] for row in yaw_scored) / len(yaw_scored)) if yaw_scored else None,
        "constructed_self_pair_zero_control_false_response_rate": 0.0,
    })
    stop_result = binary_summary(stop_rows, stop_declared_pairs)
    stop_scored = [row for row in stop_rows if row.get("hit") is not None]
    absolute_endpoint_hits = sum(bool(row.get("absolute_endpoint_match")) for row in stop_scored)
    stop_result.update({
        "declared_pairs": stop_declared_pairs,
        "eligible_stop_move_pairs": stop_declared_pairs,
        "response_min_delta_px": float(motion_adapter.get("response_min_delta_px", 0.0)),
        "reversed_action_score": (sum(bool(row.get("reversed_hit")) for row in stop_scored) / len(stop_scored)) if stop_scored else None,
        "constructed_self_pair_zero_control_false_response_rate": 0.0,
        "absolute_endpoint_match_score_diagnostic": absolute_endpoint_hits / len(stop_scored) if stop_scored else None,
        "absolute_endpoint_match_hits_diagnostic": absolute_endpoint_hits,
        "status": "scored" if stop_declared_pairs >= 20 else "unavailable_insufficient_stop_move_pairs",
    })
    stop_ci = stop_result.get("source_bootstrap_ci95")
    stop_result["promotion_status"] = (
        "passed" if stop_result["status"] == "scored" and stop_result["coverage"] >= 0.90
        and stop_ci is not None and stop_ci[0] >= 0.75
        else "failed_performance" if stop_result["status"] == "scored"
        else "not_eligible"
    )
    return {
        "declared_pairs": declared_groups,
        "declared_counterfactual_groups": declared_groups,
        "direction": yaw_result,
        "stop": stop_result,
        "pair_exclusion_reasons": dict(pair_reasons),
        "claim_boundary": "same-source action/visual response consistency; RCS.stop tests whether a stop intervention reduces visual motion relative to move, while absolute stopping remains MAS.stop; not future-to-action mediation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source-filter-manifest", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-radius-m", type=float, default=0.25)
    parser.add_argument("--action-yaw-delta-rad", type=float, default=0.01)
    parser.add_argument("--visual-yaw-delta", type=float, default=0.0001)
    args = parser.parse_args()

    manifests = direction.read_many(args.manifest)
    if args.source_filter_manifest:
        allowed = {str(row["source_key"]) for row in direction.read_many(args.source_filter_manifest)}
        manifests = [row for row in manifests if str(row.get("source_key")) in allowed]
    outputs = direction.read_many(args.outputs)
    config = json.loads(Path(args.frozen_config).read_text(encoding="utf-8"))
    yaw_adapter, motion_adapter = load_frozen(config)
    rows = joined(manifests, outputs, args.stop_radius_m)
    payload = {
        "protocol": "iac-flow-token-mas-rcs-v1",
        "model_id": args.model_id,
        "candidate_blind": True,
        "records": len(rows),
        "MAS": score_mas(rows, yaw_adapter, motion_adapter),
        "RCS": score_rcs(
            rows,
            orientation=yaw_adapter["orientation"],
            action_yaw_delta=args.action_yaw_delta_rad,
            visual_yaw_delta=args.visual_yaw_delta,
            motion_adapter=motion_adapter,
        ),
        "frozen_config": str(args.frozen_config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

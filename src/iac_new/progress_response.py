"""Ordinal longitudinal response scoring for pure-speed counterfactuals.

This is deliberately a pilot metric, not the frozen RCS-yaw score.  It only
asks the question that can be identified from a same-history speed pair:
does the generated future with the larger commanded longitudinal progress
also have the larger measured future-only longitudinal response?
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def action_progress(record: dict[str, Any]) -> float | None:
    """Return the final commanded longitudinal position (metres)."""
    traj = record.get("action_trajectory")
    if traj is not None and isinstance(traj, list) and traj:
        try:
            return float(traj[-1][0])
        except (TypeError, ValueError, IndexError):
            pass
    # DriveWAM stores [batch, channels, time], with channel 0 longitudinal.
    traj = record.get("predicted_action_trajectory")
    try:
        if isinstance(traj, list) and len(traj) == 1:
            traj = traj[0]
        return float(traj[0][-1])
    except (TypeError, ValueError, IndexError):
        return None


def visual_progress(
    row: dict[str, Any],
    *,
    min_inlier_fraction: float = 0.20,
    min_intervals: int = 2,
    interval_indices: Iterable[int] = (1, 2, 3),
) -> dict[str, Any]:
    """Aggregate future-to-future longitudinal estimates conservatively."""
    accepted: list[float] = []
    raw: list[float | None] = []
    quality: list[float | None] = []
    intervals = row.get("intervals") or []
    for index in interval_indices:
        if index >= len(intervals):
            continue
        motion = (((intervals[index].get("estimators") or {}).get("all") or {}).get("motion") or {})
        value = motion.get("longitudinal_m")
        fraction = motion.get("inlier_fraction")
        raw.append(float(value) if value is not None else None)
        quality.append(float(fraction) if fraction is not None else None)
        if value is not None and fraction is not None and math.isfinite(float(value)) and float(fraction) >= min_inlier_fraction:
            accepted.append(float(value))
    return {
        "value": sum(accepted) if len(accepted) >= min_intervals else None,
        "accepted_intervals": len(accepted),
        "raw_values": raw,
        "inlier_fraction": quality,
    }


def score_progress_response(
    manifest_rows: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    *,
    min_action_delta: float = 0.5,
    min_inlier_fraction: float = 0.20,
    min_intervals: int = 2,
) -> dict[str, Any]:
    visual_by_id = {str(row.get("sample_id")): row for row in visual_rows}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        groups[str(row.get("counterfactual_group_id") or row.get("source_key"))].append(row)

    pairs: list[dict[str, Any]] = []
    monotonic_groups: list[dict[str, Any]] = []
    for group_id, rows in groups.items():
        by_role = {str(row.get("speed_role")): row for row in rows}
        if {"stop", "slow", "normal", "fast"}.issubset(by_role):
            ordered_roles = ("stop", "slow", "normal", "fast")
            action_values = {role: action_progress(by_role[role]) for role in ordered_roles}
            visual_values = {
                role: visual_progress(
                    visual_by_id.get(str(by_role[role].get("sample_id")), {}),
                    min_inlier_fraction=min_inlier_fraction,
                    min_intervals=min_intervals,
                )
                for role in ordered_roles
            }
            action_deltas = [
                None if action_values[a] is None or action_values[b] is None else action_values[b] - action_values[a]
                for a, b in zip(ordered_roles, ordered_roles[1:])
            ]
            action_valid = all(delta is not None and delta >= min_action_delta for delta in action_deltas)
            visual_deltas = [
                None if visual_values[a]["value"] is None or visual_values[b]["value"] is None else visual_values[b]["value"] - visual_values[a]["value"]
                for a, b in zip(ordered_roles, ordered_roles[1:])
            ]
            scored = bool(action_valid and all(delta is not None for delta in visual_deltas))
            hits = [bool(delta > 0) for delta in visual_deltas if delta is not None]
            monotonic_groups.append({
                "counterfactual_group_id": group_id,
                "action_values": action_values,
                "action_deltas": action_deltas,
                "visual_values": {role: visual_values[role]["value"] for role in ordered_roles},
                "visual_deltas": visual_deltas,
                "action_order_valid": action_valid,
                "scored": scored,
                "adjacent_hits": hits,
                "monotonic_hit": bool(scored and len(hits) == 3 and all(hits)),
            })
        if "fast" not in by_role or "slow" not in by_role:
            continue
        fast, slow = by_role["fast"], by_role["slow"]
        fast_action, slow_action = action_progress(fast), action_progress(slow)
        action_delta = None if fast_action is None or slow_action is None else fast_action - slow_action
        fast_visual = visual_progress(visual_by_id.get(str(fast.get("sample_id")), {}), min_inlier_fraction=min_inlier_fraction, min_intervals=min_intervals)
        slow_visual = visual_progress(visual_by_id.get(str(slow.get("sample_id")), {}), min_inlier_fraction=min_inlier_fraction, min_intervals=min_intervals)
        visual_delta = None if fast_visual["value"] is None or slow_visual["value"] is None else fast_visual["value"] - slow_visual["value"]
        action_valid = action_delta is not None and action_delta >= min_action_delta
        scored = bool(action_valid and visual_delta is not None)
        pairs.append({
            "counterfactual_group_id": group_id,
            "action_fast": fast_action,
            "action_slow": slow_action,
            "action_delta": action_delta,
            "visual_fast": fast_visual,
            "visual_slow": slow_visual,
            "visual_delta": visual_delta,
            "action_order_valid": action_valid,
            "scored": scored,
            "hit": bool(scored and visual_delta > 0),
        })

    eligible = [pair for pair in pairs if pair["action_order_valid"]]
    scored = [pair for pair in eligible if pair["scored"]]
    hits = sum(1 for pair in scored if pair["hit"])
    monotonic_eligible = [group for group in monotonic_groups if group["action_order_valid"]]
    monotonic_scored = [group for group in monotonic_eligible if group["scored"]]
    adjacent_hits = sum(sum(group["adjacent_hits"]) for group in monotonic_scored)
    adjacent_total = 3 * len(monotonic_scored)
    return {
        "protocol": "iac-progress-response-pilot-v1",
        "descriptor": "future_only_metric3d_reloc3r_longitudinal_sum_m",
        "interval_policy": "future_to_future_only_indices_1_2_3",
        "min_action_delta_m": min_action_delta,
        "min_inlier_fraction": min_inlier_fraction,
        "min_intervals": min_intervals,
        "declared_groups": len(groups),
        "eligible_pairs": len(eligible),
        "scored_pairs": len(scored),
        "coverage": len(scored) / len(eligible) if eligible else 0.0,
        "hits": hits,
        "ordering_accuracy": hits / len(scored) if scored else None,
        "four_role": {
            "declared_groups": len(monotonic_groups),
            "eligible_groups": len(monotonic_eligible),
            "scored_groups": len(monotonic_scored),
            "coverage": len(monotonic_scored) / len(monotonic_eligible) if monotonic_eligible else 0.0,
            "adjacent_hits": adjacent_hits,
            "adjacent_total": adjacent_total,
            "adjacent_accuracy": adjacent_hits / adjacent_total if adjacent_total else None,
            "monotonic_hits": sum(1 for group in monotonic_scored if group["monotonic_hit"]),
            "monotonic_accuracy": (
                sum(1 for group in monotonic_scored if group["monotonic_hit"]) / len(monotonic_scored)
                if monotonic_scored else None
            ),
            "groups": monotonic_groups,
        },
        "pairs": pairs,
    }

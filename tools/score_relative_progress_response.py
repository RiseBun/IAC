#!/usr/bin/env python3
"""Score counterfactual relative-progress response without metric-distance targets."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from iac_new.progress_response import action_progress, visual_progress


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    return ranks


def _spearman(values):
    target = list(range(len(values)))
    ranks = _rank(values)
    mean_target = sum(target) / len(target)
    mean_rank = sum(ranks) / len(ranks)
    numerator = sum((a - mean_target) * (b - mean_rank) for a, b in zip(target, ranks))
    denominator = math.sqrt(sum((a - mean_target) ** 2 for a in target) * sum((b - mean_rank) ** 2 for b in ranks))
    return numerator / denominator if denominator else None


def score(manifest, visual, roles, min_action_delta=0.2, min_intervals=2, min_inlier_fraction=0.2):
    visual_by_id = {str(row.get("sample_id")): row for row in visual}
    groups = defaultdict(list)
    for row in manifest:
        groups[str(row.get("counterfactual_group_id") or row.get("source_key"))].append(row)
    details = []
    for group_id, rows in groups.items():
        by_role = {str(row.get("speed_role")): row for row in rows}
        if not all(role in by_role for role in roles):
            continue
        action_values = [action_progress(by_role[role]) for role in roles]
        action_deltas = [None if a is None or b is None else b - a for a, b in zip(action_values, action_values[1:])]
        action_valid = all(delta is not None and delta >= min_action_delta for delta in action_deltas)
        visual_values = []
        diagnostics = {}
        for role in roles:
            value = visual_progress(
                visual_by_id.get(str(by_role[role].get("sample_id")), {}),
                min_inlier_fraction=min_inlier_fraction,
                min_intervals=min_intervals,
            )
            visual_values.append(value["value"])
            diagnostics[role] = value
        visual_valid = all(value is not None for value in visual_values)
        adjacent_hits = [b > a for a, b in zip(visual_values, visual_values[1:]) if a is not None and b is not None]
        pair_hits = []
        for left in range(len(roles)):
            for right in range(left + 1, len(roles)):
                if action_values[left] is None or action_values[right] is None or visual_values[left] is None or visual_values[right] is None:
                    continue
                expected = action_values[right] - action_values[left]
                observed = visual_values[right] - visual_values[left]
                pair_hits.append(observed * expected > 0)
        details.append({
            "counterfactual_group_id": group_id,
            "action_values": dict(zip(roles, action_values)),
            "visual_values": dict(zip(roles, visual_values)),
            "visual_diagnostics": diagnostics,
            "action_order_valid": action_valid,
            "visual_valid": visual_valid,
            "adjacent_hits": adjacent_hits,
            "pair_hits": pair_hits,
            "spearman": _spearman(visual_values) if visual_valid else None,
        })
    eligible = [row for row in details if row["action_order_valid"]]
    scored = [row for row in eligible if row["visual_valid"]]
    adjacent_hits = sum(sum(row["adjacent_hits"]) for row in scored)
    adjacent_total = (len(roles) - 1) * len(scored)
    pair_hits = sum(sum(row["pair_hits"]) for row in scored)
    pair_total = len(roles) * (len(roles) - 1) // 2 * len(scored)
    endpoint_hits = sum(int(row["visual_values"][roles[-1]] > row["visual_values"][roles[0]]) for row in scored)
    spearman = [row["spearman"] for row in scored if row["spearman"] is not None]
    return {
        "protocol": "iac-relative-progress-response-v1",
        "roles": roles,
        "target": "within-group relative longitudinal progress ordering",
        "absolute_distance_required": False,
        "declared_groups": len(groups),
        "eligible_groups": len(eligible),
        "scored_groups": len(scored),
        "coverage": len(scored) / len(eligible) if eligible else 0.0,
        "adjacent_hits": adjacent_hits,
        "adjacent_total": adjacent_total,
        "adjacent_accuracy": adjacent_hits / adjacent_total if adjacent_total else None,
        "all_pair_hits": pair_hits,
        "all_pair_total": pair_total,
        "all_pair_accuracy": pair_hits / pair_total if pair_total else None,
        "endpoint_hits": endpoint_hits,
        "endpoint_total": len(scored),
        "endpoint_accuracy": endpoint_hits / len(scored) if scored else None,
        "complete_monotonic_hits": sum(int(sum(row["adjacent_hits"]) == len(roles) - 1) for row in scored),
        "complete_monotonic_accuracy": sum(int(sum(row["adjacent_hits"]) == len(roles) - 1) for row in scored) / len(scored) if scored else None,
        "mean_spearman": sum(spearman) / len(spearman) if spearman else None,
        "median_spearman": sorted(spearman)[len(spearman) // 2] if spearman else None,
        "groups": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", default="stop,quarter,half,threequarter,normal")
    args = parser.parse_args()
    manifest = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    visual = json.loads(args.visual.read_text(encoding="utf-8"))["rows"]
    report = score(manifest, visual, [role.strip() for role in args.roles.split(",")])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("protocol", "declared_groups", "eligible_groups", "scored_groups", "coverage", "adjacent_accuracy", "all_pair_accuracy", "endpoint_accuracy", "complete_monotonic_accuracy", "mean_spearman", "median_spearman")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

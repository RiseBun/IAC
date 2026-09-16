#!/usr/bin/env python3
"""Score relative-progress response from candidate-blind dense optical flow."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from iac_new.progress_response import action_progress


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    return ranks


def _spearman(values):
    target = list(range(len(values)))
    ranks = _rank(values)
    mt, mr = sum(target) / len(target), sum(ranks) / len(ranks)
    denom = math.sqrt(sum((v - mt) ** 2 for v in target) * sum((v - mr) ** 2 for v in ranks))
    return sum((a - mt) * (b - mr) for a, b in zip(target, ranks)) / denom if denom else None


def score(manifest, flow_rows, roles, descriptor, interval_indices=(1, 2, 3), min_action_delta=0.2):
    flow_by_id = {str(row.get("sample_id")): row for row in flow_rows}
    groups = defaultdict(list)
    for row in manifest:
        groups[str(row.get("counterfactual_group_id") or row.get("source_key"))].append(row)
    details = []
    for group_id, rows in groups.items():
        by_role = {str(row.get("speed_role")): row for row in rows}
        if not all(role in by_role for role in roles):
            continue
        actions = [action_progress(by_role[role]) for role in roles]
        action_deltas = [None if a is None or b is None else b - a for a, b in zip(actions, actions[1:])]
        action_valid = all(delta is not None and delta >= min_action_delta for delta in action_deltas)
        values = []
        raw_by_role = {}
        for role in roles:
            flow = flow_by_id.get(str(by_role[role].get("sample_id")), {})
            intervals = (flow.get("flow_structure") or {}).get("rows") or []
            raw = []
            for index in interval_indices:
                item = intervals[index] if index < len(intervals) else {}
                value = item.get(descriptor)
                raw.append(float(value) if value is not None and math.isfinite(float(value)) and item.get("input_available", True) else None)
            raw_by_role[role] = raw
            accepted = [value for value in raw if value is not None]
            values.append(sum(accepted) / len(accepted) if len(accepted) == len(interval_indices) else None)
        visual_valid = all(value is not None for value in values)
        adjacent = [b > a for a, b in zip(values, values[1:]) if a is not None and b is not None]
        pairs = []
        for left in range(len(roles)):
            for right in range(left + 1, len(roles)):
                if values[left] is not None and values[right] is not None:
                    pairs.append((values[right] - values[left]) * (actions[right] - actions[left]) > 0)
        details.append({
            "counterfactual_group_id": group_id,
            "action_values": dict(zip(roles, actions)),
            "flow_values": dict(zip(roles, values)),
            "flow_values_by_interval": raw_by_role,
            "action_order_valid": action_valid,
            "flow_valid": visual_valid,
            "adjacent_hits": adjacent,
            "pair_hits": pairs,
            "spearman": _spearman(values) if visual_valid else None,
        })
    eligible = [row for row in details if row["action_order_valid"]]
    scored = [row for row in eligible if row["flow_valid"]]
    adjacent_total = (len(roles) - 1) * len(scored)
    adjacent_hits = sum(sum(row["adjacent_hits"]) for row in scored)
    pair_total = len(roles) * (len(roles) - 1) // 2 * len(scored)
    pair_hits = sum(sum(row["pair_hits"]) for row in scored)
    rhos = [row["spearman"] for row in scored if row["spearman"] is not None]
    return {
        "protocol": "iac-relative-flow-response-v1",
        "descriptor": descriptor,
        "interval_indices": list(interval_indices),
        "roles": roles,
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
        "complete_monotonic_hits": sum(int(sum(row["adjacent_hits"]) == len(roles) - 1) for row in scored),
        "complete_monotonic_accuracy": sum(int(sum(row["adjacent_hits"]) == len(roles) - 1) for row in scored) / len(scored) if scored else None,
        "mean_spearman": sum(rhos) / len(rhos) if rhos else None,
        "median_spearman": sorted(rhos)[len(rhos) // 2] if rhos else None,
        "groups": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptor", action="append", required=True)
    parser.add_argument("--roles", default="stop,quarter,half,threequarter,normal")
    args = parser.parse_args()
    manifest = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    flow_rows = [json.loads(line) for line in args.flow.read_text(encoding="utf-8").splitlines() if line.strip()]
    roles = [role.strip() for role in args.roles.split(",")]
    report = {descriptor: score(manifest, flow_rows, roles, descriptor) for descriptor in args.descriptor}
    args.output.write_text(json.dumps({"protocol": "iac-relative-flow-response-batch-v1", "reports": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({descriptor: {key: value for key, value in result.items() if key not in {"groups"}} for descriptor, result in report.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

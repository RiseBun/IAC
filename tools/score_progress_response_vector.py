#!/usr/bin/env python3
"""Score longitudinal intervention response without collapsing time first."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from iac_new.progress_response import action_progress


def _rank(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    return ranks


def _spearman(x, y):
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    den_x = sum((value - mx) ** 2 for value in rx)
    den_y = sum((value - my) ** 2 for value in ry)
    if den_x <= 0 or den_y <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(den_x * den_y)


def score(manifest, visual, roles, interval_indices=(1, 2, 3), min_action_delta=0.2, min_inlier_fraction=0.2):
    visual_by_id = {str(row.get("sample_id")): row for row in visual}
    groups = defaultdict(list)
    for row in manifest:
        groups[str(row.get("counterfactual_group_id") or row.get("source_key"))].append(row)
    details = []
    interval_records = []
    for group_id, rows in groups.items():
        by_role = {str(row.get("speed_role")): row for row in rows}
        if not all(role in by_role for role in roles):
            continue
        actions = [action_progress(by_role[role]) for role in roles]
        action_deltas = [None if a is None or b is None else b - a for a, b in zip(actions, actions[1:])]
        action_valid = all(delta is not None and delta >= min_action_delta for delta in action_deltas)
        role_values = {role: [] for role in roles}
        role_quality = {role: [] for role in roles}
        for role in roles:
            row = visual_by_id.get(str(by_role[role].get("sample_id")), {})
            intervals = row.get("intervals") or []
            for index in interval_indices:
                motion = (((intervals[index].get("estimators") or {}).get("all") or {}).get("motion") or {}) if index < len(intervals) else {}
                value = motion.get("longitudinal_m")
                quality = motion.get("inlier_fraction")
                value = float(value) if value is not None and math.isfinite(float(value)) else None
                quality = float(quality) if quality is not None else None
                role_values[role].append(value if quality is not None and quality >= min_inlier_fraction else None)
                role_quality[role].append(quality)
        interval_hits = []
        interval_valid = []
        interval_rho = []
        for position, index in enumerate(interval_indices):
            values = [role_values[role][position] for role in roles]
            valid = all(value is not None for value in values)
            deltas = [None if a is None or b is None else b - a for a, b in zip(values, values[1:])]
            hits = [delta > 0 for delta in deltas if delta is not None]
            interval_valid.append(valid)
            interval_hits.append(bool(valid and len(hits) == len(roles) - 1 and all(hits)))
            interval_rho.append(_spearman(list(range(len(roles))), values) if valid else None)
            interval_records.append({"group": group_id, "interval": index, "valid": valid, "deltas": deltas, "spearman": interval_rho[-1]})
        details.append({
            "counterfactual_group_id": group_id,
            "action_values": dict(zip(roles, actions)),
            "visual_values_by_interval": {role: role_values[role] for role in roles},
            "quality_by_interval": role_quality,
            "action_order_valid": action_valid,
            "interval_valid": interval_valid,
            "interval_monotonic_hits": interval_hits,
            "interval_spearman": interval_rho,
            "full_vector_monotonic": bool(action_valid and all(interval_hits)),
        })
    eligible = [row for row in details if row["action_order_valid"]]
    valid_interval_rows = [row for row in eligible for valid in [row["interval_valid"]] if all(valid)]
    local_hits = sum(sum(row["interval_monotonic_hits"]) for row in eligible)
    local_total = len(interval_indices) * len(eligible)
    adjacent_hits = 0
    adjacent_total = 0
    endpoint_hits = 0
    endpoint_total = 0
    for row in eligible:
        for position in range(len(interval_indices)):
            values = [row["visual_values_by_interval"][role][position] for role in roles]
            for left, right in zip(values, values[1:]):
                if left is not None and right is not None:
                    adjacent_total += 1
                    adjacent_hits += int(right > left)
            if values[0] is not None and values[-1] is not None:
                endpoint_total += 1
                endpoint_hits += int(values[-1] > values[0])
    rhos = [rho for row in eligible for rho in row["interval_spearman"] if rho is not None]
    return {
        "protocol": "iac-progress-response-vector-pilot-v1",
        "roles": roles,
        "interval_indices": list(interval_indices),
        "declared_groups": len(groups),
        "eligible_groups": len(eligible),
        "interval_monotonic_hits": local_hits,
        "interval_monotonic_total": local_total,
        "interval_monotonic_accuracy": local_hits / local_total if local_total else None,
        "local_adjacent_hits": adjacent_hits,
        "local_adjacent_total": adjacent_total,
        "local_adjacent_accuracy": adjacent_hits / adjacent_total if adjacent_total else None,
        "endpoint_hits": endpoint_hits,
        "endpoint_total": endpoint_total,
        "endpoint_accuracy": endpoint_hits / endpoint_total if endpoint_total else None,
        "full_vector_monotonic_hits": sum(1 for row in eligible if row["full_vector_monotonic"]),
        "full_vector_monotonic_accuracy": sum(1 for row in eligible if row["full_vector_monotonic"]) / len(eligible) if eligible else None,
        "mean_interval_spearman": sum(rhos) / len(rhos) if rhos else None,
        "median_interval_spearman": sorted(rhos)[len(rhos) // 2] if rhos else None,
        "groups": details,
        "interval_records": interval_records,
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
    print(json.dumps({key: report[key] for key in ("protocol", "declared_groups", "eligible_groups", "interval_monotonic_accuracy", "full_vector_monotonic_accuracy", "mean_interval_spearman", "median_interval_spearman")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score an ordered multi-level longitudinal response pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from iac_new.progress_response import action_progress, visual_progress


def score(manifest, visual, roles, min_action_delta=0.2):
    visual_by_id = {str(row.get("sample_id")): row for row in visual}
    groups = defaultdict(list)
    for row in manifest:
        groups[str(row.get("counterfactual_group_id") or row.get("source_key"))].append(row)
    details = []
    for group_id, rows in groups.items():
        by_role = {str(row.get("speed_role")): row for row in rows}
        if not all(role in by_role for role in roles):
            continue
        actions = [action_progress(by_role[role]) for role in roles]
        visuals = [visual_progress(visual_by_id.get(str(by_role[role].get("sample_id")), {}))['value'] for role in roles]
        action_deltas = [None if a is None or b is None else b - a for a, b in zip(actions, actions[1:])]
        visual_deltas = [None if a is None or b is None else b - a for a, b in zip(visuals, visuals[1:])]
        action_valid = all(delta is not None and delta >= min_action_delta for delta in action_deltas)
        scored = action_valid and all(delta is not None for delta in visual_deltas)
        hits = [bool(delta > 0) for delta in visual_deltas if delta is not None]
        details.append({
            "counterfactual_group_id": group_id,
            "action_values": dict(zip(roles, actions)),
            "visual_values": dict(zip(roles, visuals)),
            "action_deltas": action_deltas,
            "visual_deltas": visual_deltas,
            "action_order_valid": action_valid,
            "scored": scored,
            "adjacent_hits": hits,
            "monotonic_hit": bool(scored and len(hits) == len(roles) - 1 and all(hits)),
        })
    eligible = [row for row in details if row["action_order_valid"]]
    scored_rows = [row for row in eligible if row["scored"]]
    adjacent_hits = sum(sum(row["adjacent_hits"]) for row in scored_rows)
    adjacent_total = (len(roles) - 1) * len(scored_rows)
    return {
        "protocol": "iac-progress-response-multiscale-pilot-v1",
        "roles": roles,
        "declared_groups": len(groups),
        "eligible_groups": len(eligible),
        "scored_groups": len(scored_rows),
        "coverage": len(scored_rows) / len(eligible) if eligible else 0.0,
        "adjacent_hits": adjacent_hits,
        "adjacent_total": adjacent_total,
        "adjacent_accuracy": adjacent_hits / adjacent_total if adjacent_total else None,
        "monotonic_hits": sum(1 for row in scored_rows if row["monotonic_hit"]),
        "monotonic_accuracy": (
            sum(1 for row in scored_rows if row["monotonic_hit"]) / len(scored_rows)
            if scored_rows else None
        ),
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
    print(json.dumps({key: report[key] for key in ("protocol", "declared_groups", "eligible_groups", "scored_groups", "coverage", "adjacent_accuracy", "monotonic_accuracy")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

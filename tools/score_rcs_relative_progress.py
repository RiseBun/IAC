#!/usr/bin/env python3
"""Score the frozen coarse relative-progress RCS endpoint protocol."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from iac_new.progress_response import action_progress, visual_progress


def _bootstrap(values, draws, seed):
    if not values:
        return None
    numeric = [float(value) for value in values]
    rng = random.Random(seed)
    sampled = sorted(
        sum(rng.choice(numeric) for _ in numeric) / len(numeric)
        for _ in range(draws)
    )
    low = sampled[int(0.025 * (len(sampled) - 1))]
    high = sampled[int(0.975 * (len(sampled) - 1))]
    return [float(low), float(high)]


def _wilson(hits, total, z=1.959963984540054):
    if total <= 0:
        return None
    p = hits / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * ((p * (1.0 - p) / total + z * z / (4.0 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _contract_issues(rows, contract):
    issues = []
    if contract.get("same_source_within_group_required"):
        sources = {str(row.get("source_sample") or "") for row in rows}
        if len(sources) != 1 or "" in sources:
            issues.append("source_mismatch")
    if contract.get("same_history_required"):
        histories = {tuple(row.get("history_frame_paths") or []) for row in rows}
        fingerprints = {str(row.get("history_fingerprint") or "") for row in rows}
        if len(histories) != 1 or len(fingerprints) != 1 or "" in fingerprints:
            issues.append("history_mismatch")
    if contract.get("same_nuisance_seed_required"):
        seeds = {row.get("nuisance_seed") for row in rows}
        if len(seeds) != 1 or None in seeds:
            issues.append("nuisance_seed_mismatch")
    if contract.get("action_injection_verified_required") and not all(row.get("action_injection_verified") is True for row in rows):
        issues.append("action_injection_unverified")
    required_source = contract.get("future_images_source")
    if required_source and not all(row.get("future_images_source") == required_source for row in rows):
        issues.append("future_images_source_invalid")
    if contract.get("actual_regeneration_required"):
        paths = [tuple(row.get("future_frame_paths") or []) for row in rows]
        if any(not value for value in paths) or len(set(paths)) != len(paths):
            issues.append("branches_not_independently_generated")
    return issues


def score(manifest, visual_rows, config):
    progress_cfg = config["relative_progress"]
    stats = config["statistics"]
    visual_by_id = {str(row.get("sample_id")): row for row in visual_rows}
    groups = defaultdict(list)
    for row in manifest:
        groups[str(row.get("counterfactual_group_id") or row.get("source_key"))].append(row)
    results = []
    invalid = Counter()
    for group_id, rows in sorted(groups.items()):
        source_values = {str(row.get("source_sample") or "") for row in rows}
        source_sample = next(iter(source_values)) if len(source_values) == 1 else None
        issues = _contract_issues(rows, config["intervention_contract"])
        if issues:
            invalid.update(issues)
            results.append({
                "counterfactual_group_id": group_id,
                "source_sample": source_sample,
                "status": "invalid_contract",
                "issues": issues,
            })
            continue
        with_action = [(action_progress(row), row) for row in rows]
        with_action = [(value, row) for value, row in with_action if value is not None]
        if len(with_action) < 2:
            results.append({
                "counterfactual_group_id": group_id,
                "source_sample": source_sample,
                "status": "action_unavailable",
            })
            continue
        with_action.sort(key=lambda item: item[0])
        low_action, low = with_action[0]
        high_action, high = with_action[-1]
        action_delta = float(high_action - low_action)
        if action_delta < float(progress_cfg["minimum_action_delta_m"]):
            results.append({
                "counterfactual_group_id": group_id,
                "source_sample": source_sample,
                "status": "action_delta_below_threshold",
                "action_delta_m": action_delta,
            })
            continue
        kwargs = {
            "min_inlier_fraction": float(progress_cfg["minimum_inlier_fraction"]),
            "min_intervals": int(progress_cfg["minimum_quality_intervals"]),
            "interval_indices": tuple(int(value) for value in progress_cfg["interval_indices"]),
        }
        low_visual = visual_progress(visual_by_id.get(str(low.get("sample_id")), {}), **kwargs)
        high_visual = visual_progress(visual_by_id.get(str(high.get("sample_id")), {}), **kwargs)
        if low_visual["value"] is None or high_visual["value"] is None:
            results.append({
                "counterfactual_group_id": group_id,
                "source_sample": source_sample,
                "status": "visual_unavailable",
                "action_delta_m": action_delta,
                "low_visual": low_visual,
                "high_visual": high_visual,
            })
            continue
        visual_delta = float(high_visual["value"] - low_visual["value"])
        results.append({
            "counterfactual_group_id": group_id,
            "source_sample": source_sample,
            "status": "scored",
            "low_role": low.get("branch_role"),
            "high_role": high.get("branch_role"),
            "action_low_m": low_action,
            "action_high_m": high_action,
            "action_delta_m": action_delta,
            "visual_low": low_visual,
            "visual_high": high_visual,
            "visual_delta": visual_delta,
            "hit": visual_delta > 0.0,
        })
    contract_valid = [row for row in results if row["status"] != "invalid_contract"]
    eligible = [row for row in contract_valid if row["status"] not in {"action_unavailable", "action_delta_below_threshold"}]
    scored = [row for row in eligible if row["status"] == "scored"]
    hits = [bool(row["hit"]) for row in scored]
    accuracy = sum(hits) / len(hits) if hits else None
    bootstrap_ci = _bootstrap(hits, int(stats["bootstrap_draws"]), int(stats["bootstrap_seed"]))
    ci = _wilson(sum(hits), len(hits))
    criteria = stats["promotion_criteria"]
    coverage = len(scored) / len(eligible) if eligible else 0.0
    source_counts = Counter(row.get("source_sample") for row in eligible)
    source_counts.pop(None, None)
    independent_source_count = len(source_counts)
    repeated_eligible_sources = {
        source: count for source, count in sorted(source_counts.items()) if count > 1
    }
    independence_requirement_met = bool(
        not config["intervention_contract"].get("one_counterfactual_group_per_source_for_formal_report")
        or not repeated_eligible_sources
    )
    formal_sample_size_met = bool(
        independent_source_count >= int(stats["minimum_independent_sources_for_formal_report"])
        and independence_requirement_met
    )
    promotion_pass = bool(
        formal_sample_size_met
        and coverage >= float(criteria["coverage_min"])
        and accuracy is not None
        and accuracy >= float(criteria["endpoint_accuracy_min"])
        and ci is not None
        and ci[0] > float(criteria["endpoint_accuracy_ci95_lower_min"])
    )
    return {
        "protocol": config["protocol"],
        "status": "formal" if formal_sample_size_met else "pilot",
        "metric": "coarse_relative_progress_endpoint",
        "absolute_distance_required": False,
        "declared_groups": len(groups),
        "contract_valid_groups": len(contract_valid),
        "eligible_groups": len(eligible),
        "independent_eligible_sources": independent_source_count,
        "repeated_eligible_sources": repeated_eligible_sources,
        "independence_requirement_met": independence_requirement_met,
        "scored_groups": len(scored),
        "coverage": coverage,
        "hits": sum(hits),
        "endpoint_accuracy": accuracy,
        "endpoint_accuracy_ci95": ci,
        "endpoint_accuracy_bootstrap_ci95_diagnostic": bootstrap_ci,
        "formal_sample_size_met": formal_sample_size_met,
        "promotion_pass": promotion_pass,
        "invalid_contract_reasons": dict(invalid),
        "groups": results,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--visual", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs" / "response_consistency_v2.json")
    args = parser.parse_args()
    manifest = [
        json.loads(line)
        for path in args.manifest
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    visual = [
        row
        for path in args.visual
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]
    ]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = score(manifest, visual, config)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "groups"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

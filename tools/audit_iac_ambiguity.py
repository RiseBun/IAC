"""Audit hard-label ambiguity in IAC candidate ranking scores.

The benchmark gives each candidate group one hard GT positive. In practice,
small speed/lateral/heading perturbations can be visually compatible with the
same generated future image. This tool separates top-1 misses into:

1. ambiguous accept: a near-trajectory negative beats GT by a small margin;
2. evidence-supported miss: GT has positive path evidence but loses ranking;
3. likely model error: the remaining misses.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _numeric_consistency_fields(rows: Iterable[Dict[str, Any]]) -> List[str]:
    fields = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith("iac_consistency") and isinstance(value, (int, float)):
                fields.add(key)
    return sorted(fields)


def _recompute_delta_fields(row: Dict[str, Any]) -> None:
    if "iac_consistency" not in row:
        return
    score = float(row["iac_consistency"])
    if "iac_consistency_path_masked" in row:
        row["path_mask_delta"] = score - float(row["iac_consistency_path_masked"])
    if "iac_consistency_sky_masked" in row:
        row["sky_mask_delta"] = score - float(row["iac_consistency_sky_masked"])
    if "path_mask_delta" in row and "sky_mask_delta" in row:
        row["path_minus_sky_delta"] = (
            float(row["path_mask_delta"]) - float(row["sky_mask_delta"])
        )
    if "iac_consistency_wrong_path_masked" in row:
        row["wrong_path_delta"] = score - float(row["iac_consistency_wrong_path_masked"])
    if "path_mask_delta" in row and "wrong_path_delta" in row:
        row["candidate_minus_wrong_path_delta"] = (
            float(row["path_mask_delta"]) - float(row["wrong_path_delta"])
        )
    if "iac_consistency_candidate_exclusive_path_masked" in row:
        row["candidate_exclusive_path_delta"] = score - float(
            row["iac_consistency_candidate_exclusive_path_masked"]
        )
    if "iac_consistency_wrong_exclusive_path_masked" in row:
        row["wrong_exclusive_path_delta"] = score - float(
            row["iac_consistency_wrong_exclusive_path_masked"]
        )
    if "candidate_exclusive_path_delta" in row and "wrong_exclusive_path_delta" in row:
        row["candidate_minus_wrong_exclusive_path_delta"] = (
            float(row["candidate_exclusive_path_delta"])
            - float(row["wrong_exclusive_path_delta"])
        )


def _fuse_rows(
    primary_rows: Sequence[Dict[str, Any]],
    aux_rows: Sequence[Dict[str, Any]],
    alpha: float,
    group_key: str,
) -> List[Dict[str, Any]]:
    if len(primary_rows) != len(aux_rows):
        raise ValueError(
            f"row count mismatch: primary={len(primary_rows)} aux={len(aux_rows)}"
        )
    fields = sorted(
        set(_numeric_consistency_fields(primary_rows))
        & set(_numeric_consistency_fields(aux_rows))
    )
    fused: List[Dict[str, Any]] = []
    for idx, (primary, aux) in enumerate(zip(primary_rows, aux_rows)):
        group_a = primary.get(group_key) or primary.get("anchor_id") or primary.get("sample_id")
        group_b = aux.get(group_key) or aux.get("anchor_id") or aux.get("sample_id")
        if group_a != group_b:
            raise ValueError(f"row {idx} group mismatch: {group_a!r} vs {group_b!r}")
        row = dict(primary)
        for field in fields:
            row[field] = _sigmoid(_logit(primary[field]) + alpha * _logit(aux[field]))
        _recompute_delta_fields(row)
        fused.append(row)
    return fused


def _source(row: Dict[str, Any], wam_key: str) -> str:
    for key in ("source_type", "action_type", wam_key, "wam_name", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _is_positive(row: Dict[str, Any], wam_key: str) -> bool:
    if row.get("consistency_label") is not None:
        return int(row["consistency_label"]) == 1
    if row.get("label") is not None:
        return int(row["label"]) == 1
    return _source(row, wam_key) == "gt_pos"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _audit(
    rows: Sequence[Dict[str, Any]],
    group_key: str,
    wam_key: str,
    close_margin: float,
    near_sources: set[str],
    evidence_margin: float,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        group_id = row.get(group_key) or row.get("anchor_id") or row.get("sample_id") or idx
        groups[str(group_id)].append(row)

    stats = Counter()
    miss_sources: Counter[str] = Counter()
    close_sources: Counter[str] = Counter()
    evidence_supported_sources: Counter[str] = Counter()
    gaps: List[float] = []
    close_gaps: List[float] = []
    model_error_gaps: List[float] = []
    per_group: List[Dict[str, Any]] = []

    for group_id, items in groups.items():
        if len(items) < 2:
            continue
        positives = [row for row in items if _is_positive(row, wam_key)]
        if not positives:
            continue
        positive = positives[0]
        ranked = sorted(items, key=lambda row: float(row["iac_consistency"]), reverse=True)
        pos_rank = next(i + 1 for i, row in enumerate(ranked) if row is positive)
        top = ranked[0]
        top_source = _source(top, wam_key)
        stats["groups"] += 1
        if pos_rank == 1:
            stats["top1"] += 1
            path_minus_sky = positive.get("path_minus_sky_delta")
            exact_delta = positive.get("candidate_minus_wrong_exclusive_path_delta")
            if exact_delta is None:
                exact_delta = positive.get("candidate_minus_wrong_path_delta")
            per_group.append(
                {
                    "group_id": group_id,
                    "category": "hit",
                    "positive_rank": pos_rank,
                    "winning_source": top_source,
                    "score_gap": 0.0,
                    "gt_score": positive.get("iac_consistency"),
                    "winner_score": top.get("iac_consistency"),
                    "gt_path_minus_sky_delta": path_minus_sky,
                    "gt_exact_path_delta": exact_delta,
                    "recovered_path_agreement_score": (
                        exact_delta if exact_delta is not None else path_minus_sky
                    ),
                    "winner_sample_id": top.get("sample_id"),
                    "gt_sample_id": positive.get("sample_id"),
                }
            )
            continue

        stats["misses"] += 1
        miss_sources[top_source] += 1
        gap = float(top["iac_consistency"]) - float(positive["iac_consistency"])
        gaps.append(gap)

        exact_delta = positive.get("candidate_minus_wrong_exclusive_path_delta")
        if exact_delta is None:
            exact_delta = positive.get("candidate_minus_wrong_path_delta")
        path_minus_sky = positive.get("path_minus_sky_delta")
        exact_supported = exact_delta is not None and float(exact_delta) > evidence_margin
        path_supported = path_minus_sky is not None and float(path_minus_sky) > 0.0
        is_ambiguous_accept = (
            top_source in near_sources
            and gap <= close_margin
            and path_supported
        )

        if is_ambiguous_accept:
            stats["ambiguous_accept"] += 1
            close_sources[top_source] += 1
            close_gaps.append(gap)
            category = "ambiguous_accept"
        elif exact_supported:
            stats["evidence_supported_miss"] += 1
            evidence_supported_sources[top_source] += 1
            category = "evidence_supported_miss"
        else:
            stats["likely_model_error"] += 1
            model_error_gaps.append(gap)
            category = "likely_model_error"

        per_group.append(
            {
                "group_id": group_id,
                "category": category,
                "positive_rank": pos_rank,
                "winning_source": top_source,
                "score_gap": gap,
                "gt_score": positive.get("iac_consistency"),
                "winner_score": top.get("iac_consistency"),
                "gt_path_minus_sky_delta": path_minus_sky,
                "gt_exact_path_delta": exact_delta,
                "recovered_path_agreement_score": (
                    exact_delta if exact_delta is not None else path_minus_sky
                ),
                "winner_sample_id": top.get("sample_id"),
                "gt_sample_id": positive.get("sample_id"),
            }
        )

    groups_n = int(stats["groups"])
    misses = int(stats["misses"])
    summary = {
        "num_groups": groups_n,
        "raw_miss_fraction": misses / groups_n if groups_n else None,
        "hard_top1": int(stats["top1"]) / groups_n if groups_n else None,
        "ambiguity_adjusted_top1": (
            (int(stats["top1"]) + int(stats["ambiguous_accept"])) / groups_n
            if groups_n else None
        ),
        "misses": misses,
        "miss_breakdown": {
            "hit": int(stats["top1"]),
            "ambiguous_accept": int(stats["ambiguous_accept"]),
            "evidence_supported_miss": int(stats["evidence_supported_miss"]),
            "likely_model_error": int(stats["likely_model_error"]),
        },
        "miss_breakdown_fraction_of_misses": {
            key: (int(stats[key]) / misses if misses else None)
            for key in ("ambiguous_accept", "evidence_supported_miss", "likely_model_error")
        },
        "formal_categories": [
            "hit",
            "ambiguous_accept",
            "evidence_supported_miss",
            "likely_model_error",
        ],
        "top_miss_sources": miss_sources.most_common(),
        "ambiguous_accept_sources": close_sources.most_common(),
        "evidence_supported_miss_sources": evidence_supported_sources.most_common(),
        "gap_stats": {
            "mean": _mean(gaps),
            "median": _quantile(gaps, 0.5),
            "p25": _quantile(gaps, 0.25),
            "p75": _quantile(gaps, 0.75),
            "close_mean": _mean(close_gaps),
            "likely_model_error_mean": _mean(model_error_gaps),
            "le_close_margin_count": sum(g <= close_margin for g in gaps),
            "le_close_margin_fraction_of_misses": (
                sum(g <= close_margin for g in gaps) / misses if misses else None
            ),
        },
        "example_misses": per_group[:20],
    }
    return summary, per_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", help="Primary or already-fused wam_iac_scores.jsonl")
    parser.add_argument("--primary-scores", help="Primary score JSONL for fusion")
    parser.add_argument("--aux-scores", help="Aux/path-head score JSONL for fusion")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--per-sample-output",
        default=None,
        help="Optional JSONL path with one ambiguity record per ranked group.",
    )
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--close-margin", type=float, default=0.02)
    parser.add_argument(
        "--near-sources",
        default="perturb_speed,perturb_lateral,perturb_heading",
        help="Comma-separated negative source types treated as visual near-neighbors.",
    )
    parser.add_argument("--evidence-margin", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scores:
        rows = _load_rows(Path(args.scores))
    else:
        if not args.primary_scores or not args.aux_scores:
            raise SystemExit("provide either --scores or both --primary-scores/--aux-scores")
        rows = _fuse_rows(
            _load_rows(Path(args.primary_scores)),
            _load_rows(Path(args.aux_scores)),
            args.alpha,
            args.group_key,
        )

    near_sources = {item.strip() for item in args.near_sources.split(",") if item.strip()}
    report, per_group = _audit(
        rows=rows,
        group_key=args.group_key,
        wam_key=args.wam_key,
        close_margin=args.close_margin,
        near_sources=near_sources,
        evidence_margin=args.evidence_margin,
    )
    report["config"] = {
        "scores": args.scores,
        "primary_scores": args.primary_scores,
        "aux_scores": args.aux_scores,
        "alpha": args.alpha,
        "group_key": args.group_key,
        "wam_key": args.wam_key,
        "close_margin": args.close_margin,
        "near_sources": sorted(near_sources),
        "evidence_margin": args.evidence_margin,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.per_sample_output:
        sample_path = Path(args.per_sample_output)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_path.open("w", encoding="utf-8") as f:
            for record in per_group:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

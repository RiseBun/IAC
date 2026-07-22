"""Audit where recovered-set agreement helps or hurts IAC ranking.

The recovered-set path currently improves ambiguity-adjusted top1 when fused
with consistency+path. This script breaks that aggregate gain into group-level
transitions and source-level failure modes so the next change targets the real
bottleneck instead of another blind fusion sweep.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


NEAR_SOURCES = {"perturb_speed", "perturb_lateral", "perturb_heading"}
HARD_MISMATCH_SOURCES = {
    "image_swap",
    "time_shift",
    "time_shift_future",
    "traj_swap",
    "reverse",
    "high_pdm_image_mismatch",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _source(row: Dict[str, Any], wam_key: str) -> str:
    for key in ("source_type", "action_type", wam_key, "wam_name", "sample_type", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _is_positive(row: Dict[str, Any], wam_key: str) -> bool:
    if row.get("consistency_label") is not None:
        return float(row["consistency_label"]) > 0.5
    if row.get("label") is not None:
        return float(row["label"]) > 0.5
    return _source(row, wam_key) == "gt_pos"


def _group_id(row: Dict[str, Any], group_key: str) -> str | None:
    value = row.get(group_key) or row.get("anchor_id") or row.get("group_id")
    if value is not None:
        return str(value)
    sample_id = row.get("sample_id")
    if sample_id is None:
        return None
    sample_id = str(sample_id)
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def _groups(rows: Iterable[Dict[str, Any]], group_key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        gid = _group_id(row, group_key)
        if gid is None:
            gid = str(idx)
        out[gid].append(row)
    return out


def _rank(items: Sequence[Dict[str, Any]], score_key: str) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda row: _safe_float(row, score_key), reverse=True)


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


def _index_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_sample: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id is not None:
            by_sample[str(sample_id)] = row
    return by_sample


def _align_rows(
    base_rows: Sequence[Dict[str, Any]],
    recovered_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    recovered_by_sample = _index_rows(recovered_rows)
    aligned: List[Dict[str, Any]] = []
    for idx, row in enumerate(base_rows):
        merged = dict(row)
        sample_id = row.get("sample_id")
        rec = recovered_by_sample.get(str(sample_id)) if sample_id is not None else None
        if rec is None and idx < len(recovered_rows):
            rec = recovered_rows[idx]
        if rec is not None:
            for key, value in rec.items():
                if key.startswith("recovered_set_"):
                    merged[key] = value
            if "score_fusion_label" in rec:
                merged["recovered_score_fusion_label"] = rec["score_fusion_label"]
        aligned.append(merged)
    return aligned


def _top_summary(
    items: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    wam_key: str,
) -> Dict[str, Any]:
    ranked = _rank(items, score_key)
    top = ranked[0]
    pos_idx = next(
        (idx for idx, row in enumerate(ranked) if _is_positive(row, wam_key)),
        None,
    )
    positive = ranked[pos_idx] if pos_idx is not None else None
    return {
        "top": top,
        "positive": positive,
        "top_source": _source(top, wam_key),
        "positive_rank": None if pos_idx is None else pos_idx + 1,
        "hit": bool(pos_idx == 0),
        "top_score": _safe_float(top, score_key),
        "positive_score": None if positive is None else _safe_float(positive, score_key),
    }


def _audit_split(
    rows: Sequence[Dict[str, Any]],
    *,
    split: str,
    group_key: str,
    wam_key: str,
    cp_score_key: str,
    recovered_score_key: str,
    fused_score_key: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    grouped = _groups(rows, group_key)
    stats = Counter()
    cp_miss_sources: Counter[str] = Counter()
    fused_miss_sources: Counter[str] = Counter()
    recovered_top_sources: Counter[str] = Counter()
    improves_from_sources: Counter[str] = Counter()
    regress_to_sources: Counter[str] = Counter()
    source_rows: Dict[str, Dict[str, List[float] | int]] = defaultdict(
        lambda: {
            "count": 0,
            "agreement": [],
            "minade": [],
            "supported": [],
            "path_iou": [],
        }
    )
    group_rows: List[Dict[str, Any]] = []

    for group_id, items in grouped.items():
        if len(items) < 2:
            continue
        positives = [row for row in items if _is_positive(row, wam_key)]
        if not positives:
            continue
        gt = positives[0]
        cp = _top_summary(items, score_key=cp_score_key, wam_key=wam_key)
        rec = _top_summary(items, score_key=recovered_score_key, wam_key=wam_key)
        fused = _top_summary(items, score_key=fused_score_key, wam_key=wam_key)

        stats["groups"] += 1
        stats["cp_hit"] += int(cp["hit"])
        stats["recovered_hit"] += int(rec["hit"])
        stats["fused_hit"] += int(fused["hit"])
        stats["gt_supported"] += int(_safe_float(gt, "recovered_set_supported") > 0.5)
        stats["cp_winner_supported"] += int(
            _safe_float(cp["top"], "recovered_set_supported") > 0.5
        )
        stats["fused_winner_supported"] += int(
            _safe_float(fused["top"], "recovered_set_supported") > 0.5
        )

        if not cp["hit"]:
            cp_miss_sources[str(cp["top_source"])] += 1
        if not fused["hit"]:
            fused_miss_sources[str(fused["top_source"])] += 1
        recovered_top_sources[str(rec["top_source"])] += 1

        transition = "unchanged"
        if (not cp["hit"]) and fused["hit"]:
            transition = "fixed_by_recovered"
            improves_from_sources[str(cp["top_source"])] += 1
            stats["fixed_by_recovered"] += 1
        elif cp["hit"] and (not fused["hit"]):
            transition = "broken_by_recovered"
            regress_to_sources[str(fused["top_source"])] += 1
            stats["broken_by_recovered"] += 1
        elif (not cp["hit"]) and (not fused["hit"]):
            stats["still_miss"] += 1
        else:
            stats["still_hit"] += 1

        hard_above_gt = []
        near_above_gt = []
        gt_agreement = _safe_float(gt, recovered_score_key)
        for row in items:
            source = _source(row, wam_key)
            bucket = source_rows[source]
            bucket["count"] = int(bucket["count"]) + 1
            for metric_key, out_key in (
                (recovered_score_key, "agreement"),
                ("recovered_set_minade", "minade"),
                ("recovered_set_supported", "supported"),
                ("recovered_set_path_iou", "path_iou"),
            ):
                cast_bucket = bucket[out_key]
                assert isinstance(cast_bucket, list)
                cast_bucket.append(_safe_float(row, metric_key))
            if row is gt:
                continue
            if _safe_float(row, recovered_score_key) > gt_agreement:
                if source in HARD_MISMATCH_SOURCES:
                    hard_above_gt.append(source)
                if source in NEAR_SOURCES:
                    near_above_gt.append(source)

        stats["has_hard_mismatch_above_gt"] += int(bool(hard_above_gt))
        stats["has_near_perturb_above_gt"] += int(bool(near_above_gt))

        group_rows.append(
            {
                "split": split,
                "group_id": group_id,
                "transition": transition,
                "cp_hit": bool(cp["hit"]),
                "recovered_hit": bool(rec["hit"]),
                "fused_hit": bool(fused["hit"]),
                "cp_winner_source": cp["top_source"],
                "recovered_winner_source": rec["top_source"],
                "fused_winner_source": fused["top_source"],
                "cp_positive_rank": cp["positive_rank"],
                "fused_positive_rank": fused["positive_rank"],
                "gt_recovered_agreement": gt_agreement,
                "cp_winner_recovered_agreement": _safe_float(cp["top"], recovered_score_key),
                "fused_winner_recovered_agreement": _safe_float(
                    fused["top"], recovered_score_key
                ),
                "gt_minade": _safe_float(gt, "recovered_set_minade"),
                "cp_winner_minade": _safe_float(cp["top"], "recovered_set_minade"),
                "fused_winner_minade": _safe_float(fused["top"], "recovered_set_minade"),
                "gt_supported": bool(_safe_float(gt, "recovered_set_supported") > 0.5),
                "cp_winner_supported": bool(
                    _safe_float(cp["top"], "recovered_set_supported") > 0.5
                ),
                "fused_winner_supported": bool(
                    _safe_float(fused["top"], "recovered_set_supported") > 0.5
                ),
                "hard_mismatch_above_gt_sources": sorted(set(hard_above_gt)),
                "near_perturb_above_gt_sources": sorted(set(near_above_gt)),
                "gt_sample_id": gt.get("sample_id"),
                "cp_winner_sample_id": cp["top"].get("sample_id"),
                "fused_winner_sample_id": fused["top"].get("sample_id"),
            }
        )

    groups_n = max(int(stats["groups"]), 1)
    source_summary = {}
    for source, values in sorted(source_rows.items()):
        source_summary[source] = {
            "count": values["count"],
            "mean_agreement": _mean(values["agreement"]),  # type: ignore[arg-type]
            "mean_minade": _mean(values["minade"]),  # type: ignore[arg-type]
            "supported_rate": _mean(values["supported"]),  # type: ignore[arg-type]
            "mean_path_iou": _mean(values["path_iou"]),  # type: ignore[arg-type]
        }

    summary = {
        "split": split,
        "num_groups": int(stats["groups"]),
        "cp_top1": stats["cp_hit"] / groups_n,
        "recovered_top1": stats["recovered_hit"] / groups_n,
        "fused_top1": stats["fused_hit"] / groups_n,
        "fixed_by_recovered_rate": stats["fixed_by_recovered"] / groups_n,
        "broken_by_recovered_rate": stats["broken_by_recovered"] / groups_n,
        "still_miss_rate": stats["still_miss"] / groups_n,
        "gt_supported_rate": stats["gt_supported"] / groups_n,
        "cp_winner_supported_rate": stats["cp_winner_supported"] / groups_n,
        "fused_winner_supported_rate": stats["fused_winner_supported"] / groups_n,
        "hard_mismatch_above_gt_group_rate": stats["has_hard_mismatch_above_gt"] / groups_n,
        "near_perturb_above_gt_group_rate": stats["has_near_perturb_above_gt"] / groups_n,
        "cp_miss_sources": dict(cp_miss_sources),
        "fused_miss_sources": dict(fused_miss_sources),
        "recovered_top_sources": dict(recovered_top_sources),
        "improves_from_sources": dict(improves_from_sources),
        "regress_to_sources": dict(regress_to_sources),
        "source_summary": source_summary,
    }
    return summary, group_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--splits", default="regular,low_iou,holdout")
    parser.add_argument("--recovered-alpha", default="0.3")
    parser.add_argument("--path-alpha", default="0.2")
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-groups")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    all_summaries: List[Dict[str, Any]] = []
    all_groups: List[Dict[str, Any]] = []
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        cp_path = eval_root / f"{split}_fused_consistency_path_scores.jsonl"
        recovered_path = eval_root / f"{split}_recovered_set_scores.jsonl"
        fused_path = (
            eval_root
            / f"{split}_fused_consistency_path_recovered_a{args.recovered_alpha}_scores.jsonl"
        )
        if not cp_path.exists():
            raise FileNotFoundError(cp_path)
        if not recovered_path.exists():
            raise FileNotFoundError(recovered_path)
        if not fused_path.exists():
            raise FileNotFoundError(fused_path)

        cp_rows = _load_jsonl(cp_path)
        recovered_rows = _load_jsonl(recovered_path)
        fused_rows = _load_jsonl(fused_path)
        rows = _align_rows(cp_rows, recovered_rows)
        fused_by_sample = _index_rows(fused_rows)
        for idx, row in enumerate(rows):
            sample_id = row.get("sample_id")
            fused = fused_by_sample.get(str(sample_id)) if sample_id is not None else None
            if fused is None and idx < len(fused_rows):
                fused = fused_rows[idx]
            if fused is not None:
                row["iac_consistency_fused_cpr"] = fused.get("iac_consistency")

        summary, group_rows = _audit_split(
            rows,
            split=split,
            group_key=args.group_key,
            wam_key=args.wam_key,
            cp_score_key="iac_consistency",
            recovered_score_key="recovered_set_agreement",
            fused_score_key="iac_consistency_fused_cpr",
        )
        all_summaries.append(summary)
        all_groups.extend(group_rows)

    output = {
        "eval_root": str(eval_root),
        "path_alpha": args.path_alpha,
        "recovered_alpha": args.recovered_alpha,
        "splits": all_summaries,
    }
    out_path = Path(args.output_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_groups:
        _write_jsonl(Path(args.output_groups), all_groups)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

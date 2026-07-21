"""Bootstrap confidence intervals for IAC-PathBench v2 metrics."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_id(row: Dict[str, Any], group_key: str) -> str:
    value = row.get(group_key) or row.get("anchor_id") or row.get("group_id")
    if value is not None:
        return str(value)
    return str(row.get("sample_id", "unknown"))


def _group_rows(rows: Iterable[Dict[str, Any]], group_key: str) -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for row in rows:
        gid = _group_id(row, group_key)
        if gid not in grouped:
            order.append(gid)
        grouped[gid].append(row)
    return [grouped[gid] for gid in order]


def _rename_group(row: Dict[str, Any], gid: str, group_key: str) -> Dict[str, Any]:
    out = dict(row)
    out[group_key] = gid
    out["group_id"] = gid
    return out


def _mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def _percentile(values: List[float], pct: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _compact(summary: Dict[str, Any]) -> Dict[str, Any]:
    v2 = summary.get("iac_pathbench_v2", {})
    primary = v2.get("primary_scientific_metrics", {})
    secondary = v2.get("secondary_ranking_metrics", {})
    diagnostic = v2.get("diagnostic_metrics", {})
    out = {
        "hard_top1": secondary.get("hard_top1"),
        "ambiguity_adjusted_top1": primary.get("ambiguity_adjusted_top1"),
        "mrr": secondary.get("mrr"),
        "exact_path_win_fraction": primary.get("exact_path_win_fraction"),
        "exact_path_delta": primary.get("exact_path_delta"),
        "path_minus_sky_delta": primary.get("path_minus_sky_delta"),
        "likely_model_error_fraction": diagnostic.get("likely_model_error_fraction"),
        "ambiguity_supported_miss_fraction": diagnostic.get(
            "ambiguity_supported_miss_fraction"
        ),
    }
    v3 = summary.get("iac_pathbench_v3", {})
    if v3:
        v3_primary = v3.get("primary_scientific_metrics", {})
        v3_secondary = v3.get("secondary_ranking_metrics", {})
        v3_diagnostic = v3.get("diagnostic_metrics", {})
        out.update(
            {
                "v3_recovered_set_gt_supported_fraction": v3_primary.get(
                    "recovered_set_gt_supported_fraction"
                ),
                "v3_recovered_set_winner_supported_fraction": v3_primary.get(
                    "recovered_set_winner_supported_fraction"
                ),
                "v3_recovered_set_top1": v3_secondary.get("recovered_set_top1"),
                "v3_recovered_set_gt_minade": v3_secondary.get(
                    "recovered_set_gt_minade"
                ),
                "v3_recovered_set_gt_better_than_winner_fraction": v3_diagnostic.get(
                    "recovered_set_gt_better_than_winner_fraction"
                ),
                "v3_recovered_set_mean_ambiguity_set_size": v3_diagnostic.get(
                    "recovered_set_mean_ambiguity_set_size"
                ),
            }
        )
        visual = v3_diagnostic.get("visual_indistinguishability") or {}
        out.update(
            {
                "v32_visual_support_set_accuracy": visual.get(
                    "visual_support_set_accuracy"
                ),
                "v32_visually_indistinguishable_near_miss_fraction": visual.get(
                    "visually_indistinguishable_near_miss_fraction"
                ),
                "v32_true_model_error_fraction": visual.get(
                    "true_model_error_fraction"
                ),
                "v32_clear_negative_rejection_fraction": visual.get(
                    "clear_negative_rejection_fraction"
                ),
                "v32_mean_indistinguishable_gap": visual.get(
                    "mean_indistinguishable_gap"
                ),
            }
        )
    return out


def _ci(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted({key for sample in samples for key in sample})
    out: Dict[str, Any] = {}
    for key in keys:
        vals = [
            float(sample[key])
            for sample in samples
            if isinstance(sample.get(key), (int, float))
        ]
        out[key] = {
            "mean": _mean(vals),
            "p2_5": _percentile(vals, 2.5),
            "p50": _percentile(vals, 50.0),
            "p97_5": _percentile(vals, 97.5),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--score-key", default="iac_consistency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(_repo_root()))
    from benchmark_wam import _summary  # type: ignore

    rows = _load_jsonl(Path(args.scores))
    groups = _group_rows(rows, args.group_key)
    rng = random.Random(int(args.seed))
    point = _compact(_summary(rows, args.wam_key, args.group_key, args.score_key))
    samples: List[Dict[str, Any]] = []
    for _ in range(int(args.num_bootstrap)):
        boot_rows: List[Dict[str, Any]] = []
        for draw_idx in range(len(groups)):
            group = rng.choice(groups)
            base_gid = _group_id(group[0], args.group_key)
            boot_gid = f"{draw_idx:04d}__{base_gid}"
            boot_rows.extend(_rename_group(row, boot_gid, args.group_key) for row in group)
        samples.append(_compact(_summary(boot_rows, args.wam_key, args.group_key, args.score_key)))

    out = {
        "score_key": args.score_key,
        "num_groups": len(groups),
        "num_rows": len(rows),
        "num_bootstrap": int(args.num_bootstrap),
        "seed": int(args.seed),
        "point": point,
        "ci95": _ci(samples),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

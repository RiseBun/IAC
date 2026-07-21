"""Bootstrap confidence intervals for IAC-PathBench score JSONL files.

Rows are resampled by candidate group, not by individual candidate row. This
keeps candidate ranking metrics meaningful while estimating uncertainty across
driving scenes/groups.
"""

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


def _percentile(values: List[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(sum(values) / len(values))


def _group_id(row: Dict[str, Any], group_key: str) -> str:
    value = row.get(group_key) or row.get("anchor_id") or row.get("group_id")
    if value is not None:
        return str(value)
    return str(row.get("sample_id", "unknown"))


def _group_rows(rows: List[Dict[str, Any]], group_key: str) -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for row in rows:
        gid = _group_id(row, group_key)
        if gid not in grouped:
            order.append(gid)
        grouped[gid].append(row)
    return [grouped[gid] for gid in order]


def _compact(summary: Dict[str, Any]) -> Dict[str, Any]:
    traj = summary.get("trajectory_specific_causal_metrics", {})
    pos = traj.get("positive_rows", {})
    return {
        "top1": summary.get("ranking", {}).get("top1_hit_rate"),
        "mrr": summary.get("ranking", {}).get("mrr"),
        "best_balanced_accuracy": (
            summary.get("overall", {})
            .get("consistency_threshold_sweep", {})
            .get("best_balanced_accuracy", {})
            .get("balanced_accuracy")
        ),
        "path_minus_sky": summary.get("path_causal_metrics", {}).get(
            "mean_path_minus_sky_delta"
        ),
        "positive_exact_exclusive_delta": pos.get(
            "mean_candidate_minus_wrong_exclusive_delta"
        ),
        "positive_exact_exclusive_win_fraction": pos.get(
            "candidate_exclusive_delta_gt_wrong_fraction"
        ),
    }


def _ci(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted({key for sample in samples for key in sample.keys()})
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


def _rename_group(row: Dict[str, Any], gid: str, group_key: str) -> Dict[str, Any]:
    out = dict(row)
    out[group_key] = gid
    out["group_id"] = gid
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", help="Score JSONL to bootstrap.")
    parser.add_argument("--primary-scores", help="Primary score JSONL for fusion.")
    parser.add_argument("--aux-scores", help="Aux/path score JSONL for fusion.")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--wam-key", default="wam_name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(_repo_root()))
    from benchmark_wam import _summary  # type: ignore

    if args.primary_scores and args.aux_scores:
        from tools.sweep_iac_fused_scores import _fuse_rows  # type: ignore

        rows = _fuse_rows(
            _load_jsonl(Path(args.primary_scores)),
            _load_jsonl(Path(args.aux_scores)),
            float(args.alpha),
        )
        score_key = f"fused_alpha_{args.alpha:g}"
    elif args.scores:
        rows = _load_jsonl(Path(args.scores))
        score_key = "iac_consistency"
    else:
        raise ValueError("provide either --scores or --primary-scores + --aux-scores")

    groups = _group_rows(rows, args.group_key)
    rng = random.Random(int(args.seed))
    point = _compact(_summary(rows, args.wam_key, args.group_key, score_key))
    samples: List[Dict[str, Any]] = []
    for _ in range(int(args.num_bootstrap)):
        boot_rows: List[Dict[str, Any]] = []
        for draw_idx in range(len(groups)):
            group = rng.choice(groups)
            base_gid = _group_id(group[0], args.group_key)
            boot_gid = f"{draw_idx:04d}__{base_gid}"
            boot_rows.extend(_rename_group(row, boot_gid, args.group_key) for row in group)
        samples.append(
            _compact(_summary(boot_rows, args.wam_key, args.group_key, score_key))
        )

    out = {
        "score_key": score_key,
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


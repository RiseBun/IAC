"""Sweep fused IAC scores from two benchmark score JSONL files.

The primary score is usually consistency_logit-derived `iac_consistency`.
The auxiliary score is usually path_evidence_logit-derived `iac_consistency`.
For every alpha, this tool combines probabilities in logit space:

    fused_logit = logit(primary_prob) + alpha * logit(aux_prob)

All causal masked consistency fields are fused the same way, so existing
path/trajectory-specific metrics remain meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def _parse_alphas(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _consistency_fields(rows: Iterable[Dict[str, Any]]) -> List[str]:
    fields = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith("iac_consistency") and isinstance(value, (int, float)):
                fields.add(key)
    return sorted(fields)


def _fuse_rows(
    primary_rows: List[Dict[str, Any]],
    aux_rows: List[Dict[str, Any]],
    alpha: float,
) -> List[Dict[str, Any]]:
    if len(primary_rows) != len(aux_rows):
        raise ValueError(
            f"row count mismatch: primary={len(primary_rows)} aux={len(aux_rows)}"
        )
    fields = sorted(
        set(_consistency_fields(primary_rows)) & set(_consistency_fields(aux_rows))
    )
    fused: List[Dict[str, Any]] = []
    for idx, (primary, aux) in enumerate(zip(primary_rows, aux_rows)):
        group_a = primary.get("group_id") or primary.get("anchor_id") or primary.get("sample_id")
        group_b = aux.get("group_id") or aux.get("anchor_id") or aux.get("sample_id")
        if group_a != group_b:
            raise ValueError(f"row {idx} group mismatch: {group_a!r} vs {group_b!r}")
        row = dict(primary)
        for field in fields:
            row[field] = _sigmoid(_logit(primary[field]) + alpha * _logit(aux[field]))
        _recompute_delta_fields(row)
        fused.append(row)
    return fused


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
        row["wrong_path_delta"] = score - float(
            row["iac_consistency_wrong_path_masked"]
        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-scores", required=True, help="Main score JSONL")
    parser.add_argument("--aux-scores", required=True, help="Aux/path-head score JSONL")
    parser.add_argument(
        "--alphas",
        default="0,0.02,0.05,0.1,0.15,0.2,0.3,0.5,1.0",
        help="Comma-separated alpha values.",
    )
    parser.add_argument("--output", required=True, help="Output summary JSON")
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--group-key", default="group_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import sys

    sys.path.insert(0, str(_repo_root()))
    from benchmark_wam import _summary  # type: ignore

    primary_rows = _load_rows(Path(args.primary_scores))
    aux_rows = _load_rows(Path(args.aux_scores))
    results: List[Dict[str, Any]] = []
    for alpha in _parse_alphas(args.alphas):
        rows = _fuse_rows(primary_rows, aux_rows, alpha)
        summary = _summary(
            rows,
            args.wam_key,
            args.group_key,
            consistency_score_key=f"fused_consistency_plus_{alpha:g}_path_evidence",
        )
        traj = summary.get("trajectory_specific_causal_metrics", {})
        pos = traj.get("positive_rows", {})
        result = {
            "alpha": alpha,
            "summary": summary,
            "compact": {
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
            },
        }
        results.append(result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "primary_scores": args.primary_scores,
                "aux_scores": args.aux_scores,
                "alphas": _parse_alphas(args.alphas),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    for result in results:
        compact = result["compact"]
        print(
            f"alpha={result['alpha']:g} "
            f"top1={compact['top1']} "
            f"mrr={compact['mrr']} "
            f"bal={compact['best_balanced_accuracy']} "
            f"path_sky={compact['path_minus_sky']} "
            f"pos_excl={compact['positive_exact_exclusive_delta']} "
            f"pos_win={compact['positive_exact_exclusive_win_fraction']}"
        )
    print(f"summary={out_path}")


if __name__ == "__main__":
    main()

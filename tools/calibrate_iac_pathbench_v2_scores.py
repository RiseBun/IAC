"""Calibrate IAC ranking scores with path-grounded evidence.

This is a post-hoc scoring layer for IAC-PathBench v2. It keeps the model
outputs unchanged, then adjusts the final ranking probability using causal
path evidence already computed by benchmark_wam.py:

    calibrated_logit = logit(iac_consistency) + beta * evidence

where evidence is candidate-vs-wrong exact-path delta by default, with an
optional path-minus-sky term. This tests whether failures are caused by weak
score fusion rather than lack of path evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _consistency_fields(rows: Iterable[Dict[str, Any]]) -> List[str]:
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
    out: List[Dict[str, Any]] = []
    for idx, (primary, aux) in enumerate(zip(primary_rows, aux_rows)):
        group_a = primary.get("group_id") or primary.get("anchor_id") or primary.get("sample_id")
        group_b = aux.get("group_id") or aux.get("anchor_id") or aux.get("sample_id")
        if group_a != group_b:
            raise ValueError(f"row {idx} group mismatch: {group_a!r} vs {group_b!r}")
        row = dict(primary)
        for field in fields:
            row[field] = _sigmoid(_logit(primary[field]) + alpha * _logit(aux[field]))
        _recompute_delta_fields(row)
        out.append(row)
    return out


def _path_evidence(row: Dict[str, Any], path_minus_sky_weight: float) -> float:
    exact = row.get("candidate_minus_wrong_exclusive_path_delta")
    if exact is None:
        exact = row.get("candidate_minus_wrong_path_delta")
    exact_value = float(exact) if exact is not None else 0.0
    path_minus_sky = float(row.get("path_minus_sky_delta") or 0.0)
    return exact_value + float(path_minus_sky_weight) * path_minus_sky


def _calibrate_rows(
    rows: List[Dict[str, Any]],
    *,
    beta: float,
    path_minus_sky_weight: float,
    evidence_clamp: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        calibrated = dict(row)
        evidence = _path_evidence(calibrated, path_minus_sky_weight)
        evidence = max(-float(evidence_clamp), min(float(evidence_clamp), evidence))
        calibrated["iac_consistency_before_path_calibration"] = float(
            calibrated["iac_consistency"]
        )
        calibrated["iac_path_calibration_evidence"] = evidence
        calibrated["iac_path_calibration_beta"] = float(beta)
        calibrated["iac_consistency"] = _sigmoid(
            _logit(calibrated["iac_consistency"]) + float(beta) * evidence
        )
        _recompute_delta_fields(calibrated)
        out.append(calibrated)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", help="Already-scored JSONL.")
    parser.add_argument("--primary-scores", help="Main score JSONL.")
    parser.add_argument("--aux-scores", help="Aux/path-head score JSONL.")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--path-minus-sky-weight", type=float, default=0.0)
    parser.add_argument("--evidence-clamp", type=float, default=0.25)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--group-key", default="group_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(_repo_root()))
    from benchmark_wam import _summary  # type: ignore

    if args.scores:
        rows = _load_rows(Path(args.scores))
    else:
        if not args.primary_scores or not args.aux_scores:
            raise SystemExit("provide either --scores or both --primary-scores/--aux-scores")
        rows = _fuse_rows(
            _load_rows(Path(args.primary_scores)),
            _load_rows(Path(args.aux_scores)),
            args.alpha,
        )
    calibrated = _calibrate_rows(
        rows,
        beta=args.beta,
        path_minus_sky_weight=args.path_minus_sky_weight,
        evidence_clamp=args.evidence_clamp,
    )

    score_path = Path(args.output_scores)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", encoding="utf-8") as f:
        for row in calibrated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summary(
        calibrated,
        args.wam_key,
        args.group_key,
        consistency_score_key=(
            f"path_calibrated_alpha_{args.alpha:g}_beta_{args.beta:g}"
        ),
    )
    summary["path_calibration"] = {
        "alpha": args.alpha,
        "beta": args.beta,
        "path_minus_sky_weight": args.path_minus_sky_weight,
        "evidence_clamp": args.evidence_clamp,
    }
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["iac_pathbench_v2"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Train a lightweight score calibrator for IAC-PathBench v2.

The calibrator is intentionally small: a logistic linear model over already
computed IAC scores and path-grounded evidence. It turns the post-hoc beta
sweep into a reproducible learned fusion layer without re-training DINOv2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


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
        group_a = primary.get("group_id") or primary.get("anchor_id") or primary.get("sample_id")
        group_b = aux.get("group_id") or aux.get("anchor_id") or aux.get("sample_id")
        if group_a != group_b:
            raise ValueError(f"row {idx} group mismatch: {group_a!r} vs {group_b!r}")
        row = dict(primary)
        for field in fields:
            row[field] = _sigmoid(_logit(primary[field]) + alpha * _logit(aux[field]))
        row["path_head_iac_consistency"] = float(aux["iac_consistency"])
        _recompute_delta_fields(row)
        fused.append(row)
    return fused


def _exact_evidence(row: Dict[str, Any]) -> float:
    value = row.get("candidate_minus_wrong_exclusive_path_delta")
    if value is None:
        value = row.get("candidate_minus_wrong_path_delta")
    return float(value or 0.0)


def _features(row: Dict[str, Any], feature_names: Sequence[str]) -> List[float]:
    values: List[float] = []
    for name in feature_names:
        if name == "bias":
            values.append(1.0)
        elif name == "main_logit":
            values.append(_logit(float(row["iac_consistency"])))
        elif name == "path_head_logit":
            values.append(_logit(float(row.get("path_head_iac_consistency", row["iac_consistency"]))))
        elif name == "exact_path_delta":
            values.append(_exact_evidence(row))
        elif name == "path_minus_sky_delta":
            values.append(float(row.get("path_minus_sky_delta") or 0.0))
        else:
            raise ValueError(f"unknown feature: {name}")
    return values


def _build_xy(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([_features(row, feature_names) for row in rows], dtype=torch.float32)
    y = torch.tensor([float(row.get("consistency_label", row.get("label", 0.0))) for row in rows], dtype=torch.float32)
    return x, y


def _train_linear(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    *,
    lr: float,
    steps: int,
    l2: float,
    pos_weight: float,
) -> torch.Tensor:
    x, y = _build_xy(rows, feature_names)
    w = torch.zeros((x.size(1),), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        if "main_logit" in feature_names:
            w[feature_names.index("main_logit")] = 1.0
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=l2)
    pw = torch.tensor(float(pos_weight), dtype=torch.float32)
    for _ in range(int(steps)):
        logits = x @ w
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return w.detach()


def _apply_calibrator(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    weights: torch.Tensor,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        calibrated = dict(row)
        feats = torch.tensor(_features(calibrated, feature_names), dtype=torch.float32)
        logit = float((feats * weights).sum().item())
        calibrated["iac_consistency_before_learned_calibration"] = float(
            calibrated["iac_consistency"]
        )
        calibrated["iac_consistency"] = _sigmoid(logit)
        calibrated["iac_learned_calibration_logit"] = logit
        _recompute_delta_fields(calibrated)
        out.append(calibrated)
    return out


def _load_score_pair(primary: str, aux: str | None, alpha: float) -> List[Dict[str, Any]]:
    primary_rows = _load_rows(Path(primary))
    if aux is None:
        return primary_rows
    return _fuse_rows(primary_rows, _load_rows(Path(aux)), alpha)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-primary-scores", required=True)
    parser.add_argument("--train-aux-scores")
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        help=(
            "Evaluation spec label:primary_scores[:aux_scores]. Repeat for "
            "regular/low_iou/holdout."
        ),
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--features",
        default="bias,main_logit,path_head_logit,exact_path_delta,path_minus_sky_delta",
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=6.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--group-key", default="group_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(_repo_root()))
    from benchmark_wam import _summary  # type: ignore

    feature_names = [item.strip() for item in args.features.split(",") if item.strip()]
    train_rows = _load_score_pair(
        args.train_primary_scores,
        args.train_aux_scores,
        args.alpha,
    )
    weights = _train_linear(
        train_rows,
        feature_names,
        lr=args.lr,
        steps=args.steps,
        l2=args.l2,
        pos_weight=args.pos_weight,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "alpha": args.alpha,
        "features": feature_names,
        "weights": [float(v) for v in weights.tolist()],
        "train_primary_scores": args.train_primary_scores,
        "train_aux_scores": args.train_aux_scores,
        "lr": args.lr,
        "steps": args.steps,
        "l2": args.l2,
        "pos_weight": args.pos_weight,
    }
    (out_dir / "learned_calibrator.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("learned", json.dumps(model, ensure_ascii=False))
    for spec in args.eval:
        parts = spec.split(":")
        if len(parts) not in {2, 3}:
            raise SystemExit(f"bad --eval spec: {spec}")
        label, primary = parts[0], parts[1]
        aux = parts[2] if len(parts) == 3 else None
        rows = _load_score_pair(primary, aux, args.alpha)
        calibrated = _apply_calibrator(rows, feature_names, weights)
        score_path = out_dir / f"{label}_scores.jsonl"
        with score_path.open("w", encoding="utf-8") as f:
            for row in calibrated:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = _summary(
            calibrated,
            args.wam_key,
            args.group_key,
            consistency_score_key="learned_iac_pathbench_v2_calibrator",
        )
        summary["learned_calibrator"] = model
        summary_path = out_dir / f"{label}_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(label, json.dumps(summary["iac_pathbench_v2"], ensure_ascii=False))


if __name__ == "__main__":
    main()

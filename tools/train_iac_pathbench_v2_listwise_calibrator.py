"""Train a listwise, ambiguity-aware score calibrator for IAC-PathBench v2.

This is a lightweight post-hoc calibrator over existing score JSONL files. It
optimizes candidate-group ranking directly instead of treating every row as an
independent binary example. Near speed/lateral/heading perturbations can receive
small soft target mass when their trajectory is close to GT; hard negatives keep
zero target mass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _load_score_pair(primary: str, aux: str | None, alpha: float) -> List[Dict[str, Any]]:
    primary_rows = _load_rows(Path(primary))
    if aux is None:
        return primary_rows
    return _fuse_rows(primary_rows, _load_rows(Path(aux)), alpha)


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
    source = row.get("source_type") or row.get("sample_type") or row.get("action_type")
    if source is not None:
        suffix = f"__{source}"
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    if "__" in sample_id:
        return sample_id.rsplit("__", 1)[0]
    return sample_id


def _groups(rows: Sequence[Dict[str, Any]], group_key: str) -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gid = _group_id(row, group_key)
        if gid is not None:
            grouped[gid].append(row)
    return [items for items in grouped.values() if len(items) >= 2]


def _traj_xy(row: Dict[str, Any]) -> List[tuple[float, float]]:
    traj = row.get("candidate_traj") or []
    pts: List[tuple[float, float]] = []
    for item in traj:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                pts.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                pass
    return pts


def _traj_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float | None:
    aa = _traj_xy(a)
    bb = _traj_xy(b)
    n = min(len(aa), len(bb))
    if n == 0:
        return None
    dists = [
        math.hypot(aa[idx][0] - bb[idx][0], aa[idx][1] - bb[idx][1])
        for idx in range(n)
    ]
    return sum(dists) / len(dists)


def _exact_evidence(row: Dict[str, Any]) -> float:
    value = row.get("candidate_minus_wrong_exclusive_path_delta")
    if value is None:
        value = row.get("candidate_minus_wrong_path_delta")
    return float(value or 0.0)


def _features(
    row: Dict[str, Any],
    feature_names: Sequence[str],
    *,
    wam_key: str,
    near_sources: set[str],
    hard_sources: set[str],
) -> List[float]:
    source = _source(row, wam_key)
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
        elif name == "recovered_set_agreement":
            if row.get("recovered_set_agreement") is not None:
                values.append(float(row["recovered_set_agreement"]))
            elif row.get("recovered_set_minade") is not None:
                values.append(-float(row["recovered_set_minade"]))
            elif row.get("recovered_path_agreement") is not None:
                values.append(float(row["recovered_path_agreement"]))
            elif row.get("recovered_path_ade") is not None:
                values.append(-float(row["recovered_path_ade"]))
            else:
                values.append(0.0)
        elif name == "recovered_set_supported":
            if row.get("recovered_set_supported") is not None:
                values.append(float(row["recovered_set_supported"]))
            else:
                values.append(0.0)
        elif name == "is_near_source":
            values.append(1.0 if source in near_sources else 0.0)
        elif name == "is_hard_negative_source":
            values.append(1.0 if source in hard_sources else 0.0)
        elif name == "is_gt_source":
            values.append(1.0 if source == "gt_pos" else 0.0)
        else:
            raise ValueError(f"unknown feature: {name}")
    return values


def _soft_targets_for_group(
    group: Sequence[Dict[str, Any]],
    *,
    wam_key: str,
    near_sources: set[str],
    near_soft_weight: float,
    min_near_soft_weight: float,
    distance_tau: float,
) -> List[float]:
    positives = [row for row in group if _is_positive(row, wam_key)]
    if not positives:
        return [0.0 for _ in group]
    gt = positives[0]
    weights: List[float] = []
    for row in group:
        if row is gt:
            weights.append(1.0)
            continue
        source = _source(row, wam_key)
        if source not in near_sources:
            weights.append(0.0)
            continue
        distance = _traj_distance(row, gt)
        if distance is None:
            soft = min_near_soft_weight
        else:
            soft = math.exp(-distance / max(distance_tau, 1e-6))
            soft = max(min_near_soft_weight, soft)
        weights.append(float(near_soft_weight) * min(1.0, soft))
    total = sum(weights)
    if total <= 0.0:
        return [0.0 for _ in group]
    return [value / total for value in weights]


def _group_tensors(
    group: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    *,
    wam_key: str,
    near_sources: set[str],
    hard_sources: set[str],
    near_soft_weight: float,
    min_near_soft_weight: float,
    distance_tau: float,
) -> tuple[torch.Tensor, torch.Tensor, int | None, List[int], List[int]]:
    feats = [
        _features(
            row,
            feature_names,
            wam_key=wam_key,
            near_sources=near_sources,
            hard_sources=hard_sources,
        )
        for row in group
    ]
    targets = _soft_targets_for_group(
        group,
        wam_key=wam_key,
        near_sources=near_sources,
        near_soft_weight=near_soft_weight,
        min_near_soft_weight=min_near_soft_weight,
        distance_tau=distance_tau,
    )
    gt_idx = next((idx for idx, row in enumerate(group) if _is_positive(row, wam_key)), None)
    near_idxs = [
        idx for idx, row in enumerate(group)
        if idx != gt_idx and _source(row, wam_key) in near_sources
    ]
    hard_idxs = [
        idx for idx, row in enumerate(group)
        if idx != gt_idx and _source(row, wam_key) in hard_sources
    ]
    return (
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        gt_idx,
        near_idxs,
        hard_idxs,
    )


def _train_listwise(
    row_groups: Sequence[Sequence[Dict[str, Any]]],
    feature_names: Sequence[str],
    *,
    wam_key: str,
    near_sources: set[str],
    hard_sources: set[str],
    near_soft_weight: float,
    min_near_soft_weight: float,
    distance_tau: float,
    lr: float,
    steps: int,
    l2: float,
    hard_margin: float,
    near_margin: float,
    pairwise_weight: float,
) -> torch.Tensor:
    prepared = [
        _group_tensors(
            group,
            feature_names,
            wam_key=wam_key,
            near_sources=near_sources,
            hard_sources=hard_sources,
            near_soft_weight=near_soft_weight,
            min_near_soft_weight=min_near_soft_weight,
            distance_tau=distance_tau,
        )
        for group in row_groups
    ]
    prepared = [item for item in prepared if item[2] is not None and float(item[1].sum()) > 0.0]
    if not prepared:
        raise ValueError("no trainable candidate groups found")

    w = torch.zeros((len(feature_names),), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        if "main_logit" in feature_names:
            w[feature_names.index("main_logit")] = 1.0
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=l2)

    for _ in range(int(steps)):
        listwise_losses = []
        pairwise_losses = []
        for x, target, gt_idx, near_idxs, hard_idxs in prepared:
            logits = x @ w
            log_probs = F.log_softmax(logits, dim=0)
            listwise_losses.append(-(target * log_probs).sum())
            gt_logit = logits[int(gt_idx)]
            if hard_idxs:
                hard_logits = logits[torch.tensor(hard_idxs, dtype=torch.long)]
                pairwise_losses.append(F.softplus(hard_margin - gt_logit + hard_logits).mean())
            if near_idxs and near_margin > 0.0:
                near_logits = logits[torch.tensor(near_idxs, dtype=torch.long)]
                pairwise_losses.append(F.softplus(near_margin - gt_logit + near_logits).mean())
        loss = torch.stack(listwise_losses).mean()
        if pairwise_losses and pairwise_weight > 0.0:
            loss = loss + float(pairwise_weight) * torch.stack(pairwise_losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return w.detach()


def _apply_calibrator(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    weights: torch.Tensor,
    *,
    wam_key: str,
    near_sources: set[str],
    hard_sources: set[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        calibrated = dict(row)
        feats = torch.tensor(
            _features(
                calibrated,
                feature_names,
                wam_key=wam_key,
                near_sources=near_sources,
                hard_sources=hard_sources,
            ),
            dtype=torch.float32,
        )
        logit = float((feats * weights).sum().item())
        calibrated["iac_consistency_before_listwise_calibration"] = float(
            calibrated["iac_consistency"]
        )
        calibrated["iac_consistency"] = _sigmoid(logit)
        calibrated["iac_listwise_calibration_logit"] = logit
        _recompute_delta_fields(calibrated)
        out.append(calibrated)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-primary-scores", required=True)
    parser.add_argument("--train-aux-scores")
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        help="Evaluation spec label:primary_scores[:aux_scores]. Repeat per split.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--features",
        default=(
            "bias,main_logit,path_head_logit,exact_path_delta,path_minus_sky_delta"
        ),
    )
    parser.add_argument("--near-sources", default="perturb_speed,perturb_lateral,perturb_heading")
    parser.add_argument("--hard-sources", default="image_swap,time_shift_future,traj_swap,reverse_traj")
    parser.add_argument("--near-soft-weight", type=float, default=0.20)
    parser.add_argument("--min-near-soft-weight", type=float, default=0.02)
    parser.add_argument("--distance-tau", type=float, default=2.0)
    parser.add_argument("--hard-margin", type=float, default=1.0)
    parser.add_argument("--near-margin", type=float, default=0.05)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--group-key", default="group_id")
    return parser.parse_args()


def _parse_eval_spec(spec: str) -> tuple[str, str, str | None]:
    if "=" in spec:
        label, rest = spec.split("=", 1)
        paths = rest.split(",")
        if len(paths) not in {1, 2}:
            raise SystemExit(f"bad --eval spec: {spec}")
        return label, paths[0], paths[1] if len(paths) == 2 else None
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise SystemExit(f"bad --eval spec: {spec}")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else None


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(_repo_root()))
    from benchmark_wam import _summary  # type: ignore

    feature_names = [item.strip() for item in args.features.split(",") if item.strip()]
    near_sources = {item.strip() for item in args.near_sources.split(",") if item.strip()}
    hard_sources = {item.strip() for item in args.hard_sources.split(",") if item.strip()}

    train_rows = _load_score_pair(args.train_primary_scores, args.train_aux_scores, args.alpha)
    weights = _train_listwise(
        _groups(train_rows, args.group_key),
        feature_names,
        wam_key=args.wam_key,
        near_sources=near_sources,
        hard_sources=hard_sources,
        near_soft_weight=args.near_soft_weight,
        min_near_soft_weight=args.min_near_soft_weight,
        distance_tau=args.distance_tau,
        lr=args.lr,
        steps=args.steps,
        l2=args.l2,
        hard_margin=args.hard_margin,
        near_margin=args.near_margin,
        pairwise_weight=args.pairwise_weight,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "kind": "iac_pathbench_v2_listwise_tie_aware_calibrator",
        "alpha": args.alpha,
        "features": feature_names,
        "weights": [float(value) for value in weights.tolist()],
        "train_primary_scores": args.train_primary_scores,
        "train_aux_scores": args.train_aux_scores,
        "near_sources": sorted(near_sources),
        "hard_sources": sorted(hard_sources),
        "near_soft_weight": args.near_soft_weight,
        "min_near_soft_weight": args.min_near_soft_weight,
        "distance_tau": args.distance_tau,
        "hard_margin": args.hard_margin,
        "near_margin": args.near_margin,
        "pairwise_weight": args.pairwise_weight,
        "lr": args.lr,
        "steps": args.steps,
        "l2": args.l2,
    }
    (out_dir / "listwise_calibrator.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("learned", json.dumps(model, ensure_ascii=False))

    for spec in args.eval:
        label, primary, aux = _parse_eval_spec(spec)
        rows = _load_score_pair(primary, aux, args.alpha)
        calibrated = _apply_calibrator(
            rows,
            feature_names,
            weights,
            wam_key=args.wam_key,
            near_sources=near_sources,
            hard_sources=hard_sources,
        )
        score_path = out_dir / f"{label}_scores.jsonl"
        _write_jsonl(score_path, calibrated)
        summary = _summary(
            calibrated,
            args.wam_key,
            args.group_key,
            consistency_score_key="listwise_iac_pathbench_v2_calibrator",
        )
        summary["listwise_calibrator"] = model
        summary_path = out_dir / f"{label}_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(label, json.dumps(summary["iac_pathbench_v2"], ensure_ascii=False))


if __name__ == "__main__":
    main()

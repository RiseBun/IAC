"""Train a conservative recovered-set agreement calibrator.

The recovered-set probe predicts K future-supported trajectories. This tool
learns a monotonic, listwise scorer over candidate-vs-set geometry, then writes a
WAM score JSONL whose ``iac_consistency`` is the learned agreement score.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


def _load_rows(path: Path) -> List[Dict[str, Any]]:
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


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


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


def _groups(rows: Sequence[Dict[str, Any]], group_key: str) -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gid = _group_id(row, group_key)
        if gid is not None:
            grouped[gid].append(row)
    return [items for items in grouped.values() if len(items) >= 2]


def _traj_xy(row: Dict[str, Any]) -> List[tuple[float, float]]:
    pts: List[tuple[float, float]] = []
    for item in row.get("candidate_traj") or []:
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
    return sum(math.hypot(aa[i][0] - bb[i][0], aa[i][1] - bb[i][1]) for i in range(n)) / n


def _soft_targets(
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
        if _source(row, wam_key) not in near_sources:
            weights.append(0.0)
            continue
        distance = _traj_distance(row, gt)
        soft = min_near_soft_weight if distance is None else max(
            min_near_soft_weight,
            math.exp(-distance / max(distance_tau, 1e-6)),
        )
        weights.append(float(near_soft_weight) * min(1.0, soft))
    total = sum(weights)
    return [value / total for value in weights] if total > 0.0 else [0.0 for _ in group]


FEATURE_NAMES = [
    "neg_minade",
    "neg_best_fde",
    "neg_heading",
    "neg_progress",
    "path_iou",
    "support_margin",
    "supported_flag",
    "base_agreement_logit",
    "topmode_margin",
]


def _safe_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _features(row: Dict[str, Any]) -> List[float]:
    minade = _safe_float(row, "recovered_set_minade")
    fde = _safe_float(row, "recovered_set_best_mode_fde")
    heading = _safe_float(row, "recovered_set_heading_error")
    progress = _safe_float(row, "recovered_set_progress_error")
    iou = _safe_float(row, "recovered_set_path_iou")
    radius = max(_safe_float(row, "recovered_set_conformal_radius", 1.0), 1e-6)
    topmode_ade = _safe_float(row, "recovered_set_topmode_ade")
    agreement_logit = _safe_float(row, "recovered_set_agreement_logit")
    return [
        -minade / 2.0,
        -fde / 4.0,
        -heading / 0.75,
        -progress / 4.0,
        iou,
        (radius - minade) / radius,
        _safe_float(row, "recovered_set_supported"),
        agreement_logit / 4.0,
        (topmode_ade - minade) / 2.0,
    ]


def _standardize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    return mean, std


def _prepare(
    groups: Sequence[Sequence[Dict[str, Any]]],
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    wam_key: str,
    near_sources: set[str],
    hard_sources: set[str],
    near_soft_weight: float,
    min_near_soft_weight: float,
    distance_tau: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for group in groups:
        targets = _soft_targets(
            group,
            wam_key=wam_key,
            near_sources=near_sources,
            near_soft_weight=near_soft_weight,
            min_near_soft_weight=min_near_soft_weight,
            distance_tau=distance_tau,
        )
        if sum(targets) <= 0.0:
            continue
        gt_idx = next((idx for idx, row in enumerate(group) if _is_positive(row, wam_key)), None)
        if gt_idx is None:
            continue
        sources = [_source(row, wam_key) for row in group]
        raw = torch.tensor([_features(row) for row in group], dtype=torch.float32)
        out.append(
            {
                "x": (raw - mean) / std,
                "target": torch.tensor(targets, dtype=torch.float32),
                "gt_idx": gt_idx,
                "hard_idxs": [i for i, src in enumerate(sources) if i != gt_idx and src in hard_sources],
                "near_idxs": [i for i, src in enumerate(sources) if i != gt_idx and src in near_sources],
            }
        )
    return out


def _score(x: torch.Tensor, raw_weight: torch.Tensor, bias: torch.Tensor, max_weight: float) -> torch.Tensor:
    weights = float(max_weight) * torch.sigmoid(raw_weight)
    return x @ weights + bias


def _train(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    near_sources = {item.strip() for item in args.near_sources.split(",") if item.strip()}
    hard_sources = {item.strip() for item in args.hard_sources.split(",") if item.strip()}
    all_x = torch.tensor([_features(row) for row in rows], dtype=torch.float32)
    mean, std = _standardize(all_x)
    prepared = _prepare(
        _groups(rows, args.group_key),
        mean,
        std,
        wam_key=args.wam_key,
        near_sources=near_sources,
        hard_sources=hard_sources,
        near_soft_weight=args.near_soft_weight,
        min_near_soft_weight=args.min_near_soft_weight,
        distance_tau=args.distance_tau,
    )
    if not prepared:
        raise ValueError("no trainable groups")
    raw_weight = torch.full((len(FEATURE_NAMES),), -2.0, dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.AdamW([raw_weight, bias], lr=args.lr, weight_decay=args.weight_decay)
    history: List[Dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        losses = []
        for item in prepared:
            logits = _score(item["x"], raw_weight, bias, args.max_weight)
            log_probs = F.log_softmax(logits, dim=0)
            loss = -(item["target"] * log_probs).sum()
            gt = logits[int(item["gt_idx"])]
            pair_losses = []
            if item["hard_idxs"]:
                hard = logits[torch.tensor(item["hard_idxs"], dtype=torch.long)]
                pair_losses.append(F.softplus(float(args.hard_margin) - gt + hard).mean())
            if item["near_idxs"] and args.near_margin > 0.0:
                near = logits[torch.tensor(item["near_idxs"], dtype=torch.long)]
                pair_losses.append(F.softplus(float(args.near_margin) - gt + near).mean())
            if pair_losses:
                loss = loss + float(args.pairwise_weight) * torch.stack(pair_losses).mean()
            losses.append(loss)
        total = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if step == 1 or step % max(1, int(args.steps) // 10) == 0 or step == int(args.steps):
            weights = float(args.max_weight) * torch.sigmoid(raw_weight.detach())
            record = {
                "step": step,
                "loss": float(total.detach().item()),
                "bias": float(bias.detach().item()),
                "weights": {name: float(value) for name, value in zip(FEATURE_NAMES, weights.tolist())},
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    weights = float(args.max_weight) * torch.sigmoid(raw_weight.detach())
    return {
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "bias": float(bias.detach().item()),
        "mean": mean,
        "std": std,
        "history": history,
        "near_sources": sorted(near_sources),
        "hard_sources": sorted(hard_sources),
    }


def _apply(rows: Sequence[Dict[str, Any]], model: Dict[str, Any]) -> List[Dict[str, Any]]:
    weights: torch.Tensor = model["weights"]
    mean: torch.Tensor = model["mean"]
    std: torch.Tensor = model["std"]
    bias = float(model["bias"])
    out: List[Dict[str, Any]] = []
    for row in rows:
        x = torch.tensor(_features(row), dtype=torch.float32)
        logit = float((((x - mean) / std) * weights).sum().item() + bias)
        new_row = dict(row)
        new_row["base_recovered_set_agreement"] = row.get("recovered_set_agreement")
        new_row["learned_recovered_agreement_logit"] = logit
        new_row["learned_recovered_agreement"] = _sigmoid(logit)
        new_row["iac_consistency"] = new_row["learned_recovered_agreement"]
        new_row["score_fusion_label"] = "learned_recovered_agreement"
        out.append(new_row)
    return out


def _parse_eval(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"bad --eval spec: {spec}")
    label, path = spec.split("=", 1)
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scores", required=True)
    parser.add_argument("--eval", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--near-sources", default="perturb_speed,perturb_lateral,perturb_heading")
    parser.add_argument(
        "--hard-sources",
        default="image_swap,time_shift_future,traj_swap,reverse_traj,high_pdm_image_mismatch",
    )
    parser.add_argument("--near-soft-weight", type=float, default=0.20)
    parser.add_argument("--min-near-soft-weight", type=float, default=0.02)
    parser.add_argument("--distance-tau", type=float, default=2.0)
    parser.add_argument("--hard-margin", type=float, default=1.0)
    parser.add_argument("--near-margin", type=float, default=0.05)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--max-weight", type=float, default=1.25)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = _load_rows(Path(args.train_scores))
    model = _train(train_rows, args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        "kind": "conservative_recovered_agreement_calibrator",
        "train_scores": args.train_scores,
        "feature_names": model["feature_names"],
        "weights": [float(v) for v in model["weights"].tolist()],
        "bias": model["bias"],
        "mean": [float(v) for v in model["mean"].tolist()],
        "std": [float(v) for v in model["std"].tolist()],
        "history": model["history"],
        "near_sources": model["near_sources"],
        "hard_sources": model["hard_sources"],
    }
    (out_dir / "recovered_agreement_calibrator.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for spec in args.eval:
        label, path = _parse_eval(spec)
        rows = _load_rows(path)
        scored = _apply(rows, model)
        _write_jsonl(out_dir / f"{label}_scores.jsonl", scored)
        print(label, len(scored), out_dir / f"{label}_scores.jsonl")


if __name__ == "__main__":
    main()

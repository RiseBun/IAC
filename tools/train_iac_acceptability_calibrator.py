"""Train a lightweight IAC acceptability calibrator.

The calibrator is a post-hoc scoring layer over existing score JSONLs. It
learns the metric definition we actually want for WAM evaluation:

    acceptable: gt_pos + visually plausible near-action perturbations
    hard mismatch: image/time/trajectory swaps

It does not use source labels as input features. Labels are used only for
training/evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
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


def _logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _safe_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _group_id(row: Dict[str, Any], group_key: str) -> str:
    if "_calibrator_group_id" in row:
        return str(row["_calibrator_group_id"])
    return str(row.get(group_key) or row.get("anchor_id") or row.get("sample_id"))


def _source(row: Dict[str, Any], source_key: str) -> str:
    for key in (source_key, "source_type", "action_type", "wam_name", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _traj_xy(row: Dict[str, Any]) -> List[tuple[float, float]]:
    pts: List[tuple[float, float]] = []
    for item in row.get("candidate_traj") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                pts.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                pass
    return pts


def _traj_features(row: Dict[str, Any]) -> List[float]:
    pts = _traj_xy(row)
    if len(pts) < 2:
        return [0.0] * 10
    origin = [(0.0, 0.0)] + pts[:-1]
    steps = [
        math.hypot(x - px, y - py)
        for (x, y), (px, py) in zip(pts, origin)
    ]
    final_x, final_y = pts[-1]
    path_len = sum(steps)
    direct = math.hypot(final_x, final_y)
    headings = [
        math.atan2(y - py, x - px)
        for (x, y), (px, py) in zip(pts, origin)
    ]
    heading_delta = headings[-1] - headings[0]
    heading_delta = math.atan2(math.sin(heading_delta), math.cos(heading_delta))
    return [
        final_x / 40.0,
        final_y / 10.0,
        abs(final_y) / 10.0,
        path_len / 40.0,
        direct / max(path_len, 1e-4),
        sum(steps) / max(len(steps), 1) / 5.0,
        max(steps) / 5.0,
        heading_delta / math.pi,
        abs(heading_delta) / math.pi,
        max(abs(y) for _, y in pts) / 10.0,
    ]


def _row_features(
    primary: Dict[str, Any],
    aux_rows: Sequence[Dict[str, Any]],
) -> List[float]:
    scores = [_safe_float(primary, "iac_consistency", 0.5)]
    scores.extend(_safe_float(row, "iac_consistency", 0.5) for row in aux_rows)
    logits = [_logit(score) for score in scores]
    features: List[float] = []
    features.extend(logits)
    features.extend(scores)
    if len(logits) > 1:
        features.extend(logits[0] - value for value in logits[1:])
        features.extend(abs(logits[0] - value) for value in logits[1:])
    for key in (
        "recovered_set_agreement",
        "recovered_set_minade",
        "recovered_set_topmode_ade",
        "recovered_set_best_mode_fde",
        "recovered_set_heading_error",
        "recovered_set_progress_error",
        "recovered_set_path_iou",
        "recovered_set_supported",
        "path_minus_sky_delta",
        "candidate_minus_wrong_path_delta",
        "candidate_minus_wrong_exclusive_path_delta",
    ):
        features.append(_safe_float(primary, key, 0.0))
    features.extend(_traj_features(primary))
    return features


def _aligned_rows(
    primary_path: Path,
    aux_paths: Sequence[Path],
    group_key: str,
) -> tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    primary_rows = _load_rows(primary_path)
    aux_all = [_load_rows(path) for path in aux_paths]
    for path, rows in zip(aux_paths, aux_all):
        if len(rows) != len(primary_rows):
            raise ValueError(f"row count mismatch for {path}: {len(rows)}")
    for idx, primary in enumerate(primary_rows):
        group = _group_id(primary, group_key)
        sample_id = primary.get("sample_id")
        for path, rows in zip(aux_paths, aux_all):
            other = rows[idx]
            if _group_id(other, group_key) != group:
                raise ValueError(f"{path} row {idx} group mismatch")
            if sample_id is not None and other.get("sample_id") != sample_id:
                raise ValueError(f"{path} row {idx} sample_id mismatch")
    return primary_rows, aux_all


def _dataset(
    primary_path: Path,
    aux_paths: Sequence[Path],
    *,
    group_key: str,
    source_key: str,
    acceptable_sources: set[str],
    hard_sources: set[str],
) -> tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, aux_all = _aligned_rows(primary_path, aux_paths, group_key)
    x_rows: List[List[float]] = []
    y: List[float] = []
    weights: List[float] = []
    kept: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        source = _source(row, source_key)
        if source in acceptable_sources:
            target = 1.0
            weight = 1.0 if source == "gt_pos" else 0.85
        elif source in hard_sources:
            target = 0.0
            weight = 1.2 if source in {"traj_swap", "time_shift_future"} else 1.0
        else:
            continue
        x_rows.append(_row_features(row, [aux[idx] for aux in aux_all]))
        y.append(target)
        weights.append(weight)
        kept.append(row)
    return (
        kept,
        torch.tensor(x_rows, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


class Calibrator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(in_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _build_group_pairs(
    rows: Sequence[Dict[str, Any]],
    labels: torch.Tensor,
    *,
    group_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[_group_id(row, group_key)].append(idx)
    pos_indices: List[int] = []
    neg_indices: List[int] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices, dtype=torch.long)
        g_labels = labels.index_select(0, idx)
        pos = g_labels > 0.5
        neg = ~pos
        if not bool(pos.any()) or not bool(neg.any()):
            continue
        group_pos = idx[pos].tolist()
        group_neg = idx[neg].tolist()
        for pos_idx in group_pos:
            for neg_idx in group_neg:
                pos_indices.append(int(pos_idx))
                neg_indices.append(int(neg_idx))
    return (
        torch.tensor(pos_indices, dtype=torch.long),
        torch.tensor(neg_indices, dtype=torch.long),
    )


def _group_pair_loss(
    logits: torch.Tensor,
    pos_idx: torch.Tensor,
    neg_idx: torch.Tensor,
    weights: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if pos_idx.numel() == 0:
        return logits.sum() * 0.0
    pos_idx = pos_idx.to(logits.device)
    neg_idx = neg_idx.to(logits.device)
    gap = logits.index_select(0, pos_idx) - logits.index_select(0, neg_idx)
    pair_weights = weights.index_select(0, neg_idx)
    return (F.relu(float(margin) - gap) * pair_weights).mean()


def _standardize(
    x: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mean is None:
        mean = x.mean(dim=0)
    if std is None:
        std = x.std(dim=0).clamp_min(1e-4)
    return (x - mean) / std, mean, std


def _evaluate_rows(
    rows: Sequence[Dict[str, Any]],
    probs: torch.Tensor,
    *,
    group_key: str,
    source_key: str,
    acceptable_sources: set[str],
    hard_sources: set[str],
) -> Dict[str, Any]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[_group_id(row, group_key)].append(idx)
    top_sources: Counter[str] = Counter()
    strict = 0
    acceptable = 0
    hard = 0
    n = 0
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        n += 1
        best = max(indices, key=lambda i: float(probs[i]))
        source = _source(rows[best], source_key)
        top_sources[source] += 1
        strict += int(source == "gt_pos")
        acceptable += int(source in acceptable_sources)
        hard += int(source in hard_sources)
    return {
        "num_groups": n,
        "strict_gt_top1": strict / n if n else None,
        "acceptable_top1": acceptable / n if n else None,
        "hard_mismatch_top1": hard / n if n else None,
        "top_sources": top_sources.most_common(),
    }


def _apply(
    model: Calibrator,
    rows_path: Path,
    aux_paths: Sequence[Path],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    group_key: str,
    source_key: str,
    acceptable_sources: set[str],
    hard_sources: set[str],
    output_scores: Path,
    device: torch.device,
) -> Dict[str, Any]:
    rows, aux_all = _aligned_rows(rows_path, aux_paths, group_key)
    x = torch.tensor(
        [
            _row_features(row, [aux[idx] for aux in aux_all])
            for idx, row in enumerate(rows)
        ],
        dtype=torch.float32,
        device=device,
    )
    x = (x - mean.to(device)) / std.to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu()
    out_rows: List[Dict[str, Any]] = []
    for row, prob in zip(rows, probs.tolist()):
        new_row = dict(row)
        new_row["base_iac_consistency"] = row.get("iac_consistency")
        new_row["iac_acceptability_calibrated"] = float(prob)
        new_row["iac_consistency"] = float(prob)
        out_rows.append(new_row)
    _write_jsonl(output_scores, out_rows)
    return _evaluate_rows(
        out_rows,
        probs,
        group_key=group_key,
        source_key=source_key,
        acceptable_sources=acceptable_sources,
        hard_sources=hard_sources,
    )


def _parse_paths(raw: str) -> List[Path]:
    if not raw:
        return []
    return [Path(item) for item in raw.split(",") if item]


def _parse_train_extra(raw: str) -> tuple[Path, List[Path]]:
    primary_raw, sep, aux_raw = raw.partition(":")
    if not sep:
        raise SystemExit(f"--train-extra must be PRIMARY:AUX1,AUX2, got {raw!r}")
    return Path(primary_raw), _parse_paths(aux_raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-primary", required=True)
    parser.add_argument("--train-aux", default="")
    parser.add_argument(
        "--train-extra",
        action="append",
        default=[],
        metavar="PRIMARY:AUX1,AUX2",
        help="Additional training split with the same feature layout.",
    )
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        metavar="NAME:PRIMARY:AUX1,AUX2",
        help="Evaluation split. AUX part may be empty.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.35)
    parser.add_argument("--pairwise-margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--source-key", default="source_type")
    parser.add_argument(
        "--acceptable-sources",
        default="gt_pos,perturb_speed,perturb_lateral,perturb_heading",
    )
    parser.add_argument(
        "--hard-sources",
        default="image_swap,time_shift_future,traj_swap,reverse_traj,high_pdm_image_mismatch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acceptable_sources = {
        item.strip() for item in args.acceptable_sources.split(",") if item.strip()
    }
    hard_sources = {
        item.strip() for item in args.hard_sources.split(",") if item.strip()
    }
    train_specs = [(Path(args.train_primary), _parse_paths(args.train_aux))]
    train_specs.extend(_parse_train_extra(raw) for raw in args.train_extra)
    train_rows: List[Dict[str, Any]] = []
    x_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    weight_parts: List[torch.Tensor] = []
    for split_idx, (primary_path, aux_paths) in enumerate(train_specs):
        rows_part, x_part, y_part, weights_part = _dataset(
            primary_path,
            aux_paths,
            group_key=args.group_key,
            source_key=args.source_key,
            acceptable_sources=acceptable_sources,
            hard_sources=hard_sources,
        )
        for row in rows_part:
            row["_calibrator_group_id"] = f"train{split_idx}:{_group_id(row, args.group_key)}"
        train_rows.extend(rows_part)
        x_parts.append(x_part)
        y_parts.append(y_part)
        weight_parts.append(weights_part)
    x = torch.cat(x_parts, dim=0)
    y = torch.cat(y_parts, dim=0)
    weights = torch.cat(weight_parts, dim=0)
    x, mean, std = _standardize(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Calibrator(x.shape[1], int(args.hidden_dim), float(args.dropout)).to(device)
    x = x.to(device)
    y = y.to(device)
    weights = weights.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    pair_pos_idx, pair_neg_idx = _build_group_pairs(
        train_rows,
        y.detach().cpu(),
        group_key=args.group_key,
    )
    pair_count = int(pair_pos_idx.numel())

    history: List[Dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        model.train()
        logits = model(x)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            y,
            weight=weights,
        )
        pair_loss = _group_pair_loss(
            logits,
            pair_pos_idx,
            pair_neg_idx,
            weights,
            margin=float(args.pairwise_margin),
        )
        loss = bce + float(args.pairwise_weight) * pair_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, int(args.steps) // 10) == 0:
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).cpu()
            record = {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "bce": float(bce.detach().cpu().item()),
                "pair_loss": float(pair_loss.detach().cpu().item()),
                "pair_count": pair_count,
                **{
                    f"train_{k}": v
                    for k, v in _evaluate_rows(
                        train_rows,
                        probs,
                        group_key=args.group_key,
                        source_key=args.source_key,
                        acceptable_sources=acceptable_sources,
                        hard_sources=hard_sources,
                    ).items()
                },
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False))

    eval_summary: Dict[str, Any] = {}
    for raw in args.eval:
        name, sep, rest = raw.partition(":")
        if not sep:
            raise SystemExit(f"--eval must be NAME:PRIMARY:AUX1,AUX2, got {raw!r}")
        primary_raw, sep2, aux_raw = rest.partition(":")
        aux_paths = _parse_paths(aux_raw if sep2 else "")
        split_dir = out_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        eval_summary[name] = _apply(
            model,
            Path(primary_raw),
            aux_paths,
            mean=mean,
            std=std,
            group_key=args.group_key,
            source_key=args.source_key,
            acceptable_sources=acceptable_sources,
            hard_sources=hard_sources,
            output_scores=split_dir / "calibrated_scores.jsonl",
            device=device,
        )

    metadata = {
        "kind": "iac_acceptability_calibrator",
        "train_primary": args.train_primary,
        "train_aux": [str(path) for path in train_specs[0][1]],
        "train_sets": [
            {
                "primary": str(primary_path),
                "aux": [str(path) for path in aux_paths],
            }
            for primary_path, aux_paths in train_specs
        ],
        "acceptable_sources": sorted(acceptable_sources),
        "hard_sources": sorted(hard_sources),
        "feature_dim": int(mean.numel()),
        "hidden_dim": int(args.hidden_dim),
        "history": history,
        "eval": eval_summary,
    }
    torch.save(
        {
            "model": model.cpu().state_dict(),
            "mean": mean,
            "std": std,
            "metadata": metadata,
        },
        out_dir / "iac_acceptability_calibrator.pt",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata["eval"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

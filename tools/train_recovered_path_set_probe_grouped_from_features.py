"""Train recovered-path set probe with group-level supported-set supervision.

The older set-probe trainer treats every supported trajectory row as an
independent single-target sample. When several supported perturbations share
the same future image, that makes the same visual input appear with different
targets. This trainer groups those rows and trains one prediction to cover the
whole supported trajectory set.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from train_recovered_path_probe import _trajectory_features
from train_recovered_path_set_probe_from_features import RecoveredPathSetProbe


DEFAULT_HARD_NEGATIVE_SOURCES = (
    "image_swap,"
    "time_shift,time_shift_future,time_shift_past,"
    "traj_swap,reverse,reverse_traj,high_pdm_image_mismatch"
)


def _load_jsonl(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_cache(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _row_source(row: Dict[str, Any]) -> str:
    for key in ("source_type", "action_type", "wam_name", "sample_type", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _row_group_id(row: Dict[str, Any]) -> str | None:
    value = row.get("group_id") or row.get("anchor_id")
    if value is not None:
        return str(value)
    sample_id = row.get("sample_id")
    if sample_id is None:
        return None
    sample_id = str(sample_id)
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def _traj_tensor(row: Dict[str, Any], steps: int, traj_dim: int) -> torch.Tensor | None:
    traj = row.get("candidate_traj")
    if traj is None:
        return None
    tensor = torch.tensor(traj, dtype=torch.float32)
    if tensor.ndim != 2:
        return None
    if tensor.shape[-1] < traj_dim:
        tensor = F.pad(tensor, (0, traj_dim - tensor.shape[-1]))
    tensor = tensor[:steps, :traj_dim]
    if tensor.shape[0] < steps:
        tensor = F.pad(tensor, (0, 0, 0, steps - tensor.shape[0]))
    return tensor


def _negative_targets_by_group(
    index_path: str | None,
    *,
    steps: int,
    traj_dim: int,
    hard_negative_sources: set[str],
    max_negatives_per_group: int,
) -> Dict[str, torch.Tensor]:
    grouped: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for row in _load_jsonl(index_path):
        if _row_source(row) not in hard_negative_sources:
            continue
        gid = _row_group_id(row)
        if gid is None:
            continue
        if max_negatives_per_group > 0 and len(grouped[gid]) >= max_negatives_per_group:
            continue
        traj = _traj_tensor(row, steps, traj_dim)
        if traj is not None:
            grouped[gid].append(traj)
    return {gid: torch.stack(items, dim=0) for gid, items in grouped.items() if items}


def _group_key(cache: Dict[str, Any], idx: int) -> str:
    group_ids = cache.get("group_id") or []
    sample_ids = cache.get("sample_id") or []
    if idx < len(group_ids) and group_ids[idx]:
        return str(group_ids[idx])
    if idx < len(sample_ids) and sample_ids[idx]:
        sample_id = str(sample_ids[idx])
        return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id
    return str(idx)


def _build_groups(
    cache: Dict[str, Any],
    *,
    steps: int,
    traj_dim: int,
    max_targets_per_group: int,
    negative_targets: Dict[str, torch.Tensor] | None = None,
) -> List[Dict[str, Any]]:
    x = cache["x"].float()
    y = cache["y"].float()[:, :steps, :traj_dim]
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx in range(x.shape[0]):
        grouped[_group_key(cache, idx)].append(idx)

    out: List[Dict[str, Any]] = []
    for gid, indices in grouped.items():
        if max_targets_per_group > 0 and len(indices) > max_targets_per_group:
            indices = indices[:max_targets_per_group]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        out.append(
            {
                "group_id": gid,
                "x": x.index_select(0, idx_tensor).mean(dim=0),
                "y": y.index_select(0, idx_tensor),
                "neg_y": (
                    torch.empty((0, steps, traj_dim), dtype=torch.float32)
                    if negative_targets is None
                    else negative_targets.get(
                        gid,
                        torch.empty((0, steps, traj_dim), dtype=torch.float32),
                    )
                ),
                "num_targets": len(indices),
            }
        )
    return out


def _path_features_2d(paths: torch.Tensor) -> torch.Tensor:
    flat = paths.reshape(-1, paths.shape[-2], paths.shape[-1])
    feats = _trajectory_features(flat)
    return feats.reshape(*paths.shape[:-2], feats.shape[-1])


def _ade_matrix(paths: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.norm(paths[:, None, :, :2] - targets[None, :, :, :2], p=2, dim=-1).mean(dim=-1)


def _fde_matrix(paths: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.norm(paths[:, None, -1, :2] - targets[None, :, -1, :2], p=2, dim=-1)


def _diversity_loss(paths: torch.Tensor, min_separation: float) -> torch.Tensor:
    if paths.shape[0] < 2:
        return paths.sum() * 0.0
    xy = paths[..., :2].flatten(1)
    dist = torch.cdist(xy[None], xy[None], p=2)[0]
    eye = torch.eye(paths.shape[0], device=paths.device, dtype=torch.bool)
    pair = dist.masked_select(~eye)
    return F.relu(float(min_separation) - pair).mean()


def _set_loss(
    paths: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reverse_loss_weight: float,
    shape_loss_weight: float,
    smoothness_loss_weight: float,
    cls_loss_weight: float,
    diversity_weight: float,
    diversity_min_separation: float,
    negative_targets: torch.Tensor | None,
    exclusion_loss_weight: float,
    exclusion_margin: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    ade = _ade_matrix(paths, targets)
    fde = _fde_matrix(paths, targets)
    target_min_ade, best_mode = ade.min(dim=0)
    coverage = target_min_ade.mean()

    best_paths = paths.index_select(0, best_mode)
    loss = F.smooth_l1_loss(best_paths, targets)
    if shape_loss_weight:
        loss = loss + float(shape_loss_weight) * F.smooth_l1_loss(
            _path_features_2d(best_paths),
            _trajectory_features(targets),
        )
    if smoothness_loss_weight and paths.shape[-2] > 2:
        pred_vel = best_paths[:, 1:, :2] - best_paths[:, :-1, :2]
        tgt_vel = targets[:, 1:, :2] - targets[:, :-1, :2]
        loss = loss + float(smoothness_loss_weight) * F.smooth_l1_loss(pred_vel, tgt_vel)
    if reverse_loss_weight:
        mode_min_ade = ade.min(dim=1).values
        mode_weights = torch.softmax(logits, dim=0)
        loss = loss + float(reverse_loss_weight) * (mode_weights * mode_min_ade).sum()
    if cls_loss_weight:
        target_dist = torch.zeros_like(logits)
        target_dist.scatter_add_(0, best_mode, torch.ones_like(best_mode, dtype=logits.dtype))
        target_dist = target_dist / target_dist.sum().clamp_min(1.0)
        loss = loss + float(cls_loss_weight) * -(target_dist * F.log_softmax(logits, dim=0)).sum()
    if diversity_weight:
        loss = loss + float(diversity_weight) * _diversity_loss(
            paths,
            float(diversity_min_separation),
        )
    negative_minade = math.nan
    negative_violation = math.nan
    if (
        negative_targets is not None
        and negative_targets.numel() > 0
        and exclusion_loss_weight > 0.0
    ):
        neg_ade = _ade_matrix(paths, negative_targets)
        neg_minade = neg_ade.min(dim=0).values
        violation = F.relu(float(exclusion_margin) - neg_minade)
        loss = loss + float(exclusion_loss_weight) * violation.mean()
        negative_minade = float(neg_minade.mean().detach().cpu().item())
        negative_violation = float(violation.mean().detach().cpu().item())
    return loss, {
        "coverage_minade": float(coverage.detach().cpu().item()),
        "coverage_minfde": float(fde.min(dim=0).values.mean().detach().cpu().item()),
        "used_modes": float(best_mode.unique().numel()),
        "negative_minade": negative_minade,
        "negative_violation": negative_violation,
    }


@torch.no_grad()
def _metrics(
    model: RecoveredPathSetProbe,
    groups: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    group_minades: List[float] = []
    target_minades: List[float] = []
    target_minfdes: List[float] = []
    used_modes: List[float] = []
    negative_minades: List[float] = []
    for item in groups:
        x = item["x"].to(device)[None]
        targets = item["y"].to(device)
        paths, _ = model(x)
        paths = paths[0]
        ade = _ade_matrix(paths, targets)
        fde = _fde_matrix(paths, targets)
        best = ade.min(dim=0).values
        target_minades.extend(float(v) for v in best.detach().cpu().tolist())
        target_minfdes.extend(float(v) for v in fde.min(dim=0).values.detach().cpu().tolist())
        group_minades.append(float(best.mean().detach().cpu().item()))
        used_modes.append(float(ade.argmin(dim=0).unique().numel()))
        neg = item.get("neg_y")
        if neg is not None and neg.numel() > 0:
            neg_ade = _ade_matrix(paths, neg.to(device))
            negative_minades.extend(
                float(v) for v in neg_ade.min(dim=0).values.detach().cpu().tolist()
            )
    xs = sorted(target_minades)
    if not xs:
        return {}
    return {
        "num_groups": len(groups),
        "num_targets": len(target_minades),
        "group_minade_mean": sum(group_minades) / len(group_minades),
        "minade_mean": sum(target_minades) / len(target_minades),
        "minfde_mean": sum(target_minfdes) / len(target_minfdes),
        "used_modes_mean": sum(used_modes) / len(used_modes),
        "negative_minade_mean": (
            sum(negative_minades) / len(negative_minades)
            if negative_minades
            else None
        ),
        "minade_p50": xs[int(0.50 * (len(xs) - 1))],
        "minade_p80": xs[int(0.80 * (len(xs) - 1))],
        "minade_p90": xs[int(0.90 * (len(xs) - 1))],
        "minade_p95": xs[int(0.95 * (len(xs) - 1))],
    }


def _batches(groups: Sequence[Dict[str, Any]], batch_size: int, shuffle: bool) -> List[List[Dict[str, Any]]]:
    items = list(groups)
    if shuffle:
        random.shuffle(items)
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-modes", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reverse-loss-weight", type=float, default=0.08)
    parser.add_argument("--shape-loss-weight", type=float, default=0.25)
    parser.add_argument("--smoothness-loss-weight", type=float, default=0.05)
    parser.add_argument("--cls-loss-weight", type=float, default=0.10)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--diversity-min-separation", type=float, default=2.0)
    parser.add_argument("--max-targets-per-group", type=int, default=0)
    parser.add_argument("--train-negative-index")
    parser.add_argument("--val-negative-index")
    parser.add_argument("--hard-negative-sources", default=DEFAULT_HARD_NEGATIVE_SOURCES)
    parser.add_argument("--max-negatives-per-group", type=int, default=8)
    parser.add_argument("--exclusion-loss-weight", type=float, default=0.0)
    parser.add_argument("--exclusion-margin", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train = _load_cache(args.train_cache)
    val = _load_cache(args.val_cache)
    steps = int(train["metadata"]["candidate_traj_steps"])
    traj_dim = int(train["metadata"]["traj_dim"])
    hard_negative_sources = {
        item.strip() for item in str(args.hard_negative_sources).split(",") if item.strip()
    }
    train_negatives = _negative_targets_by_group(
        args.train_negative_index,
        steps=steps,
        traj_dim=traj_dim,
        hard_negative_sources=hard_negative_sources,
        max_negatives_per_group=int(args.max_negatives_per_group),
    )
    val_negatives = _negative_targets_by_group(
        args.val_negative_index,
        steps=steps,
        traj_dim=traj_dim,
        hard_negative_sources=hard_negative_sources,
        max_negatives_per_group=int(args.max_negatives_per_group),
    )
    train_groups = _build_groups(
        train,
        steps=steps,
        traj_dim=traj_dim,
        max_targets_per_group=int(args.max_targets_per_group),
        negative_targets=train_negatives,
    )
    val_groups = _build_groups(
        val,
        steps=steps,
        traj_dim=traj_dim,
        max_targets_per_group=int(args.max_targets_per_group),
        negative_targets=val_negatives,
    )
    if not train_groups or not val_groups:
        raise ValueError("empty grouped train/val cache")
    input_dim = int(train["x"].shape[-1])
    model = RecoveredPathSetProbe(
        input_dim,
        steps,
        traj_dim,
        int(args.num_modes),
        int(args.hidden_dim),
        float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = math.inf
    best_state = None
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save_best() -> None:
        assert best_state is not None
        metadata = {
            "kind": "recovered_path_set_probe_grouped_from_features",
            "compatible_kind": "recovered_path_set_probe_from_features",
            "train_cache": args.train_cache,
            "val_cache": args.val_cache,
            "input_dim": input_dim,
            "steps": steps,
            "traj_dim": traj_dim,
            "num_modes": int(args.num_modes),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "best_epoch": int(best_state["epoch"]),
            "best_val_metrics": best_state["val_metrics"],
            "num_train_groups": len(train_groups),
            "num_val_groups": len(val_groups),
            "loss_config": {
                "reverse_loss_weight": float(args.reverse_loss_weight),
                "shape_loss_weight": float(args.shape_loss_weight),
                "smoothness_loss_weight": float(args.smoothness_loss_weight),
                "cls_loss_weight": float(args.cls_loss_weight),
                "diversity_weight": float(args.diversity_weight),
                "diversity_min_separation": float(args.diversity_min_separation),
                "max_targets_per_group": int(args.max_targets_per_group),
                "patience": int(args.patience),
                "train_negative_index": args.train_negative_index,
                "val_negative_index": args.val_negative_index,
                "hard_negative_sources": sorted(hard_negative_sources),
                "max_negatives_per_group": int(args.max_negatives_per_group),
                "exclusion_loss_weight": float(args.exclusion_loss_weight),
                "exclusion_margin": float(args.exclusion_margin),
            },
            "train_metadata": train["metadata"],
            "val_metadata": val["metadata"],
            "history": history,
        }
        torch.save({**best_state, "metadata": metadata}, out / "recovered_path_set_probe.pt")
        (out / "summary.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        covs: List[float] = []
        modes: List[float] = []
        for batch in _batches(train_groups, int(args.batch_size), True):
            batch_losses = []
            for item in batch:
                paths, logits = model(item["x"].to(device)[None])
                loss, aux = _set_loss(
                    paths[0],
                    logits[0],
                    item["y"].to(device),
                    reverse_loss_weight=float(args.reverse_loss_weight),
                    shape_loss_weight=float(args.shape_loss_weight),
                    smoothness_loss_weight=float(args.smoothness_loss_weight),
                    cls_loss_weight=float(args.cls_loss_weight),
                    diversity_weight=float(args.diversity_weight),
                    diversity_min_separation=float(args.diversity_min_separation),
                    negative_targets=item["neg_y"].to(device),
                    exclusion_loss_weight=float(args.exclusion_loss_weight),
                    exclusion_margin=float(args.exclusion_margin),
                )
                batch_losses.append(loss)
                covs.append(aux["coverage_minade"])
                modes.append(aux["used_modes"])
            loss = torch.stack(batch_losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = _metrics(model, val_groups, device)
        record = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(len(losses), 1),
            "train_coverage_minade": sum(covs) / max(len(covs), 1),
            "train_used_modes": sum(modes) / max(len(modes), 1),
            **val_metrics,
        }
        history.append(record)
        if epoch == 1 or epoch % 5 == 0 or epoch == int(args.epochs):
            print(json.dumps(record, ensure_ascii=False), flush=True)
        metric = float(val_metrics.get("group_minade_mean", math.inf))
        if metric < best:
            best = metric
            best_epoch = epoch
            best_state = {
                "probe": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            save_best()
        if int(args.patience) > 0 and epoch - best_epoch >= int(args.patience):
            print(
                json.dumps(
                    {
                        "early_stop": True,
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_metric": best,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    assert best_state is not None
    save_best()
    print("saved", out / "recovered_path_set_probe.pt")


if __name__ == "__main__":
    main()

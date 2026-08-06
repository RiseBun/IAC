#!/usr/bin/env python3
"""Fine-tune ordered motion evidence with a candidate-blind speed ranking loss.

The visual branch still receives only the GT video's V-JEPA tokens.  For each
scene, the GT trajectory and its same-scene ``perturb_speed`` trajectory are
compared in the loss; the candidate trajectory is never an input to the visual
network.  This isolates the current speed-ambiguity failure mode without
changing the inference-time scorer.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import _pathfix  # noqa: F401

from iac_extensions.ordered_motion_alignment import (  # noqa: E402
    gaussian_motion_loss,
    load_bundle,
    load_feature_cache,
    normalize_targets,
    save_bundle,
    trajectory_targets_from_rows,
)
from ordered_motion_common import (  # noqa: E402
    group_id,
    load_rows,
    sha256,
    source,
    write_json,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_speed_pairs(
    rows_path: Path,
    cache_path: Path,
    *,
    feature_key: str,
    segment_count: int,
) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Return GT visual tokens plus GT/speed targets, grouped by scene."""

    rows = load_rows(rows_path)
    feature_by_sample, cache_metadata = load_feature_cache(
        cache_path,
        key=feature_key,
    )
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for raw in rows:
        grouped.setdefault(group_id(raw), []).append(dict(raw))

    selected: List[Dict[str, Any]] = []
    visual: List[torch.Tensor] = []
    gt_targets: List[torch.Tensor] = []
    speed_targets: List[torch.Tensor] = []
    skipped: Dict[str, int] = {}
    for scene, candidates in sorted(grouped.items()):
        gt = sorted(
            (row for row in candidates if source(row) == "gt_pos"),
            key=lambda row: str(row.get("sample_id", "")),
        )
        speed = sorted(
            (row for row in candidates if source(row) == "perturb_speed"),
            key=lambda row: str(row.get("sample_id", "")),
        )
        if len(gt) != 1 or len(speed) != 1:
            skipped["missing_or_duplicate_pair"] = skipped.get(
                "missing_or_duplicate_pair", 0
            ) + 1
            continue
        feature = feature_by_sample.get(str(gt[0].get("sample_id", "")))
        if feature is None:
            skipped["missing_gt_visual"] = skipped.get("missing_gt_visual", 0) + 1
            continue
        selected.append(
            {
                "group_id": scene,
                "gt_sample_id": str(gt[0].get("sample_id", "")),
                "speed_sample_id": str(speed[0].get("sample_id", "")),
            }
        )
        visual.append(feature.float())
        gt_targets.append(
            trajectory_targets_from_rows(gt, segment_count=segment_count)[0]
        )
        speed_targets.append(
            trajectory_targets_from_rows(speed, segment_count=segment_count)[0]
        )

    if not visual:
        raise ValueError(f"no GT/speed pairs matched in {rows_path}")
    metadata = dict(cache_metadata)
    metadata["pair_count"] = len(visual)
    metadata["skipped_groups"] = skipped
    return (
        selected,
        torch.stack(visual),
        torch.stack(gt_targets),
        torch.stack(speed_targets),
        metadata,
    )


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    margin: float,
    lambda_speed_rank: float,
) -> Dict[str, float]:
    model.eval()
    totals = {"objective": 0.0, "gaussian_loss": 0.0, "rank_loss": 0.0}
    wins = 0
    count = 0
    for visual, gt_target, speed_target in loader:
        visual = visual.to(device, non_blocking=True)
        gt_target = gt_target.to(device, non_blocking=True)
        speed_target = speed_target.to(device, non_blocking=True)
        output = model(visual)
        # The loader contains targets normalized with the init model's frozen
        # statistics.  Do not normalize them a second time here.
        gt_normalized = gt_target
        speed_normalized = speed_target
        gaussian = gaussian_motion_loss(output, gt_normalized)
        gt_energy = model.evidence(output, gt_normalized)["ordered_motion_energy"]
        speed_energy = model.evidence(output, speed_normalized)["ordered_motion_energy"]
        rank = F.relu(float(margin) + gt_energy - speed_energy)
        size = int(visual.shape[0])
        totals["gaussian_loss"] += float(gaussian.item()) * size
        totals["rank_loss"] += float(rank.mean().item()) * size
        totals["objective"] += float(
            (gaussian + float(lambda_speed_rank) * rank.mean()).item()
        ) * size
        wins += int((gt_energy < speed_energy).sum().item())
        count += size
    denominator = max(count, 1)
    return {
        key: value / denominator for key, value in totals.items()
    } | {"speed_pairwise_gt_win": wins / denominator, "pairs": float(count)}


def train(args: argparse.Namespace) -> Dict[str, Any]:
    _seed_everything(int(args.seed))
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    init_bundle = load_bundle(Path(args.init_model), device=device)
    model = init_bundle["model"]
    target_mean = init_bundle["target_mean"]
    target_std = init_bundle["target_std"]
    segment_count = int(model.config.segment_count)

    train_rows, train_visual, train_gt, train_speed, train_cache_metadata = (
        _load_speed_pairs(
            Path(args.train_rows),
            Path(args.train_cache),
            feature_key=args.feature_key,
            segment_count=segment_count,
        )
    )
    val_rows, val_visual, val_gt, val_speed, val_cache_metadata = _load_speed_pairs(
        Path(args.val_rows),
        Path(args.val_cache),
        feature_key=args.feature_key,
        segment_count=segment_count,
    )
    train_gt_n = normalize_targets(train_gt, target_mean.cpu(), target_std.cpu())
    train_speed_n = normalize_targets(
        train_speed, target_mean.cpu(), target_std.cpu()
    )
    val_gt_n = normalize_targets(val_gt, target_mean.cpu(), target_std.cpu())
    val_speed_n = normalize_targets(val_speed, target_mean.cpu(), target_std.cpu())
    train_loader = DataLoader(
        TensorDataset(train_visual, train_gt_n, train_speed_n),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    val_loader = DataLoader(
        TensorDataset(val_visual, val_gt_n, val_speed_n),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    history: List[Dict[str, Any]] = []
    best_objective = float("inf")
    best_pairwise = float("-inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    stale = 0
    start_time = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        seen = 0
        for visual, gt_target, speed_target in train_loader:
            visual = visual.to(device, non_blocking=True)
            gt_target = gt_target.to(device, non_blocking=True)
            speed_target = speed_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(visual)
            gaussian = gaussian_motion_loss(output, gt_target)
            gt_energy = model.evidence(output, gt_target)["ordered_motion_energy"]
            speed_energy = model.evidence(output, speed_target)["ordered_motion_energy"]
            rank = F.relu(float(args.margin) + gt_energy - speed_energy).mean()
            loss = gaussian + float(args.lambda_speed_rank) * rank
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            size = int(visual.shape[0])
            running += float(loss.item()) * size
            seen += size

        train_objective = running / max(seen, 1)
        metrics = _evaluate(
            model,
            val_loader,
            device=device,
            margin=float(args.margin),
            lambda_speed_rank=float(args.lambda_speed_rank),
        )
        history.append({"epoch": epoch, "train_objective": train_objective, **metrics})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_objective": train_objective,
                    "val_objective": metrics["objective"],
                    "val_gaussian_loss": metrics["gaussian_loss"],
                    "val_rank_loss": metrics["rank_loss"],
                    "val_speed_pairwise_gt_win": metrics["speed_pairwise_gt_win"],
                    "best_val_objective": min(best_objective, metrics["objective"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.selection_metric == "speed_pairwise":
            improved = (
                metrics["speed_pairwise_gt_win"]
                > best_pairwise + float(args.min_delta)
                or (
                    abs(metrics["speed_pairwise_gt_win"] - best_pairwise)
                    <= float(args.min_delta)
                    and metrics["objective"] < best_objective - float(args.min_delta)
                )
            )
        else:
            improved = metrics["objective"] < best_objective - float(args.min_delta)
        if improved:
            best_objective = metrics["objective"]
            best_pairwise = metrics["speed_pairwise_gt_win"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break

    if best_state is None:
        raise RuntimeError("speed-rank training did not produce a checkpoint")
    model.load_state_dict(best_state)
    output_model = Path(args.output_model)
    metadata = dict(init_bundle.get("metadata", {}))
    metadata.update(
        {
            "speed_rank_ablation": {
                "candidate_blind_visual_estimator": True,
                "candidate_trajectory_used_as_visual_model_input": False,
                "source_label_used_as_visual_model_input": False,
                "ranking_pair": "gt_pos_vs_perturb_speed",
                "margin": float(args.margin),
                "lambda_speed_rank": float(args.lambda_speed_rank),
                "init_model": str(args.init_model),
            },
            "train_speed_pairs": len(train_rows),
            "val_speed_pairs": len(val_rows),
            "train_cache_metadata": train_cache_metadata,
            "val_cache_metadata": val_cache_metadata,
            "optimization": {
                "seed": int(args.seed),
                "epochs_requested": int(args.epochs),
                "best_epoch": best_epoch,
                "best_val_objective": best_objective,
                "best_val_speed_pairwise_gt_win": best_pairwise,
                "selection_metric": args.selection_metric,
                "history": history,
            },
        }
    )
    save_bundle(
        output_model,
        model=model,
        target_mean=target_mean,
        target_std=target_std,
        metadata=metadata,
    )
    summary = {
        "kind": "ordered_motion_speed_rank_ablation",
        "model": str(output_model),
        "model_sha256": sha256(output_model),
        "init_model": str(args.init_model),
        "device": str(device),
        "elapsed_seconds": time.time() - start_time,
        "best_epoch": best_epoch,
        "best_val_objective": best_objective,
        "best_val_speed_pairwise_gt_win": best_pairwise,
        "selection_metric": args.selection_metric,
        "train_speed_pairs": len(train_rows),
        "val_speed_pairs": len(val_rows),
        "candidate_blind_visual_estimator": True,
        "candidate_trajectory_used_as_visual_model_input": False,
        "source_labels_used_as_model_input": False,
        "margin": float(args.margin),
        "lambda_speed_rank": float(args.lambda_speed_rank),
        "final_val": _evaluate(
            model,
            val_loader,
            device=device,
            margin=float(args.margin),
            lambda_speed_rank=float(args.lambda_speed_rank),
        ),
    }
    if args.output_summary:
        write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-model", required=True)
    parser.add_argument("--train-rows", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-rows", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--feature-key", default="x_time_tokens")
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--lambda-speed-rank", type=float, default=0.50)
    parser.add_argument(
        "--selection-metric",
        choices=("objective", "speed_pairwise"),
        default="speed_pairwise",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

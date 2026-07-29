#!/usr/bin/env python3
"""Train the candidate-blind ordered video motion estimator.

Only GT-positive rows supervise the visual estimator by default.  Candidate
trajectories are converted to ordered segment targets for the loss, but they
are never passed into the visual network.  This prevents the visual branch from
copying its answer from a candidate at inference time.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_extensions.ordered_motion_alignment import (  # noqa: E402
    OrderedMotionAlignment,
    OrderedMotionConfig,
    gaussian_motion_loss,
    load_feature_cache,
    match_rows_to_features,
    normalize_targets,
    save_bundle,
    standardize_targets,
    trajectory_targets_from_rows,
)
from ordered_motion_common import (  # noqa: E402
    is_gt,
    is_positive,
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


def _select_positive_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selector: str,
    max_rows: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if selector == "gt_only":
        selected = [dict(row) for row in rows if is_gt(row)]
    elif selector == "label_or_gt":
        selected = [dict(row) for row in rows if is_positive(row) or is_gt(row)]
    else:
        raise ValueError(f"unknown positive selector: {selector}")

    deduplicated: Dict[str, Dict[str, Any]] = {}
    for row in selected:
        key = str(row.get("sample_id", ""))
        if key and key not in deduplicated:
            deduplicated[key] = row
    selected = list(deduplicated.values())
    if max_rows > 0 and len(selected) > max_rows:
        rng = random.Random(seed)
        selected = rng.sample(selected, max_rows)
    return selected


def _load_positive_split(
    rows_path: Path,
    cache_path: Path,
    *,
    feature_key: str,
    selector: str,
    max_rows: int,
    seed: int,
    segment_count: int,
) -> tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    raw_rows = load_rows(rows_path)
    positives = _select_positive_rows(
        raw_rows,
        selector=selector,
        max_rows=max_rows,
        seed=seed,
    )
    feature_by_sample, cache_metadata = load_feature_cache(
        cache_path,
        key=feature_key,
    )
    rows, visual = match_rows_to_features(positives, feature_by_sample)
    targets = trajectory_targets_from_rows(
        rows,
        segment_count=segment_count,
    )
    return rows, visual, targets, cache_metadata


@torch.no_grad()
def _validation_loss(
    model: OrderedMotionAlignment,
    loader: DataLoader,
    *,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for visual, target in loader:
        visual = visual.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        loss = gaussian_motion_loss(model(visual), target)
        size = int(visual.shape[0])
        total += float(loss.item()) * size
        count += size
    return total / max(count, 1)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    _seed_everything(int(args.seed))
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    train_rows, train_visual, train_target, train_cache_metadata = (
        _load_positive_split(
            Path(args.train_rows),
            Path(args.train_cache),
            feature_key=args.feature_key,
            selector=args.positive_selector,
            max_rows=int(args.max_train_rows),
            seed=int(args.seed),
            segment_count=int(args.segment_count),
        )
    )
    val_rows, val_visual, val_target, val_cache_metadata = _load_positive_split(
        Path(args.val_rows),
        Path(args.val_cache),
        feature_key=args.feature_key,
        selector=args.positive_selector,
        max_rows=int(args.max_val_rows),
        seed=int(args.seed) + 1,
        segment_count=int(args.segment_count),
    )
    if train_visual.shape[-1] != val_visual.shape[-1]:
        raise ValueError("train and validation visual feature dimensions differ")

    target_mean, target_std = standardize_targets(train_target)
    train_target_n = normalize_targets(train_target, target_mean, target_std)
    val_target_n = normalize_targets(val_target, target_mean, target_std)
    train_loader = DataLoader(
        TensorDataset(train_visual, train_target_n),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    val_loader = DataLoader(
        TensorDataset(val_visual, val_target_n),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    config = OrderedMotionConfig(
        visual_dim=int(train_visual.shape[-1]),
        hidden_dim=int(args.hidden_dim),
        segment_count=int(args.segment_count),
        bandwidth=float(args.bandwidth),
        dropout=float(args.dropout),
    )
    model = OrderedMotionAlignment(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    stale = 0
    start_time = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        seen = 0
        for visual, target in train_loader:
            visual = visual.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = gaussian_motion_loss(model(visual), target)
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(args.grad_clip),
                )
            optimizer.step()
            size = int(visual.shape[0])
            running += float(loss.item()) * size
            seen += size

        train_loss = running / max(seen, 1)
        val_loss = _validation_loss(model, val_loader, device=device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": min(best_loss, val_loss),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if val_loss < best_loss - float(args.min_delta):
            best_loss = val_loss
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
        raise RuntimeError("training did not produce a finite validation state")
    model.load_state_dict(best_state)
    output_model = Path(args.output_model)
    metadata = {
        "protocol": {
            "candidate_blind_visual_estimator": True,
            "candidate_trajectory_used_as_visual_model_input": False,
            "source_label_used_as_visual_model_input": False,
            "positive_selector": args.positive_selector,
            "feature_key": args.feature_key,
        },
        "train": {
            "rows_path": str(args.train_rows),
            "cache_path": str(args.train_cache),
            "matched_positive_rows": len(train_rows),
            "source_counts": _source_counts(train_rows),
            "cache_metadata": train_cache_metadata,
        },
        "val": {
            "rows_path": str(args.val_rows),
            "cache_path": str(args.val_cache),
            "matched_positive_rows": len(val_rows),
            "source_counts": _source_counts(val_rows),
            "cache_metadata": val_cache_metadata,
        },
        "optimization": {
            "seed": int(args.seed),
            "epochs_requested": int(args.epochs),
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "history": history,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
        },
    }
    save_bundle(
        output_model,
        model=model,
        target_mean=target_mean,
        target_std=target_std,
        metadata=metadata,
    )
    summary = {
        "kind": "ordered_motion_alignment_training",
        "model": str(output_model),
        "model_sha256": sha256(output_model),
        "device": str(device),
        "elapsed_seconds": time.time() - start_time,
        "config": config.to_dict(),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "train_positive_rows": len(train_rows),
        "val_positive_rows": len(val_rows),
        "candidate_blind_visual_estimator": True,
        "source_labels_used_as_model_input": False,
    }
    if args.output_summary:
        write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def _source_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = source(row)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-rows", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--feature-key", default="x_time_tokens")
    parser.add_argument(
        "--positive-selector",
        choices=("gt_only", "label_or_gt"),
        default="gt_only",
    )
    parser.add_argument("--segment-count", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bandwidth", type=float, default=0.22)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

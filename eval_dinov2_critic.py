#!/usr/bin/env python3
"""Evaluate a DINOv2 consistency critic (minimal v5 variant).

A thin shim around the standard eval_critic.py flow that imports
`DINOv2ConsistencyCritic` from train_dinov2_v5_minimal instead of the
CNN-based `ConsistencyCriticModel` from train.py.

All metrics, ranking evaluation, per-source-type grouping, and graded
perturbation curve output are identical to eval_critic.py.

Usage::

  python eval_dinov2_critic.py \
    --checkpoint work_dirs/iac_dinov2_v5_minimal/checkpoints/best.pth \
    --split val --max-samples 4096 --eval-ranking
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# Ensure both trainers can be imported side by side.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_critic  # uses train.py's CNN Critic for backward compatibility
from train import ConsistencyDataset, load_config  # noqa: E402
from train_dinov2_v5_minimal import DINOv2ConsistencyCritic  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DINOv2 consistency critic")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--eval-ranking", action="store_true")
    p.add_argument(
        "--baseline-mode",
        choices=["full", "no_image", "ego_only", "no_traj", "traj_only"],
        default=None,
    )
    p.add_argument("--output-prefix", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = (
        load_config(args.config)
        if args.config
        else checkpoint.get("config", {})
    )
    if not cfg:
        raise ValueError("No config in checkpoint; pass --config.")
    if args.baseline_mode is not None:
        cfg["baseline_mode"] = args.baseline_mode
    if cfg.get("model_type") != "consistency":
        raise ValueError("model_type must be 'consistency'.")
    print(f"Epoch={checkpoint.get('epoch','?')} best_val_loss={checkpoint.get('best_val_loss','?')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv2ConsistencyCritic(cfg).to(device)
    state = checkpoint["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARNING] missing keys when loading: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARNING] unexpected keys when loading: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    model.eval()

    print(f"Model loaded on {device}")

    index_key = "val_index" if args.split == "val" else "train_index"
    index_path = cfg[index_key]
    print(f"Dataset: {args.split} ({index_path})")

    dataset = ConsistencyDataset(index_path=index_path, cfg=cfg, training=False)
    print(f"Samples: {len(dataset)}")

    print("\nRunning evaluation...")
    metrics = eval_critic.evaluate_consistency(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 60)
    print("DINOv2 IAC Consistency Critic — Evaluation")
    print("=" * 60)
    print(f"  Total samples: {metrics['total_samples']}")
    eval_critic._print_head_metrics("Consistency Head", metrics["consistency"])
    eval_critic._print_head_metrics("Validity Head", metrics["validity"])

    if metrics.get("per_source_type"):
        print("\n  [Per Source Type]")
        for st, st_data in metrics["per_source_type"].items():
            print(f"    --- {st} (n={st_data['count']}) ---")
            c = st_data["consistency"]
            v = st_data["validity"]
            print(f"      consistency: {eval_critic._format_source_line(c)}")
            print(f"      validity:    {eval_critic._format_source_line(v)}")
    if metrics.get("negative_recall_by_type"):
        print("\n  [Negative Recall / TNR by Type]")
        for st, value in metrics["negative_recall_by_type"].items():
            print(f"    {st}: {value:.4f}" if value is not None else f"    {st}: N/A")
    if metrics.get("graded_perturbation_curve"):
        print("\n  [Graded Perturbation Curve]")
        for key, data in metrics["graded_perturbation_curve"].items():
            print(
                f"    {key}: n={data['count']} "
                f"mean_prob={data['mean_consistency_prob']:.4f} "
                f"mean_mag={data['mean_perturb_magnitude']}"
            )
    print("=" * 60)

    if args.eval_ranking:
        print("\n" + "=" * 60)
        print("Ranking evaluation...")
        print("=" * 60)
        ranking_metrics = eval_critic.compute_ranking_metrics(
            model=model, dataset=dataset, device=device, batch_size=args.batch_size,
        )
        if ranking_metrics:
            print("\n[Ranking Metrics]")
            print(f"  Scenes:  {ranking_metrics['num_scenes']}")
            print(f"  NDCG@3:  {ranking_metrics['ndcg@3']:.4f}")
            print(f"  NDCG@5:  {ranking_metrics['ndcg@5']:.4f}")
            print(f"  MRR:     {ranking_metrics['mrr']:.4f}")
            print(f"  Top-1:   {ranking_metrics['top1_hit_rate']:.4f}")
            print("=" * 60)
            metrics["ranking"] = ranking_metrics

    prefix = args.output_prefix or f"dinov2_v5_{args.split}"
    result_path = ckpt_path.parent.parent / f"{prefix}_results.json"
    summary_path = ckpt_path.parent.parent / f"{prefix}_summary.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nResults: {result_path}")
    summary = {
        "total_samples": metrics["total_samples"],
        "baseline_mode": cfg.get("baseline_mode", "full"),
        "backbone": "dinov2_vits14_layer11_minimal",
        "consistency": {
            k: metrics["consistency"].get(k)
            for k in ("accuracy", "auc", "pr_auc", "ece", "f1_score", "tnr", "fpr")
        },
        "validity": {
            k: metrics["validity"].get(k)
            for k in ("accuracy", "auc", "pr_auc", "ece", "f1_score", "tnr", "fpr")
        },
        "negative_recall_by_type": metrics.get("negative_recall_by_type", {}),
        "graded_perturbation_curve": metrics.get("graded_perturbation_curve", {}),
        "ranking": metrics.get("ranking", {}),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

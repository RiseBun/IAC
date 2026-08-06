#!/usr/bin/env python3
"""Train leave-one-hard-family-out clean gates and audit family transfer."""

from __future__ import annotations

import argparse
import gc
import json
import math
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

import _pathfix  # noqa: F401

from audit_clean_gate_counterfactual import (
    DEFAULT_ACCEPTABLE_SOURCES,
    DEFAULT_HARD_SOURCES,
    _empty_metrics,
    _evaluate,
    _finalize_metrics,
    _merge_metrics,
    _parse_dataset,
    _parse_sources,
    _score_control,
    _source,
)
from score_visual_mismatch_gate import _load_gate
from train_visual_mismatch_gate_scorer import _load_dataset, _train


DEFAULT_LOFO_SOURCES = (
    "image_swap,time_shift_future,traj_swap,high_pdm_image_mismatch"
)


def _empty_family_stats() -> Dict[str, float]:
    return {
        "count": 0,
        "reject_count": 0,
        "gt_better_count": 0,
        "heldout_logit_sum": 0.0,
        "gt_logit_sum": 0.0,
        "gt_margin_sum": 0.0,
    }


def _family_stats(
    rows: Sequence[Dict[str, Any]],
    logits: torch.Tensor,
    family: str,
    *,
    group_key: str,
    source_key: str,
) -> Dict[str, float]:
    by_group: Dict[str, Dict[str, int]] = defaultdict(dict)
    for idx, row in enumerate(rows):
        group = str(row.get(group_key) or row.get("anchor_id") or row.get("sample_id"))
        by_group[group][_source(row, source_key)] = idx
    result = _empty_family_stats()
    for sources in by_group.values():
        heldout_idx = sources.get(family)
        gt_idx = sources.get("gt_pos")
        if heldout_idx is None or gt_idx is None:
            continue
        heldout = float(logits[heldout_idx])
        gt = float(logits[gt_idx])
        result["count"] += 1
        result["reject_count"] += int(heldout < 0.0)
        result["gt_better_count"] += int(gt > heldout)
        result["heldout_logit_sum"] += heldout
        result["gt_logit_sum"] += gt
        result["gt_margin_sum"] += gt - heldout
    return result


def _merge_family_stats(target: Dict[str, float], source: Dict[str, float]) -> None:
    for key, value in source.items():
        target[key] += value


def _finalize_family_stats(raw: Dict[str, float]) -> Dict[str, Any]:
    count = int(raw["count"])
    if not count:
        return {
            "count": 0,
            "reject_rate_at_zero": None,
            "gt_better_rate": None,
            "heldout_logit_mean": None,
            "gt_logit_mean": None,
            "gt_margin_mean": None,
        }
    return {
        "count": count,
        "reject_rate_at_zero": raw["reject_count"] / count,
        "gt_better_rate": raw["gt_better_count"] / count,
        "heldout_logit_mean": raw["heldout_logit_sum"] / count,
        "gt_logit_mean": raw["gt_logit_sum"] / count,
        "gt_margin_mean": raw["gt_margin_sum"] / count,
    }


def _save_bundle(bundle: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": bundle["model"].state_dict(),
            "visual_mean": bundle["visual_mean"],
            "visual_std": bundle["visual_std"],
            "scalar_mean": bundle["scalar_mean"],
            "scalar_std": bundle["scalar_std"],
            "traj_mean": bundle["traj_mean"],
            "traj_std": bundle["traj_std"],
            "metadata": bundle["metadata"],
            "history": bundle["history"],
        },
        path,
    )


def _train_args(args: argparse.Namespace, family: str, hard_sources: set[str]) -> Namespace:
    return Namespace(
        train_rows=str(args.train_rows),
        train_visual_cache=str(args.train_visual_cache),
        visual_cache_key=args.train_visual_cache_key,
        scalar_feature_mode="zero",
        eval=[],
        output_dir=str(Path(args.output_dir) / family),
        group_key=args.group_key,
        wam_key=args.source_key,
        supported_sources="perturb_speed,perturb_lateral,perturb_heading",
        unknown_sources="perturb_speed,perturb_lateral,perturb_heading",
        protect_sources=None,
        hard_sources=",".join(sorted(hard_sources - {family})),
        min_supported_quality=0.90,
        unknown_weight=0.20,
        unknown_margin=0.75,
        loss_kind="margin",
        supported_margin=1.0,
        hard_margin=1.0,
        logit_l2_weight=0.001,
        standardize_clip=5.0,
        pairwise_weight=0.75,
        pairwise_margin=1.0,
        interaction_kind="traj_cross_attention",
        visual_hidden_dim=64,
        hidden_dim=128,
        dropout=0.25,
        lr=0.002,
        weight_decay=0.001,
        steps=args.steps,
        log_every=args.log_every,
    )


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance), "min": min(values), "max": max(values)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", required=True)
    parser.add_argument("--train-visual-cache", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=ROWS,CACHE")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--heldout-sources", default=DEFAULT_LOFO_SOURCES)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--train-visual-cache-key", default="x_tokens")
    parser.add_argument("--eval-visual-cache-key", default="x_tokens")
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--source-key", default="source_type")
    parser.add_argument("--acceptable-sources", default=DEFAULT_ACCEPTABLE_SOURCES)
    parser.add_argument("--hard-sources", default=DEFAULT_HARD_SOURCES)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--log-every", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    families = [item.strip() for item in args.heldout_sources.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    hard_sources = _parse_sources(args.hard_sources)
    acceptable_sources = _parse_sources(args.acceptable_sources)

    train_rows, train_visual, train_scalar, train_traj = _load_dataset(
        Path(args.train_rows),
        Path(args.train_visual_cache),
        feature_key=args.train_visual_cache_key,
        scalar_feature_mode="zero",
    )
    baseline = _load_gate(Path(args.baseline_model))
    bundles: Dict[str, Dict[str, Any]] = {"baseline": baseline}
    descriptors: Dict[str, Dict[str, Any]] = {
        "baseline": {"kind": "baseline", "heldout_source": None, "seed": None}
    }

    for family in families:
        kept = [idx for idx, row in enumerate(train_rows) if _source(row, args.source_key) != family]
        if len(kept) == len(train_rows):
            raise ValueError(f"held-out family {family!r} is absent from training rows")
        kept_index = torch.tensor(kept, dtype=torch.long)
        family_rows = [train_rows[idx] for idx in kept]
        for seed in seeds:
            torch.manual_seed(seed)
            train_args = _train_args(args, family, hard_sources)
            bundle = _train(
                family_rows,
                train_visual.index_select(0, kept_index),
                train_scalar.index_select(0, kept_index),
                train_traj.index_select(0, kept_index),
                train_args,
            )
            bundle["metadata"]["lofo"] = {
                "heldout_source": family,
                "seed": seed,
                "original_rows": len(train_rows),
                "kept_rows": len(kept),
            }
            key = f"{family}__seed_{seed}"
            model_path = output_dir / "models" / f"{key}.pt"
            _save_bundle(bundle, model_path)
            bundles[key] = bundle
            descriptors[key] = {
                "kind": "lofo",
                "heldout_source": family,
                "seed": seed,
                "model": str(model_path),
                "kept_rows": len(kept),
            }
            print(json.dumps({"trained": key, "kept_rows": len(kept)}), flush=True)

    raw_results: Dict[str, Dict[str, Any]] = {}
    for key, descriptor in descriptors.items():
        raw_results[key] = {
            "descriptor": descriptor,
            "regular": _empty_metrics(),
            "regular_family": {
                family: _empty_family_stats()
                for family in (families if key == "baseline" else [descriptor["heldout_source"]])
            },
        }

    for raw_dataset in args.dataset:
        name, rows_path, cache_path = _parse_dataset(raw_dataset)
        rows, visual, scalar, traj = _load_dataset(
            rows_path,
            cache_path,
            feature_key=args.eval_visual_cache_key,
            scalar_feature_mode="zero",
        )
        identity = torch.arange(len(rows), dtype=torch.long)
        for key, bundle in bundles.items():
            logits = _score_control(
                bundle,
                visual,
                scalar,
                traj,
                control="normal",
                permutation=identity,
                batch_size=args.batch_size,
                device=device,
            )
            metrics = _evaluate(
                rows,
                logits,
                group_key=args.group_key,
                source_key=args.source_key,
                acceptable_sources=acceptable_sources,
                hard_sources=hard_sources,
            )
            _merge_metrics(raw_results[key]["regular"], metrics)
            for family, target in raw_results[key]["regular_family"].items():
                _merge_family_stats(
                    target,
                    _family_stats(
                        rows,
                        logits,
                        family,
                        group_key=args.group_key,
                        source_key=args.source_key,
                    ),
                )
            print(json.dumps({"scored": key, "dataset": name}), flush=True)
        del rows, visual, scalar, traj, identity, logits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_identity = torch.arange(len(train_rows), dtype=torch.long)
    for key, bundle in bundles.items():
        logits = _score_control(
            bundle,
            train_visual,
            train_scalar,
            train_traj,
            control="normal",
            permutation=train_identity,
            batch_size=args.batch_size,
            device=device,
        )
        raw_results[key]["train_family"] = {}
        targets = families if key == "baseline" else [descriptors[key]["heldout_source"]]
        for family in targets:
            raw_results[key]["train_family"][family] = _family_stats(
                train_rows,
                logits,
                family,
                group_key=args.group_key,
                source_key=args.source_key,
            )

    final_results: Dict[str, Any] = {}
    for key, raw in raw_results.items():
        final_results[key] = {
            "descriptor": raw["descriptor"],
            "regular": _finalize_metrics(raw["regular"]),
            "regular_family": {
                family: _finalize_family_stats(values)
                for family, values in raw["regular_family"].items()
            },
            "train_family": {
                family: _finalize_family_stats(values)
                for family, values in raw["train_family"].items()
            },
        }

    family_summary: Dict[str, Any] = {}
    for family in families:
        baseline_regular = final_results["baseline"]["regular_family"][family]
        baseline_train = final_results["baseline"]["train_family"][family]
        per_seed = []
        for seed in seeds:
            key = f"{family}__seed_{seed}"
            regular = final_results[key]["regular_family"][family]
            train = final_results[key]["train_family"][family]
            primary = regular if regular["count"] else train
            per_seed.append(
                {
                    "seed": seed,
                    "evaluation_scope": "regular" if regular["count"] else "train_same_scene_family_heldout",
                    **primary,
                    "regular_acceptable_top1": final_results[key]["regular"]["acceptable_top1"],
                    "regular_hard_mismatch_top1": final_results[key]["regular"]["hard_mismatch_top1"],
                }
            )
        baseline_primary = baseline_regular if baseline_regular["count"] else baseline_train
        family_summary[family] = {
            "baseline": baseline_primary,
            "per_seed": per_seed,
            "aggregate": {
                field: _mean_std([float(item[field]) for item in per_seed])
                for field in ("reject_rate_at_zero", "gt_better_rate", "gt_margin_mean")
            },
        }

    summary = {
        "kind": "clean_gate_leave_one_hard_family_out_audit",
        "definition": (
            "Every row from the held-out hard-negative family is removed before training. "
            "GT-better rate and GT logit margin are the primary family-transfer metrics."
        ),
        "config": {
            "train_rows": str(args.train_rows),
            "train_visual_cache": str(args.train_visual_cache),
            "baseline_model": str(args.baseline_model),
            "datasets": args.dataset,
            "heldout_sources": families,
            "seeds": seeds,
            "steps": args.steps,
            "device": str(device),
        },
        "family_summary": family_summary,
        "models": final_results,
    }
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(family_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

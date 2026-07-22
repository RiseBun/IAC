#!/usr/bin/env python3
"""Evaluate candidate-blind scope-motion evidence.

This tool reads the visual motion branch directly.  It is intentionally
separate from final consistency evaluation: lower ``scope_motion_energy`` and
higher ``motion_rule_match_logit`` should identify candidates supported by the
future-image motion evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import ConsistencyDataset, load_config  # noqa: E402
from train_scope_motion_head import ScopeDinoMotionCritic  # noqa: E402


HARD_SOURCES = {
    "image_swap",
    "time_shift_future",
    "traj_swap",
    "reverse_traj",
    "high_pdm_image_mismatch",
}
NEAR_SOURCES = {"perturb_speed", "perturb_lateral", "perturb_heading"}


def _source(row: Dict[str, Any]) -> str:
    for key in ("source_type", "action_type", "wam_name", "sample_type", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _group_id(row: Dict[str, Any], fallback: str) -> str:
    value = row.get("group_id") or row.get("anchor_id")
    if value is not None:
        return str(value)
    sample_id = str(row.get("sample_id", fallback))
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def _is_positive(row: Dict[str, Any]) -> bool:
    if row.get("consistency_label") is not None:
        return float(row["consistency_label"]) > 0.5
    if row.get("label") is not None:
        return float(row["label"]) > 0.5
    return _source(row) == "gt_pos"


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def _auc(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
            total += 1
    return wins / total if total else None


def _select_indices(dataset: ConsistencyDataset, max_groups: int, max_samples: int, seed: int) -> List[int]:
    rows = list(getattr(dataset, "samples", []))
    if not rows:
        indices = list(range(len(dataset)))
        return indices[:max_samples] if max_samples > 0 else indices
    if max_groups <= 0:
        indices = list(range(len(rows)))
        return indices[:max_samples] if max_samples > 0 else indices
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[_group_id(row, str(idx))].append(idx)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    selected: List[int] = []
    for key in keys[:max_groups]:
        selected.extend(groups[key])
    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def _apply_control(future: torch.Tensor, control: str, seed: int) -> torch.Tensor:
    if control == "normal":
        return future
    if control == "reverse_future":
        return torch.flip(future, dims=[1])
    if control == "shuffle_future":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(future.shape[1], generator=generator).to(future.device)
        return future.index_select(1, perm)
    if control == "zero_future":
        return torch.zeros_like(future)
    raise ValueError(f"unknown control: {control}")


def _summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    source_summary: Dict[str, Dict[str, Any]] = {}
    for source, items in sorted(by_source.items()):
        energies = [float(row["scope_motion_energy"]) for row in items]
        logits = [float(row["motion_rule_match_logit"]) for row in items]
        source_summary[source] = {
            "count": len(items),
            "energy_mean": _mean(energies),
            "energy_p10": _quantile(energies, 0.1),
            "energy_p50": _quantile(energies, 0.5),
            "energy_p90": _quantile(energies, 0.9),
            "logit_mean": _mean(logits),
            "logit_p10": _quantile(logits, 0.1),
            "logit_p50": _quantile(logits, 0.5),
            "logit_p90": _quantile(logits, 0.9),
        }

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    pairwise: Dict[str, List[float]] = defaultdict(list)
    hard_above_gt: List[float] = []
    near_above_gt: List[float] = []
    for items in grouped.values():
        positives = [row for row in items if row["is_positive"]]
        if not positives:
            continue
        gt = positives[0]
        gt_energy = float(gt["scope_motion_energy"])
        gt_logit = float(gt["motion_rule_match_logit"])
        hard_hit = False
        near_hit = False
        for row in items:
            if row is gt:
                continue
            source = str(row["source"])
            energy = float(row["scope_motion_energy"])
            logit = float(row["motion_rule_match_logit"])
            # Lower energy and higher logit are better.
            pairwise[f"gt_better_energy_vs_{source}"].append(float(gt_energy < energy))
            pairwise[f"gt_better_logit_vs_{source}"].append(float(gt_logit > logit))
            if source in HARD_SOURCES and energy < gt_energy:
                hard_hit = True
            if source in NEAR_SOURCES and energy < gt_energy:
                near_hit = True
        hard_above_gt.append(float(hard_hit))
        near_above_gt.append(float(near_hit))

    positives = [float(row["motion_rule_match_logit"]) for row in rows if row["is_positive"]]
    hard = [
        float(row["motion_rule_match_logit"])
        for row in rows
        if (not row["is_positive"]) and row["source"] in HARD_SOURCES
    ]
    return {
        "rows": len(rows),
        "groups": len(grouped),
        "source_summary": source_summary,
        "pairwise_accuracy": {
            key: _mean(values) for key, values in sorted(pairwise.items())
        },
        "hard_mismatch_energy_above_gt_group_rate": _mean(hard_above_gt),
        "near_perturb_energy_above_gt_group_rate": _mean(near_above_gt),
        "positive_vs_hard_logit_auc": _auc(positives, hard),
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    cfg["baseline_mode"] = "full"
    dataset = ConsistencyDataset(cfg["val_index" if args.split == "val" else "train_index"], cfg, training=False)
    indices = _select_indices(dataset, args.max_groups, args.max_samples, args.seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ScopeDinoMotionCritic(cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    source_rows = list(getattr(dataset, "samples", []))
    rows_out: List[Dict[str, Any]] = []
    offset = 0
    for batch in loader:
        size = int(batch["candidate_traj"].shape[0])
        batch_indices = indices[offset : offset + size]
        offset += size
        hist = batch["history_images"].to(device, non_blocking=True)
        fut = batch["future_images"].to(device, non_blocking=True)
        fut = _apply_control(fut, args.control, args.seed)
        ego = batch["ego_state"].to(device, non_blocking=True)
        traj = batch["candidate_traj"].to(device, non_blocking=True)
        out = model(hist, fut, ego, traj)
        energy = out["scope_motion_energy"].detach().cpu().float().tolist()
        logit = out["motion_rule_match_logit"].detach().cpu().float().tolist()
        consistency = out["consistency_logit"].detach().cpu().float().tolist()
        for local_idx, dataset_idx in enumerate(batch_indices):
            raw = source_rows[dataset_idx] if source_rows else {}
            rows_out.append(
                {
                    "sample_id": str(raw.get("sample_id", batch.get("sample_id", [""])[local_idx])),
                    "group_id": _group_id(raw, str(dataset_idx)),
                    "source": _source(raw),
                    "is_positive": _is_positive(raw),
                    "scope_motion_energy": float(energy[local_idx]),
                    "motion_rule_match_logit": float(logit[local_idx]),
                    "consistency_logit": float(consistency[local_idx]),
                }
            )
    summary = _summarize_rows(rows_out)
    summary["config"] = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "split": args.split,
        "control": args.control,
        "max_groups": args.max_groups,
        "max_samples": args.max_samples,
        "rows": len(rows_out),
    }
    if args.output_rows:
        path = Path(args.output_rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--control",
        choices=["normal", "reverse_future", "shuffle_future", "zero_future"],
        default="normal",
    )
    parser.add_argument("--max-groups", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-rows", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    out = Path(args.output_summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

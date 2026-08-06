#!/usr/bin/env python3
"""Audit whether the clean trajectory gate depends on paired modalities.

The audit preserves source-family marginals while breaking only the pairing
between visual tokens and trajectory tokens. It reports both gate-only ranking
and the released v3 + gate fusion ranking without writing per-row scores.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

import _pathfix  # noqa: F401

from score_visual_mismatch_gate import _load_gate
from train_visual_mismatch_gate_scorer import _load_dataset, _normalize


DEFAULT_ACCEPTABLE_SOURCES = (
    "gt_pos,perturb_speed,perturb_lateral,perturb_heading"
)
DEFAULT_HARD_SOURCES = (
    "image_swap,time_shift_future,traj_swap,reverse_traj,"
    "high_pdm_image_mismatch"
)
DEFAULT_CONTROLS = (
    "normal,shuffle_visual,shuffle_traj,shuffle_both,"
    "mean_visual,mean_traj"
)


def _parse_sources(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _group_id(row: Dict[str, Any], group_key: str) -> str:
    value = row.get(group_key) or row.get("anchor_id") or row.get("sample_id")
    return str(value)


def _source(row: Dict[str, Any], source_key: str) -> str:
    for key in (source_key, "source_type", "action_type", "wam_name", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _parse_dataset(raw: str) -> tuple[str, Path, Path]:
    name, sep, rest = raw.partition("=")
    if not sep:
        raise ValueError(f"--dataset must be NAME=ROWS,CACHE, got {raw!r}")
    rows, sep, cache = rest.partition(",")
    if not sep:
        raise ValueError(f"--dataset must be NAME=ROWS,CACHE, got {raw!r}")
    return name, Path(rows), Path(cache)


def _group_source_permutation(
    rows: Sequence[Dict[str, Any]],
    *,
    group_key: str,
    source_key: str,
    seed: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    by_group: Dict[str, Dict[str, int]] = defaultdict(dict)
    by_source: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        gid = _group_id(row, group_key)
        source = _source(row, source_key)
        if source in by_group[gid]:
            raise ValueError(f"duplicate source {source!r} in group {gid!r}")
        by_group[gid][source] = idx
        by_source[source].append(idx)

    groups = sorted(by_group)
    if len(groups) < 2:
        raise ValueError("counterfactual audit requires at least two groups")
    order = list(groups)
    random.Random(seed).shuffle(order)
    donor_group = {
        group: order[(idx + 1) % len(order)] for idx, group in enumerate(order)
    }

    fallback_by_source: Dict[str, Dict[int, int]] = {}
    for source, indices in by_source.items():
        shuffled = list(indices)
        random.Random(f"{seed}:{source}").shuffle(shuffled)
        fallback_by_source[source] = {
            idx: shuffled[(pos + 1) % len(shuffled)]
            for pos, idx in enumerate(shuffled)
        }

    permutation = torch.empty(len(rows), dtype=torch.long)
    fallback_count = 0
    fixed_points = 0
    for idx, row in enumerate(rows):
        gid = _group_id(row, group_key)
        source = _source(row, source_key)
        donor = by_group[donor_group[gid]].get(source)
        if donor is None:
            donor = fallback_by_source[source][idx]
            fallback_count += 1
        permutation[idx] = donor
        fixed_points += int(donor == idx)

    return permutation, {
        "num_groups": len(groups),
        "num_rows": len(rows),
        "fallback_rows": fallback_count,
        "fixed_points": fixed_points,
    }


@torch.inference_mode()
def _score_control(
    bundle: Dict[str, Any],
    visual: torch.Tensor,
    scalar: torch.Tensor,
    traj: torch.Tensor,
    *,
    control: str,
    permutation: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model = bundle["model"].to(device)
    model.eval()
    clip = float(bundle["metadata"].get("args", {}).get("standardize_clip", 5.0))
    visual_mean = bundle["visual_mean"].to(device)
    visual_std = bundle["visual_std"].to(device)
    scalar_mean = bundle["scalar_mean"].to(device)
    scalar_std = bundle["scalar_std"].to(device)
    traj_mean = bundle["traj_mean"].to(device)
    traj_std = bundle["traj_std"].to(device)

    output: List[torch.Tensor] = []
    row_indices = torch.arange(len(visual), dtype=torch.long)
    visual_indices = permutation if control in {"shuffle_visual", "shuffle_both"} else row_indices
    traj_indices = permutation if control in {"shuffle_traj", "shuffle_both"} else row_indices

    for start in range(0, len(visual), batch_size):
        end = min(start + batch_size, len(visual))
        size = end - start
        if control == "mean_visual":
            visual_batch = visual_mean.expand(size, *visual.shape[1:])
        else:
            visual_batch = visual.index_select(0, visual_indices[start:end]).to(device)
        if control == "mean_traj":
            traj_batch = traj_mean.expand(size, *traj.shape[1:])
        else:
            traj_batch = traj.index_select(0, traj_indices[start:end]).to(device)
        scalar_batch = scalar[start:end].to(device)
        visual_batch = _normalize(visual_batch, visual_mean, visual_std, clip)
        scalar_batch = _normalize(scalar_batch, scalar_mean, scalar_std, clip)
        traj_batch = _normalize(traj_batch, traj_mean, traj_std, clip)
        output.append(model(visual_batch, scalar_batch, traj_batch).cpu())
    return torch.cat(output, dim=0)


def _empty_metrics() -> Dict[str, Any]:
    return {
        "num_groups": 0,
        "strict_gt_top1_count": 0,
        "acceptable_top1_count": 0,
        "hard_mismatch_top1_count": 0,
        "top_sources": Counter(),
        "pairwise": defaultdict(
            lambda: {
                "count": 0,
                "gt_better_count": 0,
                "gt_margin_sum": 0.0,
                "best_acceptable_better_count": 0,
                "best_acceptable_margin_sum": 0.0,
            }
        ),
        "source_scores": defaultdict(
            lambda: {"count": 0, "sum": 0.0, "sum_sq": 0.0}
        ),
    }


def _evaluate(
    rows: Sequence[Dict[str, Any]],
    scores: torch.Tensor,
    *,
    group_key: str,
    source_key: str,
    acceptable_sources: set[str],
    hard_sources: set[str],
) -> Dict[str, Any]:
    result = _empty_metrics()
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[_group_id(row, group_key)].append(idx)
        source = _source(row, source_key)
        value = float(scores[idx])
        result["source_scores"][source]["count"] += 1
        result["source_scores"][source]["sum"] += value
        result["source_scores"][source]["sum_sq"] += value * value

    for indices in groups.values():
        if len(indices) < 2:
            continue
        result["num_groups"] += 1
        winner = max(indices, key=lambda idx: float(scores[idx]))
        winner_source = _source(rows[winner], source_key)
        result["top_sources"][winner_source] += 1
        result["strict_gt_top1_count"] += int(winner_source == "gt_pos")
        result["acceptable_top1_count"] += int(winner_source in acceptable_sources)
        result["hard_mismatch_top1_count"] += int(winner_source in hard_sources)

        gt_idx = next(
            (idx for idx in indices if _source(rows[idx], source_key) == "gt_pos"),
            None,
        )
        acceptable = [
            idx for idx in indices if _source(rows[idx], source_key) in acceptable_sources
        ]
        if gt_idx is None or not acceptable:
            continue
        gt_score = float(scores[gt_idx])
        best_acceptable_score = max(float(scores[idx]) for idx in acceptable)
        for idx in indices:
            source = _source(rows[idx], source_key)
            if source == "gt_pos":
                continue
            value = float(scores[idx])
            pair = result["pairwise"][source]
            pair["count"] += 1
            pair["gt_better_count"] += int(gt_score > value)
            pair["gt_margin_sum"] += gt_score - value
            pair["best_acceptable_better_count"] += int(best_acceptable_score > value)
            pair["best_acceptable_margin_sum"] += best_acceptable_score - value
    return result


def _fuse_scores(
    rows: Sequence[Dict[str, Any]],
    gate_logits: torch.Tensor,
    *,
    group_key: str,
    v3_score_key: str,
    beta: float,
    threshold: float,
) -> torch.Tensor:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[_group_id(row, group_key)].append(idx)
    fused = torch.empty_like(gate_logits)
    for indices in groups.values():
        group_max = max(float(gate_logits[idx]) for idx in indices)
        for idx in indices:
            base = float(rows[idx][v3_score_key])
            penalty = max(0.0, group_max - float(gate_logits[idx]) - threshold)
            fused[idx] = base - beta * penalty
    return fused


def _comparison_stats(values: torch.Tensor, normal: torch.Tensor) -> Dict[str, float]:
    x = values.double()
    y = normal.double()
    return {
        "count": int(x.numel()),
        "abs_diff_sum": float((x - y).abs().sum()),
        "x_sum": float(x.sum()),
        "y_sum": float(y.sum()),
        "x_sq_sum": float((x * x).sum()),
        "y_sq_sum": float((y * y).sum()),
        "xy_sum": float((x * y).sum()),
    }


def _merge_metrics(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key in (
        "num_groups",
        "strict_gt_top1_count",
        "acceptable_top1_count",
        "hard_mismatch_top1_count",
    ):
        target[key] += source[key]
    target["top_sources"].update(source["top_sources"])
    for name, values in source["pairwise"].items():
        for key, value in values.items():
            target["pairwise"][name][key] += value
    for name, values in source["source_scores"].items():
        for key, value in values.items():
            target["source_scores"][name][key] += value


def _merge_comparison(target: Dict[str, float], source: Dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + value


def _finalize_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    groups = int(raw["num_groups"])
    pairwise: Dict[str, Any] = {}
    for source, values in sorted(raw["pairwise"].items()):
        count = int(values["count"])
        pairwise[source] = {
            "count": count,
            "gt_better_rate": values["gt_better_count"] / count if count else None,
            "gt_mean_margin": values["gt_margin_sum"] / count if count else None,
            "best_acceptable_better_rate": (
                values["best_acceptable_better_count"] / count if count else None
            ),
            "best_acceptable_mean_margin": (
                values["best_acceptable_margin_sum"] / count if count else None
            ),
        }
    source_scores: Dict[str, Any] = {}
    for source, values in sorted(raw["source_scores"].items()):
        count = int(values["count"])
        mean = values["sum"] / count if count else math.nan
        variance = max(values["sum_sq"] / count - mean * mean, 0.0) if count else math.nan
        source_scores[source] = {
            "count": count,
            "mean": mean,
            "std": math.sqrt(variance),
        }
    return {
        "num_groups": groups,
        "strict_gt_top1_count": int(raw["strict_gt_top1_count"]),
        "strict_gt_top1": raw["strict_gt_top1_count"] / groups if groups else None,
        "acceptable_top1_count": int(raw["acceptable_top1_count"]),
        "acceptable_top1": raw["acceptable_top1_count"] / groups if groups else None,
        "hard_mismatch_top1_count": int(raw["hard_mismatch_top1_count"]),
        "hard_mismatch_top1": raw["hard_mismatch_top1_count"] / groups if groups else None,
        "top_sources": raw["top_sources"].most_common(),
        "pairwise": pairwise,
        "source_scores": source_scores,
    }


def _finalize_comparison(raw: Dict[str, float]) -> Dict[str, Any]:
    count = int(raw.get("count", 0))
    if not count:
        return {"count": 0, "mean_abs_logit_change": None, "corr_with_normal": None}
    x_mean = raw["x_sum"] / count
    y_mean = raw["y_sum"] / count
    covariance = raw["xy_sum"] / count - x_mean * y_mean
    x_var = max(raw["x_sq_sum"] / count - x_mean * x_mean, 0.0)
    y_var = max(raw["y_sq_sum"] / count - y_mean * y_mean, 0.0)
    denom = math.sqrt(x_var * y_var)
    return {
        "count": count,
        "mean_abs_logit_change": raw["abs_diff_sum"] / count,
        "corr_with_normal": covariance / denom if denom > 0.0 else None,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, defaultdict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=ROWS,CACHE",
    )
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--visual-cache-key", default="x_tokens")
    parser.add_argument("--scalar-feature-mode", choices=["full", "zero"], default="zero")
    parser.add_argument("--controls", default=DEFAULT_CONTROLS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--source-key", default="source_type")
    parser.add_argument("--v3-score-key", default="iac_acceptability_calibrated")
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--acceptable-sources", default=DEFAULT_ACCEPTABLE_SOURCES)
    parser.add_argument("--hard-sources", default=DEFAULT_HARD_SOURCES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    controls = [item.strip() for item in args.controls.split(",") if item.strip()]
    valid_controls = {
        "normal",
        "shuffle_visual",
        "shuffle_traj",
        "shuffle_both",
        "mean_visual",
        "mean_traj",
    }
    unknown = set(controls) - valid_controls
    if unknown:
        raise ValueError(f"unknown controls: {sorted(unknown)}")
    if "normal" not in controls:
        raise ValueError("controls must include normal")

    acceptable_sources = _parse_sources(args.acceptable_sources)
    hard_sources = _parse_sources(args.hard_sources)
    bundle = _load_gate(Path(args.model))
    aggregate = {
        control: {
            "gate": _empty_metrics(),
            "fused": _empty_metrics(),
            "vs_normal": {},
        }
        for control in controls
    }
    shard_summaries: List[Dict[str, Any]] = []

    for dataset_index, raw_dataset in enumerate(args.dataset):
        name, rows_path, cache_path = _parse_dataset(raw_dataset)
        rows, visual, scalar, traj = _load_dataset(
            rows_path,
            cache_path,
            feature_key=args.visual_cache_key,
            scalar_feature_mode=args.scalar_feature_mode,
        )
        permutation, permutation_summary = _group_source_permutation(
            rows,
            group_key=args.group_key,
            source_key=args.source_key,
            seed=args.seed + dataset_index,
        )
        control_logits: Dict[str, torch.Tensor] = {}
        for control in controls:
            logits = _score_control(
                bundle,
                visual,
                scalar,
                traj,
                control=control,
                permutation=permutation,
                batch_size=args.batch_size,
                device=device,
            )
            control_logits[control] = logits
            print(
                json.dumps(
                    {
                        "dataset": name,
                        "control": control,
                        "rows": len(rows),
                        "logit_mean": float(logits.mean()),
                    }
                ),
                flush=True,
            )

        normal_logits = control_logits["normal"]
        shard_controls: Dict[str, Any] = {}
        for control, logits in control_logits.items():
            gate_metrics = _evaluate(
                rows,
                logits,
                group_key=args.group_key,
                source_key=args.source_key,
                acceptable_sources=acceptable_sources,
                hard_sources=hard_sources,
            )
            fused_scores = _fuse_scores(
                rows,
                logits,
                group_key=args.group_key,
                v3_score_key=args.v3_score_key,
                beta=args.beta,
                threshold=args.threshold,
            )
            fused_metrics = _evaluate(
                rows,
                fused_scores,
                group_key=args.group_key,
                source_key=args.source_key,
                acceptable_sources=acceptable_sources,
                hard_sources=hard_sources,
            )
            comparison = _comparison_stats(logits, normal_logits)
            _merge_metrics(aggregate[control]["gate"], gate_metrics)
            _merge_metrics(aggregate[control]["fused"], fused_metrics)
            _merge_comparison(aggregate[control]["vs_normal"], comparison)
            shard_controls[control] = {
                "gate": _finalize_metrics(gate_metrics),
                "fused": _finalize_metrics(fused_metrics),
                "vs_normal": _finalize_comparison(comparison),
            }
        shard_summaries.append(
            {
                "name": name,
                "rows": str(rows_path),
                "visual_cache": str(cache_path),
                "permutation": permutation_summary,
                "controls": shard_controls,
            }
        )

        del rows, visual, scalar, traj, permutation, control_logits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_final = {
        control: {
            "gate": _finalize_metrics(values["gate"]),
            "fused": _finalize_metrics(values["fused"]),
            "vs_normal": _finalize_comparison(values["vs_normal"]),
        }
        for control, values in aggregate.items()
    }
    summary = {
        "kind": "clean_gate_counterfactual_audit",
        "definition": {
            "shuffle_visual": "Move visual tokens to another group while preserving source type.",
            "shuffle_traj": "Move trajectory tokens to another group while preserving source type.",
            "shuffle_both": "Move the paired visual and trajectory tokens together to another group.",
            "mean_visual": "Replace visual tokens by the training mean (normalized zero).",
            "mean_traj": "Replace trajectory tokens by the training mean (normalized zero).",
        },
        "config": {
            "model": args.model,
            "device": str(device),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "visual_cache_key": args.visual_cache_key,
            "scalar_feature_mode": args.scalar_feature_mode,
            "v3_score_key": args.v3_score_key,
            "beta": args.beta,
            "threshold": args.threshold,
            "acceptable_sources": sorted(acceptable_sources),
            "hard_sources": sorted(hard_sources),
        },
        "aggregate": aggregate_final,
        "datasets": shard_summaries,
    }
    output = Path(args.output_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate_final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

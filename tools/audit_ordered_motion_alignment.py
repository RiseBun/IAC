#!/usr/bin/env python3
"""Audit ordered motion evidence with identity and token-order controls.

Compressed-token controls test the downstream alignment module.  They do not
replace raw-frame reversal/shuffle, which must be re-extracted through the
backbone using ``make_temporal_control_rows.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_extensions.ordered_motion_alignment import (  # noqa: E402
    MOTION_FAMILIES,
    load_bundle,
    load_feature_cache,
    match_rows_to_features,
    score_batches,
    trajectory_targets_from_rows,
)
from ordered_motion_common import (  # noqa: E402
    DEFAULT_ACCEPTABLE_SOURCES,
    DEFAULT_HARD_SOURCES,
    group_id,
    load_rows,
    ranking_metrics,
    sattolo_indices,
    sha256,
    source,
    split_csv,
    write_json,
    write_jsonl,
)


CONTROL_NAMES: Tuple[str, ...] = (
    "normal",
    "reverse_compressed_visual_time",
    "permute_compressed_visual_time",
    "reverse_trajectory_segments",
    "permute_trajectory_segments",
    "candidate_derangement",
    "visual_group_derangement",
)


def _grouped_indices(rows: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_id(row)].append(index)
    return list(grouped.values())


def _candidate_derangement(
    rows: Sequence[Mapping[str, Any]],
    targets: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    result = targets.clone()
    for group_index, indices in enumerate(_grouped_indices(rows)):
        permutation = sattolo_indices(len(indices), seed + group_index * 7919)
        for local_index, target_local_index in enumerate(permutation):
            result[indices[local_index]] = targets[indices[target_local_index]]
    return result


def _visual_group_derangement(
    rows: Sequence[Mapping[str, Any]],
    visual: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_id(row)].append(index)
    group_names = sorted(grouped)
    group_permutation = sattolo_indices(len(group_names), seed)
    result = visual.clone()
    for group_position, target_position in enumerate(group_permutation):
        destination_indices = grouped[group_names[group_position]]
        source_indices = grouped[group_names[target_position]]
        source_by_type: Dict[str, List[int]] = defaultdict(list)
        for index in source_indices:
            source_by_type[source(rows[index])].append(index)
        for local_index, destination in enumerate(destination_indices):
            aligned = source_by_type.get(source(rows[destination]), [])
            source_index = (
                aligned[local_index % len(aligned)]
                if aligned
                else source_indices[local_index % len(source_indices)]
            )
            result[destination] = visual[source_index]
    return result


def _apply_control(
    control: str,
    rows: Sequence[Mapping[str, Any]],
    visual: torch.Tensor,
    targets: torch.Tensor,
    *,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if control == "normal":
        return visual, targets
    if control == "reverse_compressed_visual_time":
        return visual.flip(1), targets
    if control == "permute_compressed_visual_time":
        order = torch.randperm(
            visual.shape[1],
            generator=torch.Generator().manual_seed(seed),
        )
        return visual[:, order], targets
    if control == "reverse_trajectory_segments":
        return visual, targets.flip(1)
    if control == "permute_trajectory_segments":
        order = torch.randperm(
            targets.shape[1],
            generator=torch.Generator().manual_seed(seed + 1),
        )
        return visual, targets[:, order]
    if control == "candidate_derangement":
        return visual, _candidate_derangement(rows, targets, seed=seed)
    if control == "visual_group_derangement":
        return _visual_group_derangement(rows, visual, seed=seed), targets
    raise ValueError(f"unknown control: {control}")


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


@torch.no_grad()
def audit(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    bundle = load_bundle(Path(args.model), device=device)
    feature_by_sample, cache_metadata = load_feature_cache(
        Path(args.visual_cache),
        key=args.feature_key,
    )
    rows, visual = match_rows_to_features(
        load_rows(Path(args.rows)),
        feature_by_sample,
    )
    targets = trajectory_targets_from_rows(
        rows,
        segment_count=int(bundle["model"].config.segment_count),
    )
    acceptable = split_csv(args.acceptable_sources)
    hard = split_csv(args.hard_sources)
    scores_by_control: Dict[str, List[float]] = {}
    family_by_control: Dict[str, List[List[float]]] = {}
    metrics: Dict[str, Any] = {}

    for control in CONTROL_NAMES:
        controlled_visual, controlled_targets = _apply_control(
            control,
            rows,
            visual,
            targets,
            seed=int(args.seed),
        )
        result = score_batches(
            bundle,
            controlled_visual,
            controlled_targets,
            batch_size=int(args.batch_size),
            device=device,
        )
        scores = [
            -float(value)
            for value in result["ordered_motion_energy"].tolist()
        ]
        scores_by_control[control] = scores
        family_by_control[control] = result["family_contribution"].tolist()
        metrics[control] = ranking_metrics(
            rows,
            scores,
            acceptable_sources=acceptable,
            hard_sources=hard,
        )

    normal = scores_by_control["normal"]
    delta: Dict[str, Any] = {}
    for control, values in scores_by_control.items():
        if control == "normal":
            continue
        absolute = [
            abs(float(left) - float(right))
            for left, right in zip(normal, values)
        ]
        delta[control] = {
            "mean_absolute_score_delta": _mean(absolute),
            "max_absolute_score_delta": max(absolute, default=0.0),
            "fraction_changed_gt_1e_6": (
                sum(value > 1e-6 for value in absolute) / max(len(absolute), 1)
            ),
        }

    summary = {
        "kind": "ordered_motion_alignment_control_audit",
        "protocol": {
            "seed": int(args.seed),
            "rows": len(rows),
            "feature_key": args.feature_key,
            "controls": list(CONTROL_NAMES),
            "source_labels_used_as_model_input": False,
            "source_labels_used_for_report_metrics_only": True,
            "warning": (
                "Time-token reverse/permutation tests the downstream alignment "
                "module after feature extraction. Raw-frame controls still "
                "require backbone re-extraction."
            ),
        },
        "model": str(args.model),
        "model_sha256": sha256(Path(args.model)),
        "visual_cache": str(args.visual_cache),
        "visual_cache_metadata": cache_metadata,
        "metrics": metrics,
        "control_score_delta_from_normal": delta,
    }
    if args.output_ledger:
        ledger: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            family_map = {
                name: float(family_by_control["normal"][index][family_index])
                for family_index, name in enumerate(MOTION_FAMILIES)
            }
            ledger.append(
                {
                    "sample_id": str(row.get("sample_id", "")),
                    "group_id": group_id(row),
                    "source_type_report_only": source(row),
                    "normal_ordered_motion_rank_score": normal[index],
                    "normal_family_contribution": family_map,
                    "control_rank_scores": {
                        control: values[index]
                        for control, values in scores_by_control.items()
                        if control != "normal"
                    },
                }
            )
        write_jsonl(Path(args.output_ledger), ledger)
        summary["output_ledger"] = str(args.output_ledger)
        summary["output_ledger_sha256"] = sha256(Path(args.output_ledger))
    write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--visual-cache", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-ledger", default="")
    parser.add_argument("--feature-key", default="x_time_tokens")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--acceptable-sources",
        default=DEFAULT_ACCEPTABLE_SOURCES,
    )
    parser.add_argument("--hard-sources", default=DEFAULT_HARD_SOURCES)
    return parser.parse_args()


def main() -> None:
    audit(parse_args())


if __name__ == "__main__":
    main()

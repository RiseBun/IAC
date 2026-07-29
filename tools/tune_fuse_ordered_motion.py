#!/usr/bin/env python3
"""Tune conservative IAC + ordered-motion fusion on validation only.

The transformation uses only the primary score, ordered motion energy and
within-group statistics.  Source labels are used to choose hyperparameters on
the validation split and to report metrics, never as inference inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ordered_motion_common import (  # noqa: E402
    DEFAULT_ACCEPTABLE_SOURCES,
    DEFAULT_HARD_SOURCES,
    finite_float,
    group_id,
    load_rows,
    ranking_metrics,
    split_csv,
    write_json,
    write_jsonl,
)


def _parse_grid(value: str) -> List[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("fusion grid cannot be empty")
    return result


def _join(
    primary_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    energy_key: str,
) -> List[Dict[str, Any]]:
    evidence_by_id = {
        str(row.get("sample_id", "")): row
        for row in evidence_rows
    }
    output: List[Dict[str, Any]] = []
    for raw in primary_rows:
        sample_id = str(raw.get("sample_id", ""))
        evidence = evidence_by_id.get(sample_id)
        if evidence is None:
            raise ValueError(f"missing ordered-motion evidence for {sample_id!r}")
        row = dict(raw)
        row[energy_key] = finite_float(evidence.get(energy_key), math.nan)
        if not math.isfinite(row[energy_key]):
            raise ValueError(f"non-finite ordered-motion energy for {sample_id!r}")
        output.append(row)
    return output


def _penalties(
    rows: Sequence[Mapping[str, Any]],
    *,
    energy_key: str,
    threshold: float,
) -> List[float]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_id(row)].append(index)
    result = [0.0] * len(rows)
    for indices in grouped.values():
        energies = [finite_float(rows[index].get(energy_key)) for index in indices]
        median = statistics.median(energies)
        absolute = [abs(value - median) for value in energies]
        scale = 1.4826 * statistics.median(absolute)
        if scale < 1e-6 and len(energies) >= 2:
            scale = statistics.pstdev(energies)
        scale = max(scale, 1e-6)
        for index, energy in zip(indices, energies):
            z_value = (energy - median) / scale
            result[index] = max(0.0, z_value - float(threshold))
    return result


def _fused_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_key: str,
    energy_key: str,
    beta: float,
    threshold: float,
) -> Tuple[List[float], List[float]]:
    penalty = _penalties(
        rows,
        energy_key=energy_key,
        threshold=threshold,
    )
    scores = [
        finite_float(row.get(primary_key)) - float(beta) * penalty[index]
        for index, row in enumerate(rows)
    ]
    return scores, penalty


def _objective(
    metrics: Mapping[str, Any],
    *,
    hard_weight: float,
    strict_weight: float,
) -> float:
    return (
        float(metrics["acceptable_top1"])
        - float(hard_weight) * float(metrics["hard_mismatch_top1"])
        + float(strict_weight) * float(metrics["strict_gt_top1"])
    )


def tune_and_apply(args: argparse.Namespace) -> Dict[str, Any]:
    acceptable = split_csv(args.acceptable_sources)
    hard = split_csv(args.hard_sources)
    val_rows = _join(
        load_rows(Path(args.val_primary)),
        load_rows(Path(args.val_evidence)),
        energy_key=args.energy_key,
    )
    eval_rows = _join(
        load_rows(Path(args.eval_primary)),
        load_rows(Path(args.eval_evidence)),
        energy_key=args.energy_key,
    )

    candidates: List[Dict[str, Any]] = []
    for beta in _parse_grid(args.beta_grid):
        for threshold in _parse_grid(args.threshold_grid):
            scores, _ = _fused_scores(
                val_rows,
                primary_key=args.primary_key,
                energy_key=args.energy_key,
                beta=beta,
                threshold=threshold,
            )
            metrics = ranking_metrics(
                val_rows,
                scores,
                acceptable_sources=acceptable,
                hard_sources=hard,
            )
            candidates.append(
                {
                    "beta": beta,
                    "threshold": threshold,
                    "objective": _objective(
                        metrics,
                        hard_weight=float(args.hard_weight),
                        strict_weight=float(args.strict_weight),
                    ),
                    "metrics": metrics,
                }
            )
    selected = max(
        candidates,
        key=lambda item: (
            float(item["objective"]),
            -float(item["beta"]),
            float(item["threshold"]),
        ),
    )
    beta = float(selected["beta"])
    threshold = float(selected["threshold"])
    eval_scores, eval_penalty = _fused_scores(
        eval_rows,
        primary_key=args.primary_key,
        energy_key=args.energy_key,
        beta=beta,
        threshold=threshold,
    )
    output_rows: List[Dict[str, Any]] = []
    for index, raw in enumerate(eval_rows):
        row = dict(raw)
        row["ordered_motion_within_group_penalty"] = float(eval_penalty[index])
        row["iac_ordered_motion_fused_rank_score"] = float(eval_scores[index])
        output_rows.append(row)
    write_jsonl(Path(args.output_scores), output_rows)
    eval_metrics = ranking_metrics(
        output_rows,
        eval_scores,
        acceptable_sources=acceptable,
        hard_sources=hard,
    )
    base_eval_metrics = ranking_metrics(
        output_rows,
        [finite_float(row.get(args.primary_key)) for row in output_rows],
        acceptable_sources=acceptable,
        hard_sources=hard,
    )
    summary = {
        "kind": "validation_tuned_ordered_motion_fusion",
        "protocol": {
            "selection_split": "validation",
            "evaluation_split_used_for_selection": False,
            "source_labels_used_as_inference_inputs": False,
            "source_labels_used_for_validation_selection_and_reporting": True,
            "primary_key": args.primary_key,
            "energy_key": args.energy_key,
            "fusion": (
                "primary - beta * relu(within_group_robust_z_energy - threshold)"
            ),
        },
        "selected": selected,
        "grid": candidates,
        "eval_base_metrics": base_eval_metrics,
        "eval_fused_metrics": eval_metrics,
        "output_scores": str(args.output_scores),
    }
    write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-primary", required=True)
    parser.add_argument("--val-evidence", required=True)
    parser.add_argument("--eval-primary", required=True)
    parser.add_argument("--eval-evidence", required=True)
    parser.add_argument("--primary-key", required=True)
    parser.add_argument("--energy-key", default="ordered_motion_energy")
    parser.add_argument("--beta-grid", default="0,0.025,0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--threshold-grid", default="0,0.5,1.0")
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--strict-weight", type=float, default=0.05)
    parser.add_argument(
        "--acceptable-sources",
        default=DEFAULT_ACCEPTABLE_SOURCES,
    )
    parser.add_argument("--hard-sources", default=DEFAULT_HARD_SOURCES)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--output-summary", required=True)
    return parser.parse_args()


def main() -> None:
    tune_and_apply(parse_args())


if __name__ == "__main__":
    main()

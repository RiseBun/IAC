#!/usr/bin/env python3
"""Score candidates with additive ordered motion evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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
    load_rows,
    ranking_metrics,
    sha256,
    source,
    split_csv,
    write_json,
    write_jsonl,
)


def build_scored_rows(
    rows: Sequence[Mapping[str, Any]],
    result: Mapping[str, torch.Tensor],
    *,
    include_segment_ledger: bool,
) -> List[Dict[str, Any]]:
    energies = result["ordered_motion_energy"].tolist()
    family = result["family_contribution"].tolist()
    predictions = result["visual_motion_mean"].tolist()
    targets = result["candidate_motion_target"].tolist()
    standard_deviation = result["visual_motion_standard_deviation"].tolist()
    normalized_residual = result["normalized_residual"].tolist()
    contribution = result["segment_component_contribution"].tolist()
    attention = result["temporal_attention"].tolist()

    output: List[Dict[str, Any]] = []
    for index, raw in enumerate(rows):
        family_map = {
            name: float(family[index][family_index])
            for family_index, name in enumerate(MOTION_FAMILIES)
        }
        row = dict(raw)
        row.update(
            {
                "ordered_motion_energy": float(energies[index]),
                "ordered_motion_rank_score": -float(energies[index]),
                "ordered_motion_family_contribution": family_map,
                "ordered_motion_dominant_family": max(
                    MOTION_FAMILIES,
                    key=lambda name: family_map[name],
                ),
            }
        )
        if include_segment_ledger:
            segment_rows: List[Dict[str, Any]] = []
            for segment_index in range(len(predictions[index])):
                component_rows = []
                for family_index, name in enumerate(MOTION_FAMILIES):
                    component_rows.append(
                        {
                            "family": name,
                            "visual_estimate": float(
                                predictions[index][segment_index][family_index]
                            ),
                            "candidate_target": float(
                                targets[index][segment_index][family_index]
                            ),
                            "visual_standard_deviation": float(
                                standard_deviation[index][segment_index][family_index]
                            ),
                            "normalized_residual": float(
                                normalized_residual[index][segment_index][family_index]
                            ),
                            "energy_contribution": float(
                                contribution[index][segment_index][family_index]
                            ),
                        }
                    )
                segment_rows.append(
                    {
                        "segment_index": segment_index,
                        "components": component_rows,
                        "temporal_attention": [
                            float(value)
                            for value in attention[index][segment_index]
                        ],
                    }
                )
            row["ordered_motion_segment_ledger"] = segment_rows
        output.append(row)
    return output


@torch.no_grad()
def score(args: argparse.Namespace) -> Dict[str, Any]:
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
    segment_count = int(bundle["model"].config.segment_count)
    targets = trajectory_targets_from_rows(
        rows,
        segment_count=segment_count,
    )
    result = score_batches(
        bundle,
        visual,
        targets,
        batch_size=int(args.batch_size),
        device=device,
    )
    scored = build_scored_rows(
        rows,
        result,
        include_segment_ledger=bool(args.include_segment_ledger),
    )
    output_scores = Path(args.output_scores)
    write_jsonl(output_scores, scored)

    metrics = ranking_metrics(
        rows,
        result["ordered_motion_rank_score"].tolist()
        if "ordered_motion_rank_score" in result
        else [-float(value) for value in result["ordered_motion_energy"].tolist()],
        acceptable_sources=split_csv(args.acceptable_sources),
        hard_sources=split_csv(args.hard_sources),
    )
    summary = {
        "kind": "ordered_motion_alignment_scores",
        "model": str(args.model),
        "model_sha256": sha256(Path(args.model)),
        "rows": str(args.rows),
        "visual_cache": str(args.visual_cache),
        "visual_cache_metadata": cache_metadata,
        "matched_rows": len(rows),
        "output_scores": str(output_scores),
        "output_scores_sha256": sha256(output_scores),
        "feature_key": args.feature_key,
        "segment_count": segment_count,
        "candidate_blind_visual_estimator": True,
        "source_labels_used_as_model_input": False,
        "source_labels_used_for_report_metrics_only": True,
        "metrics": metrics,
        "source_counts_report_only": _source_counts(rows),
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--visual-cache", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--feature-key", default="x_tokens")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-segment-ledger", action="store_true")
    parser.add_argument(
        "--acceptable-sources",
        default=DEFAULT_ACCEPTABLE_SOURCES,
    )
    parser.add_argument("--hard-sources", default=DEFAULT_HARD_SOURCES)
    return parser.parse_args()


def main() -> None:
    score(parse_args())


if __name__ == "__main__":
    main()

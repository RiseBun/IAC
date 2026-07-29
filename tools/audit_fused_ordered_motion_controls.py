#!/usr/bin/env python3
"""Apply one validation-selected fusion rule to all returned control energies."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ordered_motion_common import (  # noqa: E402
    DEFAULT_ACCEPTABLE_SOURCES,
    DEFAULT_HARD_SOURCES,
    finite_float,
    load_rows,
    ranking_metrics,
    split_csv,
    write_json,
)
from tune_fuse_ordered_motion import _fused_scores  # noqa: E402

REQUIRED_CONTROLS = {
    "reverse_compressed_visual_time",
    "permute_compressed_visual_time",
    "reverse_trajectory_segments",
    "permute_trajectory_segments",
    "candidate_derangement",
    "visual_group_derangement",
}


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _join(
    primary_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    ledger_by_id = {
        str(row.get("sample_id", "")): row
        for row in ledger_rows
    }
    output: List[Dict[str, Any]] = []
    for raw in primary_rows:
        sample_id = str(raw.get("sample_id", ""))
        ledger = ledger_by_id.get(sample_id)
        if ledger is None:
            raise ValueError(f"control ledger misses sample_id {sample_id!r}")
        output.append({**dict(raw), "_control_ledger": dict(ledger)})
    return output


def _control_names(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    names: set[str] | None = None
    for row in rows:
        ledger = row["_control_ledger"]
        current = set(dict(ledger.get("control_rank_scores", {})))
        if names is None:
            names = current
        elif current != names:
            raise ValueError(
                "control ledger has inconsistent control names across rows"
            )
    missing = REQUIRED_CONTROLS - (names or set())
    if missing:
        raise ValueError(
            "control ledger is incomplete; missing "
            + ", ".join(sorted(missing))
        )
    return ["normal", *sorted(names or set())]


def _energy(row: Mapping[str, Any], control: str) -> float:
    ledger = row["_control_ledger"]
    if control == "normal":
        rank = ledger.get("normal_ordered_motion_rank_score")
    else:
        rank = dict(ledger.get("control_rank_scores", {})).get(control)
    result = -finite_float(rank, math.nan)
    if not math.isfinite(result):
        raise ValueError(
            f"non-finite {control} energy for {row.get('sample_id')!r}"
        )
    return result


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    fusion = _load_json(Path(args.fusion_summary))
    selected = dict(fusion.get("selected", {}))
    beta = float(selected["beta"])
    threshold = float(selected["threshold"])
    rows = _join(
        load_rows(Path(args.primary_rows)),
        load_rows(Path(args.control_ledger)),
    )
    acceptable = split_csv(args.acceptable_sources)
    hard = split_csv(args.hard_sources)
    base_scores = [
        finite_float(row.get(args.primary_key), math.nan)
        for row in rows
    ]
    if not all(math.isfinite(value) for value in base_scores):
        raise ValueError(f"non-finite primary score {args.primary_key!r}")
    base_metrics = ranking_metrics(
        rows,
        base_scores,
        acceptable_sources=acceptable,
        hard_sources=hard,
    )
    control_metrics: Dict[str, Any] = {}
    for control in _control_names(rows):
        controlled: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["_controlled_ordered_motion_energy"] = _energy(row, control)
            controlled.append(item)
        scores, _ = _fused_scores(
            controlled,
            primary_key=args.primary_key,
            energy_key="_controlled_ordered_motion_energy",
            beta=beta,
            threshold=threshold,
        )
        metrics = ranking_metrics(
            controlled,
            scores,
            acceptable_sources=acceptable,
            hard_sources=hard,
        )
        metrics["delta_from_base"] = {
            key: float(metrics[key]) - float(base_metrics[key])
            for key in (
                "strict_gt_top1",
                "acceptable_top1",
                "hard_mismatch_top1",
                "mrr_gt",
            )
        }
        control_metrics[control] = metrics

    normal = control_metrics["normal"]
    order_names = [
        name
        for name in (
            "reverse_compressed_visual_time",
            "permute_compressed_visual_time",
            "reverse_trajectory_segments",
            "permute_trajectory_segments",
        )
        if name in control_metrics
    ]
    identity_names = [
        name
        for name in ("candidate_derangement", "visual_group_derangement")
        if name in control_metrics
    ]
    summary = {
        "kind": "validation_selected_fusion_control_audit",
        "protocol": {
            "fusion_summary": str(args.fusion_summary),
            "selection_split": "validation",
            "evaluation_controls_used_for_selection": False,
            "primary_key": args.primary_key,
            "beta": beta,
            "threshold": threshold,
            "source_labels_used_as_inference_inputs": False,
        },
        "base_metrics": base_metrics,
        "metrics": control_metrics,
        "decision_diagnostics": {
            "normal_mrr": normal["mrr_gt"],
            "best_order_control_mrr": max(
                (
                    float(control_metrics[name]["mrr_gt"])
                    for name in order_names
                ),
                default=None,
            ),
            "normal_minus_best_order_control_mrr": (
                float(normal["mrr_gt"])
                - max(
                    float(control_metrics[name]["mrr_gt"])
                    for name in order_names
                )
                if order_names
                else None
            ),
            "normal_beats_every_order_control_mrr": all(
                float(normal["mrr_gt"])
                > float(control_metrics[name]["mrr_gt"])
                for name in order_names
            ),
            "normal_beats_every_identity_control_mrr": all(
                float(normal["mrr_gt"])
                > float(control_metrics[name]["mrr_gt"])
                for name in identity_names
            ),
        },
    }
    write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-rows", required=True)
    parser.add_argument("--primary-key", required=True)
    parser.add_argument("--control-ledger", required=True)
    parser.add_argument("--fusion-summary", required=True)
    parser.add_argument("--output-summary", required=True)
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

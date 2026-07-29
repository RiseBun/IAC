#!/usr/bin/env python3
"""Aggregate multi-seed ordered-motion engineering decision runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _named_dir(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--run must use NAME=DIR")
    name, raw = value.split("=", 1)
    return name.strip(), Path(raw)


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _mean(values: List[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    named = [_named_dir(value) for value in args.run]
    if len(named) < 2:
        raise ValueError("provide at least two --run NAME=DIR entries")
    runs: Dict[str, Any] = {}
    for name, root in named:
        fusion = _load(root / "fusion_summary.json")
        controls = _load(root / "fused_control_audit.json")
        split = _load(root / "split_independence_audit.json")
        base = dict(fusion["eval_base_metrics"])
        fused = dict(fusion["eval_fused_metrics"])
        normal = dict(controls["metrics"]["normal"])
        diagnostic = dict(controls["decision_diagnostics"])
        base_pair = dict(base.get("pairwise_gt_win", {}))
        fused_pair = dict(fused.get("pairwise_gt_win", {}))
        runs[name] = {
            "base": {
                "strict_gt_top1": base["strict_gt_top1"],
                "acceptable_top1": base["acceptable_top1"],
                "hard_mismatch_top1": base["hard_mismatch_top1"],
                "mrr_gt": base["mrr_gt"],
            },
            "fused": {
                "strict_gt_top1": fused["strict_gt_top1"],
                "acceptable_top1": fused["acceptable_top1"],
                "hard_mismatch_top1": fused["hard_mismatch_top1"],
                "mrr_gt": fused["mrr_gt"],
            },
            "delta": {
                "strict_gt_top1": (
                    float(fused["strict_gt_top1"])
                    - float(base["strict_gt_top1"])
                ),
                "acceptable_top1": (
                    float(fused["acceptable_top1"])
                    - float(base["acceptable_top1"])
                ),
                "mrr_gt": float(fused["mrr_gt"]) - float(base["mrr_gt"]),
                "perturb_speed_pairwise": (
                    float(fused_pair.get("perturb_speed", 0.0))
                    - float(base_pair.get("perturb_speed", 0.0))
                ),
                "time_shift_future_pairwise": (
                    float(fused_pair.get("time_shift_future", 0.0))
                    - float(base_pair.get("time_shift_future", 0.0))
                ),
            },
            "normal_fused_control_metrics": {
                "strict_gt_top1": normal["strict_gt_top1"],
                "acceptable_top1": normal["acceptable_top1"],
                "mrr_gt": normal["mrr_gt"],
            },
            "normal_minus_best_order_control_mrr": diagnostic[
                "normal_minus_best_order_control_mrr"
            ],
            "normal_beats_every_order_control_mrr": diagnostic[
                "normal_beats_every_order_control_mrr"
            ],
            "normal_beats_every_identity_control_mrr": diagnostic[
                "normal_beats_every_identity_control_mrr"
            ],
            "strict_scene_and_image_disjoint": split[
                "all_pairs_strict_scene_and_image_disjoint"
            ],
        }

    values = list(runs.values())
    order_margins = [
        float(item["normal_minus_best_order_control_mrr"])
        for item in values
    ]
    aggregate = {
        "mean_delta": {
            key: _mean([float(item["delta"][key]) for item in values])
            for key in (
                "strict_gt_top1",
                "acceptable_top1",
                "mrr_gt",
                "perturb_speed_pairwise",
                "time_shift_future_pairwise",
            )
        },
        "mean_normal_minus_best_order_control_mrr": _mean(order_margins),
        "seeds_normal_beats_every_order_control": sum(
            bool(item["normal_beats_every_order_control_mrr"])
            for item in values
        ),
        "seeds_normal_beats_every_identity_control": sum(
            bool(item["normal_beats_every_identity_control_mrr"])
            for item in values
        ),
        "seed_count": len(values),
        "all_splits_strict_scene_and_image_disjoint": all(
            bool(item["strict_scene_and_image_disjoint"])
            for item in values
        ),
    }
    gates = {
        "order_gate": (
            aggregate["mean_normal_minus_best_order_control_mrr"] > 0.0
            and aggregate["seeds_normal_beats_every_order_control"]
            >= max(2, len(values) - 1)
        ),
        "identity_gate": (
            aggregate["seeds_normal_beats_every_identity_control"]
            == len(values)
        ),
        "overall_gate": (
            aggregate["mean_delta"]["mrr_gt"] >= 0.0
            and aggregate["mean_delta"]["acceptable_top1"] >= 0.0
        ),
        "target_family_gate": (
            aggregate["mean_delta"]["perturb_speed_pairwise"] >= 0.0
            and aggregate["mean_delta"]["time_shift_future_pairwise"] >= 0.0
        ),
    }
    gates["advance_corrected_time_head"] = all(gates.values())
    summary = {
        "kind": "ordered_motion_multi_seed_engineering_decision",
        "runs": runs,
        "aggregate": aggregate,
        "preregistered_gates": gates,
        "interpretation": {
            "advance_corrected_time_head": (
                "Advance to a larger strict split only if true."
            ),
            "formal_evidence_ready": (
                bool(gates["advance_corrected_time_head"])
                and bool(
                    aggregate[
                        "all_splits_strict_scene_and_image_disjoint"
                    ]
                )
            ),
            "failure_action": (
                "If the order gate fails, stop expanding the generic token "
                "alignment head and prioritize candidate-induced physical "
                "flow/track residuals."
            ),
        },
    }
    output = Path(args.output_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-summary", required=True)
    return parser.parse_args()


def main() -> None:
    summarize(parse_args())


if __name__ == "__main__":
    main()

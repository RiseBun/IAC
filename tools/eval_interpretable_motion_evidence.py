#!/usr/bin/env python3
"""Export a per-candidate, additive motion evidence ledger.

The output answers three separate questions instead of hiding them in one
consistency score:

1. What motion did the video-only head estimate?
2. What motion does this candidate trajectory imply?
3. Which named residuals contributed to rejection, and by how much?

Source labels are copied for reporting only.  They are never inputs to the
visual estimator, comparator, or family aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_extensions.motion_attribute_layout import (  # noqa: E402
    MOTION_FAMILY_NAMES,
    build_motion_attribute_layout,
)
from tools.eval_scope_motion_evidence import (  # noqa: E402
    _apply_control,
    _group_id,
    _is_positive,
    _select_indices,
    _source,
    _summarize_rows,
)
from train import ConsistencyDataset, load_config  # noqa: E402
from train_scope_interpretable_motion_head import (  # noqa: E402
    InterpretableScopeDinoMotionCritic,
)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _family_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    top_family_by_source: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        source = str(row["source"])
        by_source[source].append(row)
        contributions = row["motion_family_contribution"]
        top = max(MOTION_FAMILY_NAMES, key=lambda name: float(contributions[name]))
        top_family_by_source[source][top] += 1

    source_summary: Dict[str, Any] = {}
    for source, items in sorted(by_source.items()):
        source_summary[source] = {
            "count": len(items),
            "mean_family_contribution": {
                family: _mean(
                    [
                        float(item["motion_family_contribution"][family])
                        for item in items
                    ]
                )
                for family in MOTION_FAMILY_NAMES
            },
            "top_family_rate": {
                family: top_family_by_source[source][family] / len(items)
                for family in MOTION_FAMILY_NAMES
            },
        }
    reconstruction = [
        abs(
            float(row["scope_motion_energy"])
            - sum(
                float(row["motion_family_contribution"][family])
                for family in MOTION_FAMILY_NAMES
            )
        )
        for row in rows
    ]
    return {
        "family_names": list(MOTION_FAMILY_NAMES),
        "source_summary_report_only": source_summary,
        "max_energy_reconstruction_error": max(reconstruction, default=0.0),
        "mean_energy_reconstruction_error": _mean(reconstruction),
        "transformation_uses_source_labels": False,
    }


def _attribute_ledger(
    *,
    names: Sequence[str],
    families: Sequence[str],
    visual: Sequence[float],
    target: Sequence[float],
    standard_deviation: Sequence[float],
    normalized_residual: Sequence[float],
    contribution: Sequence[float],
) -> List[Dict[str, Any]]:
    return [
        {
            "name": str(name),
            "family": str(family),
            "visual_estimate": float(visual_value),
            "candidate_target": float(target_value),
            "visual_standard_deviation": float(std_value),
            "normalized_residual": float(residual_value),
            "energy_contribution": float(contribution_value),
        }
        for (
            name,
            family,
            visual_value,
            target_value,
            std_value,
            residual_value,
            contribution_value,
        ) in zip(
            names,
            families,
            visual,
            target,
            standard_deviation,
            normalized_residual,
            contribution,
        )
    ]


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    cfg["baseline_mode"] = "full"
    index_path = args.index or cfg[
        "val_index" if args.split == "val" else "train_index"
    ]
    dataset = ConsistencyDataset(index_path, cfg, training=False)
    indices = _select_indices(
        dataset,
        args.max_groups,
        args.max_samples,
        args.seed,
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = InterpretableScopeDinoMotionCritic(cfg).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if (missing or unexpected) and not args.allow_checkpoint_mismatch:
        raise ValueError(
            "checkpoint/model mismatch; rerun with the matching SCOPE config or "
            "--allow-checkpoint-mismatch for an explicit diagnostic only. "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()

    layout = build_motion_attribute_layout(model.motion_rule_segment_count)
    source_rows = list(getattr(dataset, "samples", []))
    rows_out: List[Dict[str, Any]] = []
    offset = 0
    for batch in loader:
        size = int(batch["candidate_traj"].shape[0])
        batch_indices = indices[offset : offset + size]
        offset += size
        history = batch["history_images"].to(device, non_blocking=True)
        future = batch["future_images"].to(device, non_blocking=True)
        future = _apply_control(future, args.control, args.seed)
        ego = batch["ego_state"].to(device, non_blocking=True)
        trajectory = batch["candidate_traj"].to(device, non_blocking=True)
        output = model(history, future, ego, trajectory)

        visual = output["visual_motion_rule_pred"].detach().cpu().float().tolist()
        target = output["traj_motion_rule_target"].detach().cpu().float().tolist()
        standard_deviation = output[
            "scope_motion_visual_standard_deviation"
        ].detach().cpu().float().tolist()
        normalized_residual = output[
            "scope_motion_normalized_residual"
        ].detach().cpu().float().tolist()
        component_contribution = output[
            "scope_motion_weighted_component_contribution"
        ].detach().cpu().float().tolist()
        family_contribution = output[
            "scope_motion_family_contribution"
        ].detach().cpu().float().tolist()
        energy = output["scope_motion_energy"].detach().cpu().float().tolist()
        match_logit = output["motion_rule_match_logit"].detach().cpu().float().tolist()
        consistency = output["consistency_logit"].detach().cpu().float().tolist()

        for local_index, dataset_index in enumerate(batch_indices):
            raw = source_rows[dataset_index] if source_rows else {}
            family = {
                name: float(family_contribution[local_index][family_index])
                for family_index, name in enumerate(MOTION_FAMILY_NAMES)
            }
            row: Dict[str, Any] = {
                "sample_id": str(
                    raw.get(
                        "sample_id",
                        batch.get("sample_id", [""])[local_index],
                    )
                ),
                "group_id": _group_id(raw, str(dataset_index)),
                "source": _source(raw),
                "is_positive": _is_positive(raw),
                "scope_motion_energy": float(energy[local_index]),
                "motion_rule_match_logit": float(match_logit[local_index]),
                "consistency_logit": float(consistency[local_index]),
                "motion_family_contribution": family,
                "dominant_motion_family": max(
                    MOTION_FAMILY_NAMES,
                    key=lambda name: family[name],
                ),
            }
            if not args.omit_attribute_ledger:
                row["motion_attribute_ledger"] = _attribute_ledger(
                    names=layout.attribute_names,
                    families=layout.attribute_families,
                    visual=visual[local_index],
                    target=target[local_index],
                    standard_deviation=standard_deviation[local_index],
                    normalized_residual=normalized_residual[local_index],
                    contribution=component_contribution[local_index],
                )
            rows_out.append(row)

    summary = _summarize_rows(rows_out)
    summary["interpretable_motion"] = _family_summary(rows_out)
    summary["checkpoint_load"] = {
        "epoch": checkpoint.get("epoch"),
        "best_metric_name": checkpoint.get("best_metric_name"),
        "best_metric_value": checkpoint.get("best_metric_value"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    summary["config"] = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "index": index_path,
        "split": args.split,
        "control": args.control,
        "max_groups": args.max_groups,
        "max_samples": args.max_samples,
        "rows": len(rows_out),
        "attribute_dim": layout.attribute_dim,
        "attribute_names": list(layout.attribute_names),
        "attribute_families": list(layout.attribute_families),
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
    parser.add_argument("--index", default="")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--control",
        choices=[
            "normal",
            "reverse_future",
            "shuffle_future",
            "roll_future",
            "zero_future",
        ],
        default="normal",
    )
    parser.add_argument("--max-groups", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-rows", default="")
    parser.add_argument("--omit-attribute-ledger", action="store_true")
    parser.add_argument("--allow-checkpoint-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    output = Path(args.output_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

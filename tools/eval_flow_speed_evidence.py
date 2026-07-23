#!/usr/bin/env python3
"""Evaluate candidate-blind optical-flow speed evidence.

Lower ``flow_speed_energy`` means the visual speed prediction is closer to the
candidate trajectory speed attributes.  This evaluator mirrors
``eval_scope_motion_evidence.py`` but uses cached/classical optical flow instead
of a learned image head.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_extensions.flow_evidence import (  # noqa: E402
    FLOW_METHODS,
    RidgeSpeedHead,
    speed_energy,
    trajectory_speed_targets,
)
from tools.flow_speed_head import (  # noqa: E402
    extract_features,
    load_extractor_settings,
    read_jsonl,
)


HARD_SOURCES = {
    "image_swap",
    "time_shift_future",
    "traj_swap",
    "reverse_traj",
    "high_pdm_image_mismatch",
}
NEAR_SOURCES = {"perturb_speed", "perturb_lateral", "perturb_heading"}


def _source(row: Mapping[str, Any]) -> str:
    for key in ("source_type", "action_type", "wam_name", "sample_type", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _group_id(row: Mapping[str, Any], fallback: str) -> str:
    value = row.get("group_id") or row.get("anchor_id")
    if value is not None:
        return str(value)
    sample_id = str(row.get("sample_id", fallback))
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def _is_positive(row: Mapping[str, Any]) -> bool:
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


def _auc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float | None:
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    total = 0
    for pos in pos_scores:
        for neg in neg_scores:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
            total += 1
    return wins / total if total else None


def _select_rows(
    rows: Sequence[Dict[str, Any]],
    max_groups: int,
    max_samples: int,
    seed: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    indexed = list(enumerate(rows))
    if max_groups <= 0:
        selected = indexed
    else:
        groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for index, row in indexed:
            groups[_group_id(row, str(index))].append((index, row))
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        selected = []
        for key in keys[:max_groups]:
            selected.extend(groups[key])
    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def _controlled_sequence(row: Mapping[str, Any], image_root: Path, control: str) -> Tuple[str, ...]:
    history = [str(value) for value in row.get("history_images", [])]
    future = [str(value) for value in row.get("future_images", [])]
    if control == "normal":
        controlled_future = future
    elif control == "reverse_future":
        controlled_future = list(reversed(future))
    elif control == "roll_future":
        controlled_future = future[-1:] + future[:-1] if len(future) > 1 else future
    elif control == "shuffle_future":
        controlled_future = future[1::2] + future[::2]
        if controlled_future == future and len(future) > 1:
            controlled_future = list(reversed(future))
    else:
        raise ValueError(f"unknown control: {control}")
    values = history + controlled_future
    if len(values) < 2:
        raise ValueError("index row does not contain a usable visual sequence")
    resolved = []
    for raw in values:
        path = Path(raw)
        if not path.is_absolute():
            path = image_root / path
        resolved.append(str(path))
    return tuple(resolved)


def _summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    source_summary = {}
    for source, items in sorted(by_source.items()):
        energies = [float(row["flow_speed_energy"]) for row in items]
        source_summary[source] = {
            "count": len(items),
            "energy_mean": _mean(energies),
            "energy_p10": _quantile(energies, 0.1),
            "energy_p50": _quantile(energies, 0.5),
            "energy_p90": _quantile(energies, 0.9),
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
        gt_energy = float(gt["flow_speed_energy"])
        hard_hit = False
        near_hit = False
        for row in items:
            if row is gt:
                continue
            source = str(row["source"])
            energy = float(row["flow_speed_energy"])
            pairwise[f"gt_better_energy_vs_{source}"].append(float(gt_energy < energy))
            if source in HARD_SOURCES and energy < gt_energy:
                hard_hit = True
            if source in NEAR_SOURCES and energy < gt_energy:
                near_hit = True
        hard_above_gt.append(float(hard_hit))
        near_above_gt.append(float(near_hit))

    # A higher score is better for AUC, while energy is lower-is-better.
    positives = [-float(row["flow_speed_energy"]) for row in rows if row["is_positive"]]
    hard = [
        -float(row["flow_speed_energy"])
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
        "positive_vs_hard_energy_auc": _auc(positives, hard),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    rows = read_jsonl(Path(args.index))
    selected = _select_rows(rows, args.max_groups, args.max_samples, args.seed)
    model_path = Path(args.model)
    model = RidgeSpeedHead.load(model_path)
    method, width, height = load_extractor_settings(model_path, args)
    image_root = Path(args.image_root)
    sequences = [
        _controlled_sequence(row, image_root, args.control)
        for _, row in selected
    ]
    features = extract_features(
        sequences,
        method=method,
        width=width,
        height=height,
        workers=args.workers,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    prediction = model.predict(features)
    targets = np.stack([trajectory_speed_targets(row["candidate_traj"]) for _, row in selected])
    energies = speed_energy(prediction, targets, model)
    rows_out = []
    for (index, row), visual, target, energy in zip(selected, prediction, targets, energies):
        rows_out.append(
            {
                "sample_id": str(row.get("sample_id", index)),
                "group_id": _group_id(row, str(index)),
                "source": _source(row),
                "is_positive": _is_positive(row),
                "flow_speed_prediction": [float(value) for value in visual],
                "flow_speed_candidate": [float(value) for value in target],
                "flow_speed_energy": float(energy),
            }
        )
    summary = _summarize_rows(rows_out)
    summary["config"] = {
        "index": args.index,
        "image_root": args.image_root,
        "model": args.model,
        "control": args.control,
        "max_groups": args.max_groups,
        "max_samples": args.max_samples,
        "rows": len(rows_out),
        "method": method,
        "resolution": [width, height],
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
    parser.add_argument("--index", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--control",
        choices=["normal", "reverse_future", "roll_future", "shuffle_future"],
        default="normal",
    )
    parser.add_argument("--method", choices=FLOW_METHODS, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--max-groups", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
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

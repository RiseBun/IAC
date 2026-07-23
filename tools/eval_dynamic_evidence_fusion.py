#!/usr/bin/env python3
"""Fuse candidate-blind dynamic visual evidence rows.

The inputs are row-level outputs from ``eval_scope_motion_evidence.py`` and
``eval_flow_speed_evidence.py``. Lower fused energy is better. This is only an
evaluator: it checks whether RGB-diff motion evidence and optical-flow speed
evidence are complementary before either signal is promoted into the main
scorer.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


HARD_SOURCES = {
    "image_swap",
    "time_shift_future",
    "traj_swap",
    "reverse_traj",
    "high_pdm_image_mismatch",
}
NEAR_SOURCES = {"perturb_speed", "perturb_lateral", "perturb_heading"}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (str(row["group_id"]), str(row["sample_id"]), str(row["source"]))


def _standardize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    std = math.sqrt(variance)
    if std < 1e-8:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def _add_group_z(rows: List[Dict[str, Any]], key: str, out_key: str) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    for items in grouped.values():
        values = [float(row[key]) for row in items]
        for row, value in zip(items, _standardize(values)):
            row[out_key] = float(value)


def _merge(scope_rows: Sequence[Mapping[str, Any]], flow_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    flow_by_key = {_key(row): row for row in flow_rows}
    merged: List[Dict[str, Any]] = []
    missing = 0
    for scope in scope_rows:
        flow = flow_by_key.get(_key(scope))
        if flow is None:
            missing += 1
            continue
        merged.append(
            {
                "sample_id": str(scope["sample_id"]),
                "group_id": str(scope["group_id"]),
                "source": str(scope["source"]),
                "is_positive": bool(scope["is_positive"]),
                "scope_motion_energy": float(scope["scope_motion_energy"]),
                "motion_rule_match_logit": float(scope.get("motion_rule_match_logit", 0.0)),
                "flow_speed_energy": float(flow["flow_speed_energy"]),
            }
        )
    if missing:
        print(f"warning: skipped {missing} scope rows without matching flow rows")
    return merged


def _score_rows(rows: List[Dict[str, Any]], scope_weight: float, flow_weight: float) -> None:
    _add_group_z(rows, "scope_motion_energy", "scope_motion_energy_group_z")
    _add_group_z(rows, "flow_speed_energy", "flow_speed_energy_group_z")
    for row in rows:
        row["fused_dynamic_energy"] = (
            scope_weight * float(row["scope_motion_energy_group_z"])
            + flow_weight * float(row["flow_speed_energy_group_z"])
        )


def _summarize(rows: Sequence[Mapping[str, Any]], score_key: str) -> Dict[str, Any]:
    by_source: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    source_summary: Dict[str, Dict[str, Any]] = {}
    for source, items in sorted(by_source.items()):
        energies = [float(row[score_key]) for row in items]
        source_summary[source] = {
            "count": len(items),
            "energy_mean": _mean(energies),
            "energy_p10": _quantile(energies, 0.1),
            "energy_p50": _quantile(energies, 0.5),
            "energy_p90": _quantile(energies, 0.9),
        }

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)

    pairwise: Dict[str, List[float]] = defaultdict(list)
    hard_above_gt: List[float] = []
    near_above_gt: List[float] = []
    for items in grouped.values():
        positives = [row for row in items if bool(row["is_positive"])]
        if not positives:
            continue
        gt = positives[0]
        gt_energy = float(gt[score_key])
        hard_hit = False
        near_hit = False
        for row in items:
            if row is gt:
                continue
            source = str(row["source"])
            energy = float(row[score_key])
            pairwise[f"gt_better_energy_vs_{source}"].append(float(gt_energy < energy))
            if source in HARD_SOURCES and energy < gt_energy:
                hard_hit = True
            if source in NEAR_SOURCES and energy < gt_energy:
                near_hit = True
        hard_above_gt.append(float(hard_hit))
        near_above_gt.append(float(near_hit))

    positives = [-float(row[score_key]) for row in rows if bool(row["is_positive"])]
    hard = [
        -float(row[score_key])
        for row in rows
        if (not bool(row["is_positive"])) and str(row["source"]) in HARD_SOURCES
    ]
    return {
        "rows": len(rows),
        "groups": len(grouped),
        "source_summary": source_summary,
        "pairwise_accuracy": {key: _mean(values) for key, values in sorted(pairwise.items())},
        "hard_mismatch_energy_above_gt_group_rate": _mean(hard_above_gt),
        "near_perturb_energy_above_gt_group_rate": _mean(near_above_gt),
        "positive_vs_hard_energy_auc": _auc(positives, hard),
    }


def _grid_weights(text: str) -> Iterable[Tuple[float, float]]:
    if not text:
        yield (1.0, 0.0)
        yield (0.0, 1.0)
        for flow_weight in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5):
            yield (1.0, flow_weight)
        return
    for item in text.split(","):
        scope, flow = item.split(":")
        yield (float(scope), float(flow))


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    scope_rows = _read_jsonl(Path(args.scope_rows))
    flow_rows = _read_jsonl(Path(args.flow_rows))
    base_rows = _merge(scope_rows, flow_rows)
    if not base_rows:
        raise ValueError("no matched evidence rows")

    rows_out: List[Dict[str, Any]] = []
    grid = []
    best = None
    for scope_weight, flow_weight in _grid_weights(args.weight_grid):
        rows = [dict(row) for row in base_rows]
        _score_rows(rows, scope_weight, flow_weight)
        summary = _summarize(rows, "fused_dynamic_energy")
        item = {
            "scope_weight": scope_weight,
            "flow_weight": flow_weight,
            "auc": summary["positive_vs_hard_energy_auc"],
            "hard_above": summary["hard_mismatch_energy_above_gt_group_rate"],
            "near_above": summary["near_perturb_energy_above_gt_group_rate"],
            "image_swap_pair": summary["pairwise_accuracy"].get("gt_better_energy_vs_image_swap"),
            "time_shift_pair": summary["pairwise_accuracy"].get("gt_better_energy_vs_time_shift_future"),
            "traj_swap_pair": summary["pairwise_accuracy"].get("gt_better_energy_vs_traj_swap"),
            "summary": summary,
        }
        grid.append(item)
        key = (
            float(item["auc"] or -1.0),
            -float(item["hard_above"] or 1.0),
            float(item["traj_swap_pair"] or -1.0),
        )
        if best is None or key > best[0]:
            best = (key, item)
            rows_out = rows

    result = {
        "best": best[1] if best else None,
        "grid": grid,
        "config": {
            "scope_rows": args.scope_rows,
            "flow_rows": args.flow_rows,
            "weight_grid": args.weight_grid,
        },
    }
    if args.output_rows and rows_out:
        path = Path(args.output_rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-rows", required=True)
    parser.add_argument("--flow-rows", required=True)
    parser.add_argument("--weight-grid", default="")
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-rows", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    out = Path(args.output_summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = [
        {
            key: row.get(key)
            for key in (
                "scope_weight",
                "flow_weight",
                "auc",
                "hard_above",
                "near_above",
                "image_swap_pair",
                "time_shift_pair",
                "traj_swap_pair",
            )
        }
        for row in summary["grid"]
    ]
    best_index = summary["grid"].index(summary["best"])
    print(json.dumps({"best": compact[best_index], "grid": compact}, indent=2))


if __name__ == "__main__":
    main()

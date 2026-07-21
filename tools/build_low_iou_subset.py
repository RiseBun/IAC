#!/usr/bin/env python3
"""Build a low path-IoU validation subset for trajectory-specific causal tests.

The standard candidate groups often contain negatives whose projected path
overlaps the positive path. This tool ranks groups by the lowest positive-vs-
negative projected path IoU, then exports full candidate groups for benchmark
evaluation. The goal is to make the causal contrast actually test exact path
dependence rather than generic road-corridor dependence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_wam import (  # noqa: E402
    _candidate_group_id,
    _candidate_traj_array,
    _load_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input consistency JSONL/JSON/PT manifest.")
    parser.add_argument("--output", required=True, help="Output JSONL subset.")
    parser.add_argument("--report", required=True, help="Output JSON report.")
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--max-groups", type=int, default=200)
    parser.add_argument(
        "--start-rank",
        type=int,
        default=0,
        help=(
            "Skip the first N eligible low-IoU groups before selecting. "
            "Use this to create a disjoint holdout slice after the tuning slice."
        ),
    )
    parser.add_argument("--max-iou", type=float, default=1.0)
    parser.add_argument("--mask-size", type=int, default=96)
    parser.add_argument("--path-mask-width", type=float, default=0.10)
    parser.add_argument(
        "--path-trajectory-mode",
        choices=("cumulative", "positions"),
        default="positions",
    )
    parser.add_argument(
        "--path-projection-mode",
        choices=("relative", "fixed"),
        default="fixed",
    )
    parser.add_argument("--path-forward-m", type=float, default=40.0)
    parser.add_argument("--path-lateral-m", type=float, default=10.0)
    return parser.parse_args()


def label(row: Dict[str, Any]) -> int:
    value = row.get("consistency_label", row.get("label", 0.0))
    return int(float(value) >= 0.5)


def source(row: Dict[str, Any]) -> str:
    return str(row.get("source_type") or row.get("sample_type") or "unknown")


def trajectory_polyline(row: Dict[str, Any], args: argparse.Namespace) -> List[tuple[int, int]]:
    size = int(args.mask_size)
    arr = _candidate_traj_array(row.get("candidate_traj", []))
    if arr.size == 0:
        return [(size // 2, size - 1), (size // 2, max(0, int(size * 0.45)))]
    xy = arr if args.path_trajectory_mode == "positions" else np.cumsum(arr, axis=0)
    forward = np.maximum(xy[:, 0], 0.0)
    lateral = xy[:, 1]
    if args.path_projection_mode == "fixed":
        max_forward = max(float(args.path_forward_m), 1.0)
        max_lateral = max(float(args.path_lateral_m), 1.0)
    else:
        max_forward = max(float(np.percentile(np.abs(forward), 90)), 1.0)
        max_lateral = max(float(np.percentile(np.abs(lateral), 90)), 2.0)

    pts = [(size // 2, size - 1)]
    for x_fwd, y_lat in zip(forward, lateral):
        v = int((size - 1) - np.clip(x_fwd / max_forward, 0.0, 1.0) * size * 0.62)
        u = int((size / 2.0) - np.clip(y_lat / max_lateral, -1.0, 1.0) * size * 0.32)
        pts.append((max(0, min(size - 1, u)), max(0, min(size - 1, v))))
    return pts


def mask_for_row(row: Dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    size = int(args.mask_size)
    radius = max(2, int(round(size * float(args.path_mask_width))))
    image = Image.new("1", (size, size), 0)
    draw = ImageDraw.Draw(image)
    pts = trajectory_polyline(row, args)
    if len(pts) >= 2:
        draw.line(pts, fill=1, width=2 * radius + 1, joint="curve")
        for x, y in pts:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
    return np.asarray(image, dtype=bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    union = float(np.logical_or(a, b).sum())
    inter = float(np.logical_and(a, b).sum())
    iou = inter / max(union, 1.0)
    exclusive = float(np.logical_xor(a, b).mean())
    return iou, exclusive


def safe_mean(values: List[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.input))

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    group_order: List[str] = []
    for row in rows:
        gid = _candidate_group_id(row, args.group_key)
        if gid is None:
            continue
        if gid not in groups:
            group_order.append(gid)
        groups[gid].append(row)

    ranked: List[Dict[str, Any]] = []
    for gid in group_order:
        items = groups[gid]
        positives = [row for row in items if label(row) == 1]
        negatives = [row for row in items if label(row) == 0]
        if not positives or not negatives:
            continue
        pos = positives[0]
        row_masks = {id(row): mask_for_row(row, args) for row in items}
        pos_mask = row_masks[id(pos)]
        best: Dict[str, Any] | None = None
        all_ious: List[float] = []
        for neg in negatives:
            iou, exclusive = mask_iou(pos_mask, row_masks[id(neg)])
            all_ious.append(iou)
            candidate = {
                "group_id": gid,
                "rows": items,
                "positive_sample_id": pos.get("sample_id"),
                "positive_source_type": source(pos),
                "wrong_sample_id": neg.get("sample_id"),
                "wrong_source_type": source(neg),
                "min_positive_negative_iou": iou,
                "exclusive_fraction": exclusive,
                "mean_positive_negative_iou": None,
                "group_size": len(items),
                "num_negatives": len(negatives),
            }
            if best is None or (
                iou,
                -exclusive,
                str(neg.get("sample_id")),
            ) < (
                best["min_positive_negative_iou"],
                -best["exclusive_fraction"],
                str(best["wrong_sample_id"]),
            ):
                best = candidate
        if best is not None:
            best["mean_positive_negative_iou"] = safe_mean(all_ious)
            ranked.append(best)

    ranked.sort(
        key=lambda item: (
            float(item["min_positive_negative_iou"]),
            -float(item["exclusive_fraction"]),
            str(item["group_id"]),
        )
    )
    eligible = [
        item for item in ranked
        if float(item["min_positive_negative_iou"]) <= float(args.max_iou)
    ]
    start_rank = max(0, int(args.start_rank))
    selected = eligible[start_rank : start_rank + max(0, int(args.max_groups))]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rank, group in enumerate(selected):
            for row in group["rows"]:
                out = dict(row)
                out["low_iou_group_rank"] = rank
                out["low_iou_positive_negative_iou"] = group["min_positive_negative_iou"]
                out["low_iou_exclusive_fraction"] = group["exclusive_fraction"]
                out["low_iou_wrong_sample_id"] = group["wrong_sample_id"]
                out["low_iou_wrong_source_type"] = group["wrong_source_type"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    selected_ious = [float(item["min_positive_negative_iou"]) for item in selected]
    selected_exclusive = [float(item["exclusive_fraction"]) for item in selected]
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "num_input_rows": len(rows),
        "num_candidate_groups": len(ranked),
        "num_eligible_groups": len(eligible),
        "num_selected_groups": len(selected),
        "num_selected_rows": sum(len(item["rows"]) for item in selected),
        "selection": {
            "max_groups": args.max_groups,
            "start_rank": args.start_rank,
            "max_iou": args.max_iou,
            "mask_size": args.mask_size,
            "path_mask_width": args.path_mask_width,
            "path_trajectory_mode": args.path_trajectory_mode,
            "path_projection_mode": args.path_projection_mode,
            "path_forward_m": args.path_forward_m,
            "path_lateral_m": args.path_lateral_m,
        },
        "selected_iou": {
            "mean": safe_mean(selected_ious),
            "min": min(selected_ious) if selected_ious else None,
            "max": max(selected_ious) if selected_ious else None,
            "p25": float(np.percentile(selected_ious, 25)) if selected_ious else None,
            "p50": float(np.percentile(selected_ious, 50)) if selected_ious else None,
            "p75": float(np.percentile(selected_ious, 75)) if selected_ious else None,
        },
        "selected_exclusive_fraction": {
            "mean": safe_mean(selected_exclusive),
            "min": min(selected_exclusive) if selected_exclusive else None,
            "max": max(selected_exclusive) if selected_exclusive else None,
        },
        "top_groups": [
            {
                key: item[key]
                for key in (
                    "group_id",
                    "positive_sample_id",
                    "wrong_sample_id",
                    "wrong_source_type",
                    "min_positive_negative_iou",
                    "exclusive_fraction",
                    "group_size",
                )
            }
            for item in selected[:20]
        ],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

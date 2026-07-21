#!/usr/bin/env python3
"""Add PDMS-style candidate-quality scores to IAC JSONL indices.

This is a conservative bridge until official NAVSIM PDMS/EPDMS fields are
available in the index. The generated score estimates whether a candidate path
is dynamically reasonable and close enough to the scene's GT maneuver to be a
soft positive. It deliberately leaves image/time/trajectory swaps as NaN so
the consistency model does not learn visual mismatches as positives.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


ALLOWED_SOURCES = {
    "gt_pos",
    "perturb_speed",
    "perturb_lateral",
    "perturb_heading",
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def _group_id(row: Dict[str, Any]) -> str:
    value = row.get("group_id") or row.get("anchor_id")
    if value is not None:
        return str(value)
    sample_id = str(row.get("sample_id", ""))
    if "__" in sample_id:
        return sample_id.rsplit("__", 1)[0]
    return sample_id


def _source(row: Dict[str, Any]) -> str:
    return str(row.get("source_type") or row.get("sample_type") or "unknown")


def _traj(row: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(row.get("candidate_traj") or [], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.shape[1] < 2:
        pad = np.zeros((arr.shape[0], 2 - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr[:, :2]


def _safe_exp_penalty(value: float, scale: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(math.exp(-max(value, 0.0) / max(scale, 1e-6)))


def _traj_metrics(path: np.ndarray) -> Dict[str, float]:
    if path.shape[0] == 0:
        return {
            "progress": 0.0,
            "lateral": 0.0,
            "length": 0.0,
            "directness": 0.0,
            "mean_step": 0.0,
            "max_step": 0.0,
            "jerk": 0.0,
            "yaw_total": 0.0,
            "max_yaw_step": 0.0,
        }

    deltas = np.diff(path, axis=0, prepend=np.zeros((1, 2), dtype=np.float32))
    steps = np.linalg.norm(deltas, axis=1)
    length = float(np.sum(steps))
    end = path[-1]
    progress = float(end[0])
    lateral = float(abs(end[1]))
    direct = float(np.linalg.norm(end))
    directness = direct / max(length, 1e-6)

    acc = np.diff(deltas, axis=0)
    jerk = float(np.mean(np.linalg.norm(acc, axis=1))) if len(acc) else 0.0

    headings = np.arctan2(deltas[:, 1], np.maximum(deltas[:, 0], 1e-6))
    headings = headings[np.isfinite(headings)]
    if len(headings) >= 2:
        yaw_steps = np.diff(np.unwrap(headings))
        yaw_total = float(np.sum(np.abs(yaw_steps)))
        max_yaw_step = float(np.max(np.abs(yaw_steps)))
    else:
        yaw_total = 0.0
        max_yaw_step = 0.0

    return {
        "progress": progress,
        "lateral": lateral,
        "length": length,
        "directness": directness,
        "mean_step": float(np.mean(steps)) if len(steps) else 0.0,
        "max_step": float(np.max(steps)) if len(steps) else 0.0,
        "jerk": jerk,
        "yaw_total": yaw_total,
        "max_yaw_step": max_yaw_step,
    }


def _ade(candidate: np.ndarray, reference: np.ndarray) -> float:
    steps = min(len(candidate), len(reference))
    if steps == 0:
        return float("inf")
    return float(np.linalg.norm(candidate[:steps] - reference[:steps], axis=1).mean())


def _fde(candidate: np.ndarray, reference: np.ndarray) -> float:
    steps = min(len(candidate), len(reference))
    if steps == 0:
        return float("inf")
    return float(np.linalg.norm(candidate[steps - 1] - reference[steps - 1]))


def _score_candidate(
    row: Dict[str, Any],
    gt_path: np.ndarray | None,
    min_score: float,
    gt_floor: float,
) -> Dict[str, float | None]:
    source = _source(row)
    if source not in ALLOWED_SOURCES:
        return {
            "pdms_proxy_score": None,
            "epdms_proxy_score": None,
            "candidate_quality_score": None,
        }

    path = _traj(row)
    metrics = _traj_metrics(path)
    validity = float(row.get("validity_label", 0.0))
    validity = 1.0 if validity >= 0.5 else 0.0

    smooth = _safe_exp_penalty(metrics["jerk"], 1.2)
    step_ok = _safe_exp_penalty(max(metrics["max_step"] - 7.0, 0.0), 2.0)
    yaw_ok = _safe_exp_penalty(max(metrics["max_yaw_step"] - 1.0, 0.0), 0.6)
    lateral_ok = _safe_exp_penalty(max(metrics["lateral"] - 8.0, 0.0), 4.0)
    forward_ok = 1.0 / (1.0 + math.exp(-(metrics["progress"] + 1.0) / 2.0))
    directness_ok = min(max(metrics["directness"], 0.0), 1.0)

    comfort = (
        0.28 * smooth
        + 0.20 * step_ok
        + 0.18 * yaw_ok
        + 0.14 * lateral_ok
        + 0.12 * forward_ok
        + 0.08 * directness_ok
    )

    if gt_path is not None and len(gt_path) > 0:
        candidate_ade = _ade(path, gt_path)
        candidate_fde = _fde(path, gt_path)
        closeness = 0.62 * _safe_exp_penalty(candidate_ade, 3.0)
        closeness += 0.38 * _safe_exp_penalty(candidate_fde, 5.0)
    else:
        candidate_ade = float("nan")
        candidate_fde = float("nan")
        closeness = 0.60 if source == "gt_pos" else 0.40

    pdms = validity * (0.62 * comfort + 0.38 * closeness)
    if validity < 0.5:
        pdms = min_score
    elif source == "gt_pos":
        pdms = max(pdms, gt_floor)

    # EPDMS is a little more permissive: it represents "expected" planning
    # quality under visual ambiguity rather than exact GT recovery.
    epdms = validity * (0.46 * comfort + 0.54 * max(closeness, 0.50))
    if validity < 0.5:
        epdms = min_score
    elif source == "gt_pos":
        epdms = max(epdms, gt_floor)

    pdms = float(min(max(pdms, 0.0), 1.0))
    epdms = float(min(max(epdms, 0.0), 1.0))
    quality = float(min(max(0.35 * pdms + 0.65 * epdms, 0.0), 1.0))

    return {
        "pdms_proxy_score": pdms,
        "epdms_proxy_score": epdms,
        "candidate_quality_score": quality,
        "pdms_proxy_ade_to_gt": candidate_ade,
        "pdms_proxy_fde_to_gt": candidate_fde,
        "pdms_proxy_progress": metrics["progress"],
        "pdms_proxy_lateral": metrics["lateral"],
        "pdms_proxy_jerk": metrics["jerk"],
    }


def enrich_rows(
    rows: Sequence[Dict[str, Any]],
    min_score: float,
    gt_floor: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_id(row)].append(row)

    gt_by_group: Dict[str, np.ndarray] = {}
    for gid, items in groups.items():
        gt_rows = [
            row
            for row in items
            if _source(row) == "gt_pos"
            or float(row.get("consistency_label", 0.0)) >= 0.5
        ]
        if gt_rows:
            gt_by_group[gid] = _traj(gt_rows[0])

    enriched: List[Dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    scored_counts: Counter[str] = Counter()
    scores_by_source: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        out = dict(row)
        source = _source(out)
        source_counts[source] += 1
        scores = _score_candidate(
            out,
            gt_by_group.get(_group_id(out)),
            min_score=min_score,
            gt_floor=gt_floor,
        )
        for key, value in scores.items():
            out[key] = value
        quality = out.get("candidate_quality_score")
        if isinstance(quality, (int, float)) and math.isfinite(float(quality)):
            scored_counts[source] += 1
            scores_by_source[source].append(float(quality))
        enriched.append(out)

    summary = {
        "rows": len(rows),
        "groups": len(groups),
        "groups_with_gt": len(gt_by_group),
        "allowed_sources": sorted(ALLOWED_SOURCES),
        "source_counts": dict(source_counts),
        "scored_counts": dict(scored_counts),
        "mean_candidate_quality_by_source": {
            source: mean(values)
            for source, values in sorted(scores_by_source.items())
            if values
        },
    }
    return enriched, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add conservative PDMS/EPDMS proxy scores to IAC JSONL."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--gt-floor", type=float, default=0.88)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_jsonl(args.input)
    enriched, summary = enrich_rows(
        rows,
        min_score=args.min_score,
        gt_floor=args.gt_floor,
    )
    _write_jsonl(args.output, enriched)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

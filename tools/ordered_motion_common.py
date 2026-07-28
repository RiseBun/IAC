"""Shared I/O and metrics for ordered motion alignment tools."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_ACCEPTABLE_SOURCES = (
    "gt_pos,perturb_speed,perturb_lateral,perturb_heading"
)
DEFAULT_HARD_SOURCES = (
    "image_swap,time_shift_future,traj_swap,high_pdm_image_mismatch,"
    "reverse_traj,video_tempo_x1p50,video_tempo_x2p00"
)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(row: Mapping[str, Any]) -> str:
    for key in (
        "source_type",
        "action_type",
        "wam_name",
        "sample_type",
        "wam",
        "source",
    ):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def group_id(row: Mapping[str, Any]) -> str:
    for key in ("group_id", "anchor_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    sample_id = str(row.get("sample_id", ""))
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def is_gt(row: Mapping[str, Any]) -> bool:
    return source(row) == "gt_pos"


def is_positive(row: Mapping[str, Any]) -> bool:
    if row.get("consistency_label") is not None:
        return float(row["consistency_label"]) > 0.5
    if row.get("label") is not None:
        return float(row["label"]) > 0.5
    return is_gt(row)


def split_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def sattolo_indices(length: int, seed: int) -> List[int]:
    """Return a deterministic permutation with no fixed point for n > 1."""

    values = list(range(length))
    if length < 2:
        return values
    rng = random.Random(seed)
    for index in range(length - 1, 0, -1):
        other = rng.randrange(index)
        values[index], values[other] = values[other], values[index]
    return values


def ranking_metrics(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    acceptable_sources: set[str],
    hard_sources: set[str],
) -> Dict[str, Any]:
    if len(rows) != len(scores):
        raise ValueError("rows and scores must have equal length")
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_id(row)].append(index)

    strict_hits = 0
    acceptable_hits = 0
    hard_hits = 0
    reciprocal_ranks: List[float] = []
    top_sources: Dict[str, int] = defaultdict(int)
    pair_wins: Dict[str, List[float]] = defaultdict(list)
    source_values: Dict[str, List[float]] = defaultdict(list)

    used_groups = 0
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        gt_indices = [index for index in indices if is_gt(rows[index])]
        if not gt_indices:
            continue
        used_groups += 1
        ordered = sorted(
            indices,
            key=lambda index: (
                -float(scores[index]),
                str(rows[index].get("sample_id", "")),
            ),
        )
        top = ordered[0]
        top_source = source(rows[top])
        top_sources[top_source] += 1
        strict_hits += int(is_gt(rows[top]))
        acceptable_hits += int(
            is_gt(rows[top]) or top_source in acceptable_sources
        )
        hard_hits += int(top_source in hard_sources)

        gt_rank = min(ordered.index(index) + 1 for index in gt_indices)
        reciprocal_ranks.append(1.0 / float(gt_rank))
        gt_score = max(float(scores[index]) for index in gt_indices)
        for index in indices:
            current_source = source(rows[index])
            source_values[current_source].append(float(scores[index]))
            if index in gt_indices:
                continue
            other_score = float(scores[index])
            pair_wins[current_source].append(
                1.0 if gt_score > other_score else 0.5 if gt_score == other_score else 0.0
            )

    denominator = max(used_groups, 1)
    return {
        "num_rows": len(rows),
        "num_groups": used_groups,
        "strict_gt_top1": strict_hits / denominator,
        "acceptable_top1": acceptable_hits / denominator,
        "hard_mismatch_top1": hard_hits / denominator,
        "mrr_gt": (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks
            else None
        ),
        "top_sources": dict(sorted(top_sources.items())),
        "pairwise_gt_win": {
            key: sum(values) / len(values)
            for key, values in sorted(pair_wins.items())
            if values
        },
        "source_score_mean": {
            key: sum(values) / len(values)
            for key, values in sorted(source_values.items())
            if values
        },
    }


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default

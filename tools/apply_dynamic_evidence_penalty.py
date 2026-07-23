#!/usr/bin/env python3
"""Apply a one-sided dynamic visual evidence penalty to IAC score rows.

``eval_dynamic_evidence_fusion.py`` emits a group-normalized dynamic energy:
lower is better, higher means the candidate trajectory is less supported by
RGB-diff + optical-flow motion evidence. This tool does not reward low energy;
it only subtracts a conservative logit-space penalty when energy exceeds a
threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_HARD_SOURCES = {
    "image_swap",
    "time_shift_future",
    "traj_swap",
    "reverse_traj",
    "high_pdm_image_mismatch",
}
DEFAULT_NEAR_SOURCES = {"perturb_speed", "perturb_lateral", "perturb_heading"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _source(row: Mapping[str, Any], wam_key: str = "wam_name") -> str:
    for key in ("source_type", "action_type", wam_key, "sample_type", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _group_id(row: Mapping[str, Any], group_key: str = "group_id") -> str:
    value = row.get(group_key) or row.get("anchor_id") or row.get("group_id")
    if value is not None:
        return str(value)
    sample_id = str(row.get("sample_id", ""))
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def _is_positive(row: Mapping[str, Any], wam_key: str = "wam_name") -> bool:
    if row.get("consistency_label") is not None:
        return float(row["consistency_label"]) > 0.5
    if row.get("label") is not None:
        return float(row["label"]) > 0.5
    return _source(row, wam_key) == "gt_pos"


def _score_fields(rows: Iterable[Mapping[str, Any]]) -> List[str]:
    fields = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith("iac_consistency") and isinstance(value, (int, float)):
                fields.add(key)
    return sorted(fields)


def _dynamic_key(row: Mapping[str, Any], group_key: str = "group_id") -> Tuple[str, str]:
    return (_group_id(row, group_key), str(row.get("sample_id")))


def _align_dynamic_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    dynamic_rows: Sequence[Mapping[str, Any]],
    group_key: str,
) -> List[Mapping[str, Any]]:
    dynamic_by_key = {_dynamic_key(row, group_key): row for row in dynamic_rows}
    aligned = []
    missing = []
    for row in primary_rows:
        key = _dynamic_key(row, group_key)
        dynamic = dynamic_by_key.get(key)
        if dynamic is None:
            missing.append(key)
        else:
            aligned.append(dynamic)
    if missing:
        example = ", ".join(f"{gid}/{sid}" for gid, sid in missing[:3])
        raise ValueError(f"missing {len(missing)} dynamic rows, examples: {example}")
    return aligned


def _apply_penalty(
    primary_rows: Sequence[Mapping[str, Any]],
    dynamic_rows: Sequence[Mapping[str, Any]],
    *,
    beta: float,
    threshold: float,
    energy_key: str,
    label: str,
) -> List[Dict[str, Any]]:
    fields = _score_fields(primary_rows)
    if not fields:
        raise ValueError("no numeric iac_consistency* fields found")
    out = []
    for primary, dynamic in zip(primary_rows, dynamic_rows):
        energy = float(dynamic[energy_key])
        penalty = float(beta) * max(0.0, energy - float(threshold))
        row = dict(primary)
        for field in fields:
            row[field] = _sigmoid(_logit(float(primary[field])) - penalty)
        row["dynamic_fused_energy"] = energy
        row["dynamic_penalty"] = penalty
        row["score_fusion_label"] = label
        row["score_fusion_aux"] = {
            "dynamic_energy_key": energy_key,
            "beta": float(beta),
            "threshold": float(threshold),
        }
        out.append(row)
    return out


def _audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    wam_key: str,
    close_margin: float,
    near_sources: set[str],
    hard_sources: set[str],
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_id(row, group_key)].append(row)

    hit = 0
    ambiguous = 0
    hard_above = 0
    total = 0
    top_sources: Counter[str] = Counter()
    for items in groups.values():
        positives = [row for row in items if _is_positive(row, wam_key)]
        if not positives:
            continue
        gt = positives[0]
        total += 1
        ranked = sorted(items, key=lambda row: float(row["iac_consistency"]), reverse=True)
        winner = ranked[0]
        winner_source = _source(winner, wam_key)
        top_sources[winner_source] += 1
        if winner is gt:
            hit += 1
        else:
            gap = float(winner["iac_consistency"]) - float(gt["iac_consistency"])
            if winner_source in near_sources and gap <= close_margin:
                ambiguous += 1
        gt_score = float(gt["iac_consistency"])
        if any(
            _source(row, wam_key) in hard_sources and float(row["iac_consistency"]) > gt_score
            for row in items
            if row is not gt
        ):
            hard_above += 1

    return {
        "groups": total,
        "hard_top1": hit / total if total else None,
        "ambiguity_adjusted_top1": (hit + ambiguous) / total if total else None,
        "hard_mismatch_above_gt_by_score": hard_above / total if total else None,
        "top_sources": top_sources.most_common(),
    }


def _parse_float_list(text: str, default: Sequence[float]) -> List[float]:
    if not text:
        return [float(value) for value in default]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_sources(text: str, default: set[str]) -> set[str]:
    if not text:
        return set(default)
    return {item.strip() for item in text.split(",") if item.strip()}


def _best_key(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary.get("ambiguity_adjusted_top1") or -1.0),
        float(summary.get("hard_top1") or -1.0),
        -float(summary.get("hard_mismatch_above_gt_by_score") or 1.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-scores", required=True)
    parser.add_argument("--dynamic-rows", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--energy-key", default="fused_dynamic_energy")
    parser.add_argument("--label", default="dynamic_evidence_penalty")
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--close-margin", type=float, default=0.02)
    parser.add_argument("--near-sources", default="")
    parser.add_argument("--hard-sources", default="")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--beta-grid", default="")
    parser.add_argument("--threshold-grid", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_rows = _load_jsonl(Path(args.primary_scores))
    dynamic_rows = _align_dynamic_rows(
        primary_rows,
        _load_jsonl(Path(args.dynamic_rows)),
        args.group_key,
    )
    near_sources = _parse_sources(args.near_sources, DEFAULT_NEAR_SOURCES)
    hard_sources = _parse_sources(args.hard_sources, DEFAULT_HARD_SOURCES)
    original = _audit(
        primary_rows,
        group_key=args.group_key,
        wam_key=args.wam_key,
        close_margin=args.close_margin,
        near_sources=near_sources,
        hard_sources=hard_sources,
    )

    if args.sweep:
        betas = _parse_float_list(args.beta_grid, (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0))
        thresholds = _parse_float_list(args.threshold_grid, (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25))
        grid = []
        best = None
        best_rows: List[Dict[str, Any]] = []
        for threshold in thresholds:
            for beta in betas:
                rows = _apply_penalty(
                    primary_rows,
                    dynamic_rows,
                    beta=beta,
                    threshold=threshold,
                    energy_key=args.energy_key,
                    label=args.label,
                )
                summary = _audit(
                    rows,
                    group_key=args.group_key,
                    wam_key=args.wam_key,
                    close_margin=args.close_margin,
                    near_sources=near_sources,
                    hard_sources=hard_sources,
                )
                item = {
                    "threshold": threshold,
                    "beta": beta,
                    **summary,
                }
                grid.append(item)
                key = _best_key(summary)
                if best is None or key > best[0]:
                    best = (key, item)
                    best_rows = rows
        final_rows = best_rows
        final = best[1] if best else None
    else:
        final_rows = _apply_penalty(
            primary_rows,
            dynamic_rows,
            beta=args.beta,
            threshold=args.threshold,
            energy_key=args.energy_key,
            label=args.label,
        )
        final = {
            "threshold": args.threshold,
            "beta": args.beta,
            **_audit(
                final_rows,
                group_key=args.group_key,
                wam_key=args.wam_key,
                close_margin=args.close_margin,
                near_sources=near_sources,
                hard_sources=hard_sources,
            ),
        }
        grid = []

    _write_jsonl(Path(args.output_scores), final_rows)
    report = {
        "original": original,
        "final": final,
        "grid": grid,
        "config": {
            "primary_scores": args.primary_scores,
            "dynamic_rows": args.dynamic_rows,
            "energy_key": args.energy_key,
            "label": args.label,
            "group_key": args.group_key,
            "wam_key": args.wam_key,
            "close_margin": args.close_margin,
            "near_sources": sorted(near_sources),
            "hard_sources": sorted(hard_sources),
            "sweep": bool(args.sweep),
        },
    }
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Score IAC groups with multi-solution acceptance and confidence.

The ordinary top-1 metric assumes a single correct candidate. For IAC this is
too strict: small speed/lateral/heading perturbations can be visually
equivalent to the GT future. This tool evaluates grouped score JSONLs under a
multi-solution interpretation and emits a group-level verdict:

    match      best acceptable candidate beats hard mismatches with margin
    mismatch   hard mismatch beats all acceptable candidates with margin
    ambiguous  margin is small, so the video/action pair is not decisive

Optional auxiliary score files are fused in logit space before scoring.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _group_id(row: Dict[str, Any], group_key: str) -> str:
    return str(row.get(group_key) or row.get("anchor_id") or row.get("sample_id"))


def _source(row: Dict[str, Any], source_key: str) -> str:
    for key in (source_key, "source_type", "action_type", "wam_name", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _score_fields(rows: Iterable[Dict[str, Any]]) -> set[str]:
    fields: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith("iac_consistency") and isinstance(value, (int, float)):
                fields.add(key)
    return fields


def _fuse_rows(
    primary_rows: Sequence[Dict[str, Any]],
    aux_specs: Sequence[tuple[Path, float, Sequence[Dict[str, Any]]]],
    group_key: str,
) -> List[Dict[str, Any]]:
    fields = _score_fields(primary_rows)
    for _, _, rows in aux_specs:
        fields &= _score_fields(rows)
    if "iac_consistency" not in fields:
        raise ValueError("no shared iac_consistency score field")

    fused: List[Dict[str, Any]] = []
    for idx, primary in enumerate(primary_rows):
        group = _group_id(primary, group_key)
        sample_id = primary.get("sample_id")
        row = dict(primary)
        for path, _, rows in aux_specs:
            if idx >= len(rows):
                raise ValueError(f"{path} has fewer rows than primary")
            other = rows[idx]
            if _group_id(other, group_key) != group:
                raise ValueError(
                    f"row {idx} group mismatch: {group!r} vs "
                    f"{_group_id(other, group_key)!r}"
                )
            if sample_id is not None and other.get("sample_id") != sample_id:
                raise ValueError(
                    f"row {idx} sample mismatch: {sample_id!r} vs "
                    f"{other.get('sample_id')!r}"
                )
        for field in fields:
            value = _logit(primary[field])
            for _, weight, rows in aux_specs:
                value += float(weight) * _logit(rows[idx][field])
            row[field] = _sigmoid(value)
        fused.append(row)
    return fused


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _score_groups(
    rows: Sequence[Dict[str, Any]],
    *,
    group_key: str,
    source_key: str,
    score_key: str,
    margin_space: str,
    acceptable_sources: set[str],
    hard_sources: set[str],
    match_margin: float,
    mismatch_margin: float,
    confidence_temperature: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_id(row, group_key)].append(row)

    records: List[Dict[str, Any]] = []
    verdict_counts: Counter[str] = Counter()
    top_sources: Counter[str] = Counter()
    bad_top_sources: Counter[str] = Counter()
    margins: List[float] = []
    confidences: List[float] = []
    acceptable_hits = 0
    strict_hits = 0
    hard_mismatch_top = 0

    for group_id, items in grouped.items():
        if len(items) < 2:
            continue
        ranked = sorted(items, key=lambda row: float(row[score_key]), reverse=True)
        top = ranked[0]
        top_source = _source(top, source_key)
        top_sources[top_source] += 1

        acceptable = [
            row for row in items
            if _source(row, source_key) in acceptable_sources
        ]
        hard = [
            row for row in items
            if _source(row, source_key) in hard_sources
        ]
        non_acceptable = [
            row for row in items
            if _source(row, source_key) not in acceptable_sources
        ]
        if not acceptable or not non_acceptable:
            continue

        best_accept = max(acceptable, key=lambda row: float(row[score_key]))
        best_bad_pool = hard if hard else non_acceptable
        best_bad = max(best_bad_pool, key=lambda row: float(row[score_key]))
        if margin_space == "logit":
            accept_margin = _logit(float(best_accept[score_key])) - _logit(
                float(best_bad[score_key])
            )
        elif margin_space == "raw":
            accept_margin = float(best_accept[score_key]) - float(best_bad[score_key])
        else:
            raise ValueError(f"unknown margin space: {margin_space}")
        decision_confidence = _sigmoid(
            abs(accept_margin) / max(float(confidence_temperature), 1e-6)
        )
        match_confidence = _sigmoid(
            accept_margin / max(float(confidence_temperature), 1e-6)
        )

        if accept_margin >= match_margin:
            verdict = "match"
        elif accept_margin <= mismatch_margin:
            verdict = "mismatch"
        else:
            verdict = "ambiguous"

        if top_source in acceptable_sources:
            acceptable_hits += 1
        else:
            bad_top_sources[top_source] += 1
        if top_source == "gt_pos":
            strict_hits += 1
        if top_source in hard_sources:
            hard_mismatch_top += 1
        verdict_counts[verdict] += 1
        margins.append(accept_margin)
        confidences.append(decision_confidence)

        accept_sorted = sorted(
            acceptable,
            key=lambda row: float(row[score_key]),
            reverse=True,
        )
        within_accept_gap = None
        if len(accept_sorted) > 1:
            if margin_space == "logit":
                within_accept_gap = _logit(float(accept_sorted[0][score_key])) - _logit(
                    float(accept_sorted[1][score_key])
                )
            else:
                within_accept_gap = float(accept_sorted[0][score_key]) - float(
                    accept_sorted[1][score_key]
                )

        records.append(
            {
                "group_id": group_id,
                "verdict": verdict,
                "decision_confidence": decision_confidence,
                "match_confidence": match_confidence,
                "accept_margin_logit": accept_margin,
                "top_source": top_source,
                "top_sample_id": top.get("sample_id"),
                "top_score": top.get(score_key),
                "best_accept_source": _source(best_accept, source_key),
                "best_accept_sample_id": best_accept.get("sample_id"),
                "best_accept_score": best_accept.get(score_key),
                "best_bad_source": _source(best_bad, source_key),
                "best_bad_sample_id": best_bad.get("sample_id"),
                "best_bad_score": best_bad.get(score_key),
                "within_acceptable_gap_logit": within_accept_gap,
                "num_candidates": len(items),
                "num_acceptable": len(acceptable),
                "num_hard": len(hard),
            }
        )

    n = len(records)
    summary = {
        "num_groups": n,
        "strict_gt_top1": strict_hits / n if n else None,
        "acceptable_top1": acceptable_hits / n if n else None,
        "hard_mismatch_top1": hard_mismatch_top / n if n else None,
        "verdict_counts": dict(verdict_counts),
        "verdict_fractions": {
            key: value / n if n else None
            for key, value in sorted(verdict_counts.items())
        },
        "top_sources": top_sources.most_common(),
        "bad_top_sources": bad_top_sources.most_common(),
        "accept_margin_logit": {
            "mean": _mean(margins),
            "median": _quantile(margins, 0.5),
            "p25": _quantile(margins, 0.25),
            "p75": _quantile(margins, 0.75),
        },
        "decision_confidence": {
            "mean": _mean(confidences),
            "median": _quantile(confidences, 0.5),
            "p25": _quantile(confidences, 0.25),
            "p75": _quantile(confidences, 0.75),
        },
        "config": {
            "acceptable_sources": sorted(acceptable_sources),
            "hard_sources": sorted(hard_sources),
            "score_key": score_key,
            "margin_space": margin_space,
            "match_margin": match_margin,
            "mismatch_margin": mismatch_margin,
            "confidence_temperature": confidence_temperature,
        },
    }
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-scores", required=True)
    parser.add_argument(
        "--aux",
        action="append",
        default=[],
        metavar="PATH:WEIGHT",
        help="Optional auxiliary score JSONL plus logit-space weight.",
    )
    parser.add_argument("--output-groups", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-scores", default=None)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--source-key", default="source_type")
    parser.add_argument("--score-key", default="iac_consistency")
    parser.add_argument(
        "--margin-space",
        choices=["logit", "raw"],
        default="logit",
        help="Use logit margins for probability scores, or raw margins for unbounded rank scores.",
    )
    parser.add_argument(
        "--acceptable-sources",
        default="gt_pos,perturb_speed,perturb_lateral,perturb_heading",
    )
    parser.add_argument(
        "--hard-sources",
        default="image_swap,time_shift_future,traj_swap,reverse_traj,high_pdm_image_mismatch",
    )
    parser.add_argument("--match-margin", type=float, default=0.50)
    parser.add_argument("--mismatch-margin", type=float, default=-0.50)
    parser.add_argument("--confidence-temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_rows = _load_rows(Path(args.primary_scores))
    aux_specs: List[tuple[Path, float, Sequence[Dict[str, Any]]]] = []
    for raw in args.aux:
        path_raw, sep, weight_raw = raw.rpartition(":")
        if not sep:
            raise SystemExit(f"--aux must be PATH:WEIGHT, got {raw!r}")
        path = Path(path_raw)
        aux_specs.append((path, float(weight_raw), _load_rows(path)))

    rows = _fuse_rows(primary_rows, aux_specs, args.group_key) if aux_specs else primary_rows
    acceptable_sources = {
        item.strip() for item in args.acceptable_sources.split(",") if item.strip()
    }
    hard_sources = {
        item.strip() for item in args.hard_sources.split(",") if item.strip()
    }
    records, summary = _score_groups(
        rows,
        group_key=args.group_key,
        source_key=args.source_key,
        score_key=str(args.score_key),
        margin_space=str(args.margin_space),
        acceptable_sources=acceptable_sources,
        hard_sources=hard_sources,
        match_margin=float(args.match_margin),
        mismatch_margin=float(args.mismatch_margin),
        confidence_temperature=float(args.confidence_temperature),
    )

    out_groups = Path(args.output_groups)
    out_groups.parent.mkdir(parents=True, exist_ok=True)
    with out_groups.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_summary = Path(args.output_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.output_scores:
        out_scores = Path(args.output_scores)
        out_scores.parent.mkdir(parents=True, exist_ok=True)
        with out_scores.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

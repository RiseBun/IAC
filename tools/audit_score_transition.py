#!/usr/bin/env python3
"""Audit group-level ranking changes between two IAC score files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


NEAR_SOURCES = {"perturb_speed", "perturb_lateral", "perturb_heading"}
HARD_SOURCES = {
    "image_swap",
    "time_shift",
    "time_shift_future",
    "time_shift_past",
    "traj_swap",
    "reverse_traj",
    "high_pdm_image_mismatch",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source(row: Mapping[str, Any], wam_key: str) -> str:
    for key in ("source_type", "action_type", wam_key, "wam_name", "sample_type", "wam"):
        if row.get(key) is not None:
            return str(row[key])
    return "unknown"


def _group_id(row: Mapping[str, Any], group_key: str) -> str:
    value = row.get(group_key) or row.get("anchor_id") or row.get("group_id")
    if value is not None:
        return str(value)
    sample_id = str(row.get("sample_id", ""))
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def _is_positive(row: Mapping[str, Any], wam_key: str) -> bool:
    if row.get("consistency_label") is not None:
        return float(row["consistency_label"]) > 0.5
    if row.get("label") is not None:
        return float(row["label"]) > 0.5
    return _source(row, wam_key) == "gt_pos"


def _score(row: Mapping[str, Any], score_key: str) -> float:
    return float(row[score_key])


def _index_groups(rows: Sequence[Mapping[str, Any]], group_key: str) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    groups: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        gid = _group_id(row, group_key)
        sid = str(row.get("sample_id"))
        groups[gid][sid] = row
    return groups


def _state(
    items_by_sid: Mapping[str, Mapping[str, Any]],
    *,
    score_key: str,
    wam_key: str,
    close_margin: float,
) -> Dict[str, Any] | None:
    items = list(items_by_sid.values())
    positives = [row for row in items if _is_positive(row, wam_key)]
    if not positives:
        return None
    gt = positives[0]
    gt_score = _score(gt, score_key)
    ranked = sorted(items, key=lambda row: _score(row, score_key), reverse=True)
    winner = ranked[0]
    winner_source = _source(winner, wam_key)
    winner_score = _score(winner, score_key)
    hard_above = [
        row for row in items
        if row is not gt and _source(row, wam_key) in HARD_SOURCES and _score(row, score_key) > gt_score
    ]
    accepted = winner is gt or (
        winner_source in NEAR_SOURCES and winner_score - gt_score <= close_margin
    )
    return {
        "accepted": bool(accepted),
        "hard_hit": bool(winner is gt),
        "gt_score": gt_score,
        "gt_sample_id": str(gt.get("sample_id")),
        "winner_score": winner_score,
        "winner_source": winner_source,
        "winner_sample_id": str(winner.get("sample_id")),
        "winner_gap": winner_score - gt_score,
        "hard_above": bool(hard_above),
        "hard_above_sources": sorted({_source(row, wam_key) for row in hard_above}),
        "hard_above_count": len(hard_above),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-groups", required=True)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--score-key", default="iac_consistency")
    parser.add_argument("--close-margin", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before_groups = _index_groups(_load_jsonl(Path(args.before)), args.group_key)
    after_groups = _index_groups(_load_jsonl(Path(args.after)), args.group_key)
    records: List[Dict[str, Any]] = []
    counters: Counter[str] = Counter()
    top_transitions: Counter[str] = Counter()

    for gid in sorted(set(before_groups) & set(after_groups)):
        before = _state(
            before_groups[gid],
            score_key=args.score_key,
            wam_key=args.wam_key,
            close_margin=float(args.close_margin),
        )
        after = _state(
            after_groups[gid],
            score_key=args.score_key,
            wam_key=args.wam_key,
            close_margin=float(args.close_margin),
        )
        if before is None or after is None:
            continue
        if before["accepted"]:
            counters["before_accepted"] += 1
        if after["accepted"]:
            counters["after_accepted"] += 1
        if before["hard_hit"]:
            counters["before_hard_hit"] += 1
        if after["hard_hit"]:
            counters["after_hard_hit"] += 1
        if before["hard_above"]:
            counters["before_hard_above"] += 1
        if after["hard_above"]:
            counters["after_hard_above"] += 1
        if before["accepted"] and not after["accepted"]:
            counters["accepted_lost"] += 1
        if not before["accepted"] and after["accepted"]:
            counters["accepted_gained"] += 1
        if before["hard_above"] and not after["hard_above"]:
            counters["hard_above_fixed"] += 1
        if not before["hard_above"] and after["hard_above"]:
            counters["hard_above_introduced"] += 1
        transition = f"{before['winner_source']}->{after['winner_source']}"
        top_transitions[transition] += 1
        records.append(
            {
                "group_id": gid,
                "transition": transition,
                "before": before,
                "after": after,
                "accepted_lost": bool(before["accepted"] and not after["accepted"]),
                "accepted_gained": bool((not before["accepted"]) and after["accepted"]),
                "hard_above_fixed": bool(before["hard_above"] and not after["hard_above"]),
                "hard_above_introduced": bool((not before["hard_above"]) and after["hard_above"]),
            }
        )

    n = len(records)
    summary = {
        "before": args.before,
        "after": args.after,
        "groups": n,
        "rates": {key: value / n if n else None for key, value in sorted(counters.items())},
        "counts": dict(sorted(counters.items())),
        "top_transitions": top_transitions.most_common(20),
    }
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(Path(args.output_groups), records)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

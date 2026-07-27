#!/usr/bin/env python3
"""Fuse v3 acceptability scores with the clean V-JEPA trajectory gate.

The gate is used only as a conservative penalty: within each candidate group,
rows that fall below the group's best clean-gate logit are downweighted. This
keeps v3 as the main ranker while giving visual/action evidence a way to veto
hard mismatches.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _group_id(row: Dict[str, Any], group_key: str) -> str:
    return str(row.get(group_key) or row.get("anchor_id") or row.get("sample_id"))


def _source(row: Dict[str, Any], source_key: str) -> str:
    for key in (source_key, "source_type", "action_type", "wam_name", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _key(row: Dict[str, Any], group_key: str) -> tuple[str, str]:
    return (_group_id(row, group_key), str(row.get("sample_id")))


def _summary(
    rows: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    group_key: str,
    source_key: str,
    acceptable_sources: set[str],
    hard_sources: set[str],
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_id(row, group_key)].append(row)
    top_sources: Counter[str] = Counter()
    strict = 0
    acceptable = 0
    hard = 0
    n = 0
    for items in groups.values():
        if len(items) < 2:
            continue
        n += 1
        top = max(items, key=lambda row: float(row[score_key]))
        src = _source(top, source_key)
        top_sources[src] += 1
        strict += int(src == "gt_pos")
        acceptable += int(src in acceptable_sources)
        hard += int(src in hard_sources)
    return {
        "num_groups": n,
        "strict_gt_top1": strict / n if n else None,
        "acceptable_top1": acceptable / n if n else None,
        "hard_mismatch_top1": hard / n if n else None,
        "top_sources": top_sources.most_common(),
    }


def _parse_sources(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-scores", required=True)
    parser.add_argument("--gate-scores", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--v3-score-key", default="iac_acceptability_calibrated")
    parser.add_argument("--gate-logit-key", default="visual_non_mismatch_logit")
    parser.add_argument("--output-score-key", default="v3_clean_gate_fused_rank_score")
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--source-key", default="source_type")
    parser.add_argument(
        "--acceptable-sources",
        default="gt_pos,perturb_speed,perturb_lateral,perturb_heading",
    )
    parser.add_argument(
        "--hard-sources",
        default="image_swap,time_shift_future,traj_swap,reverse_traj,high_pdm_image_mismatch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v3_rows = _load_jsonl(Path(args.v3_scores))
    gate_rows = _load_jsonl(Path(args.gate_scores))
    gate_by_key = {_key(row, args.group_key): row for row in gate_rows}
    group_max_gate: Dict[str, float] = defaultdict(lambda: float("-inf"))
    for row in gate_rows:
        gid = _group_id(row, args.group_key)
        group_max_gate[gid] = max(group_max_gate[gid], float(row[args.gate_logit_key]))

    fused: List[Dict[str, Any]] = []
    for row in v3_rows:
        key = _key(row, args.group_key)
        if key not in gate_by_key:
            raise KeyError(f"missing gate row for group/sample {key!r}")
        gate = gate_by_key[key]
        gid = key[0]
        base = float(row[args.v3_score_key])
        gate_logit = float(gate[args.gate_logit_key])
        penalty = max(0.0, group_max_gate[gid] - gate_logit - float(args.threshold))
        score = base - float(args.beta) * penalty
        item = dict(row)
        item["base_iac_consistency"] = row.get("iac_consistency")
        item["clean_vjepa_traj_gate_logit"] = gate_logit
        item["clean_vjepa_traj_gate_score"] = gate.get("visual_non_mismatch")
        item["clean_vjepa_traj_gate_penalty"] = penalty
        item[args.output_score_key] = score
        item["iac_consistency"] = score
        item["score_fusion_label"] = "v3_acceptability_plus_clean_vjepa_traj_gate"
        fused.append(item)

    acceptable_sources = _parse_sources(args.acceptable_sources)
    hard_sources = _parse_sources(args.hard_sources)
    summary = {
        "config": {
            "v3_scores": str(args.v3_scores),
            "gate_scores": str(args.gate_scores),
            "v3_score_key": args.v3_score_key,
            "gate_logit_key": args.gate_logit_key,
            "output_score_key": args.output_score_key,
            "beta": float(args.beta),
            "threshold": float(args.threshold),
        },
        "eval": _summary(
            fused,
            score_key=args.output_score_key,
            group_key=args.group_key,
            source_key=args.source_key,
            acceptable_sources=acceptable_sources,
            hard_sources=hard_sources,
        ),
    }
    _write_jsonl(Path(args.output_scores), fused)
    out_summary = Path(args.output_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["eval"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

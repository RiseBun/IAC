#!/usr/bin/env python3
"""Audit ordered-motion rankings against independent NAVSIM PDM labels.

PDM labels are joined by sample_id and are never used as model inputs.  The
audit is set-valued: among the same-scene physical candidates, every
trajectory within ``tolerance`` of the group's best official PDM score is
accepted.  This avoids treating the recorded trajectory as uniquely correct
when multiple trajectories are physically safe.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PHYSICAL_SOURCES = (
    "gt_pos",
    "perturb_speed",
    "perturb_lateral",
    "perturb_heading",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_id(row: Mapping[str, Any]) -> str:
    value = row.get("group_id") or row.get("anchor_id")
    if value is not None:
        return str(value)
    return str(row.get("sample_id", "")).rsplit("__", 1)[0]


def source(row: Mapping[str, Any]) -> str:
    for key in ("source_type", "action_type", "source"):
        if row.get(key) is not None:
            return str(row[key])
    return "unknown"


def parse_score_specs(values: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"score must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name or not path:
            raise ValueError(f"invalid score specification {value!r}")
        result[name] = Path(path)
    if not result:
        raise ValueError("at least one --scores NAME=PATH is required")
    return result


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_jsonl(Path(args.rows))
    official: Dict[str, Dict[str, Any]] = {}
    official_sources: Counter[str] = Counter()
    for path in args.official_rows:
        for row in load_jsonl(Path(path)):
            sample_id = str(row.get("sample_id", ""))
            if sample_id:
                official[sample_id] = row
                official_sources[source(row)] += 1

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_id(row)].append(row)

    physical_groups: Dict[str, Dict[str, Any]] = {}
    for current_group, candidates in sorted(grouped.items()):
        pdm_by_source: Dict[str, float] = {}
        for row in candidates:
            current_source = source(row)
            if current_source not in PHYSICAL_SOURCES:
                continue
            label = official.get(str(row.get("sample_id", "")), {})
            value = label.get(args.pdm_key)
            if value is None:
                continue
            try:
                pdm_by_source[current_source] = float(value)
            except (TypeError, ValueError):
                continue
        if not all(name in pdm_by_source for name in PHYSICAL_SOURCES):
            continue
        best = max(pdm_by_source.values())
        accepted = sorted(
            name
            for name, value in pdm_by_source.items()
            if value >= best - float(args.tolerance)
        )
        physical_groups[current_group] = {
            "pdm_by_source": pdm_by_source,
            "best_pdm": best,
            "accepted_sources": accepted,
        }

    score_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for name, path in parse_score_specs(args.scores).items():
        current: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in load_jsonl(path):
            current[group_id(row)].append(row)
        score_groups[name] = current

    ledger: List[Dict[str, Any]] = []
    model_values: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for current_group, physical in physical_groups.items():
        ledger_row: Dict[str, Any] = {
            "group_id": current_group,
            **physical,
        }
        for name, current_groups in score_groups.items():
            candidates = current_groups.get(current_group, [])
            if not candidates:
                continue
            top = max(
                candidates,
                key=lambda row: (
                    float(row.get("ordered_motion_rank_score", float("-inf"))),
                    str(row.get("sample_id", "")),
                ),
            )
            top_source = source(top)
            top_pdm = physical["pdm_by_source"].get(top_source)
            item = {
                "top_source": top_source,
                "top_pdm": top_pdm,
                "top_pdm_available": top_pdm is not None,
                "physics_set_hit": bool(
                    top_pdm is not None
                    and top_pdm >= physical["best_pdm"] - float(args.tolerance)
                ),
                "pdm_regret": (
                    physical["best_pdm"] - top_pdm
                    if top_pdm is not None
                    else None
                ),
            }
            ledger_row[name] = item
            model_values[name].append(item)
        ledger.append(ledger_row)

    model_metrics: Dict[str, Any] = {}
    for name, values in model_values.items():
        regrets = [item["pdm_regret"] for item in values if item["pdm_regret"] is not None]
        model_metrics[name] = {
            "groups_scored": len(values),
            "physics_set_top1": sum(item["physics_set_hit"] for item in values)
            / max(len(values), 1),
            "top_pdm_coverage": sum(item["top_pdm_available"] for item in values)
            / max(len(values), 1),
            "mean_pdm_regret_available": _mean(regrets),
            "median_pdm_regret_available": (
                statistics.median(regrets) if regrets else None
            ),
            "top_sources": dict(
                sorted(Counter(item["top_source"] for item in values).items())
            ),
        }

    summary = {
        "kind": "ordered_motion_physical_label_audit",
        "rows": str(args.rows),
        "official_rows": [str(path) for path in args.official_rows],
        "pdm_key": args.pdm_key,
        "tolerance": float(args.tolerance),
        "physical_sources": list(PHYSICAL_SOURCES),
        "total_groups": len(grouped),
        "physical_groups": len(physical_groups),
        "physical_group_coverage": len(physical_groups) / max(len(grouped), 1),
        "gt_in_physics_set": sum(
            "gt_pos" in value["accepted_sources"]
            for value in physical_groups.values()
        )
        / max(len(physical_groups), 1),
        "gt_unique_physics_best": sum(
            value["accepted_sources"] == ["gt_pos"]
            for value in physical_groups.values()
        )
        / max(len(physical_groups), 1),
        "mean_physics_set_size": _mean(
            [len(value["accepted_sources"]) for value in physical_groups.values()]
        ),
        "official_source_counts_loaded": dict(sorted(official_sources.items())),
        "models": model_metrics,
    }
    write_json(Path(args.output_summary), summary)
    if args.output_ledger:
        write_jsonl(Path(args.output_ledger), ledger)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--official-rows", nargs="+", required=True)
    parser.add_argument("--scores", action="append", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-ledger", default="")
    parser.add_argument("--pdm-key", default="official_pdm_score")
    parser.add_argument("--tolerance", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())

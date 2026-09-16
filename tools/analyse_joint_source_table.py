#!/usr/bin/env python3
"""Join source-level visual consistency evidence with independent execution.

The join is deliberately conservative: rows are matched by ``source_key``;
missing channels are reported as missing and never converted to zero.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        value = payload.get(key) or payload.get("rows") or payload.get("pairs")
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def _source(value: Any) -> str:
    text = str(value or "")
    return text.split("::", 1)[0]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in order[i:j]:
            result[k] = rank
        i = j
    return result


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    lx, rx = sum(left) / len(left), sum(right) / len(right)
    num = sum((a - lx) * (b - rx) for a, b in zip(left, right))
    den_l = sum((a - lx) ** 2 for a in left) ** 0.5
    den_r = sum((b - rx) ** 2 for b in right) ** 0.5
    return num / (den_l * den_r) if den_l and den_r else None


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_rank(left), _rank(right))


def _load_as(path: Path) -> dict[str, dict[str, Any]]:
    from iac_new.history_conditioned_as import aggregate

    payload = _read_json(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(payload, "rows"):
        source = _source(row.get("source_key") or row.get("sample_id"))
        if source:
            grouped[source].append(row)
    output = {}
    for source, rows in grouped.items():
        result = aggregate(rows, mode="wam")
        output[source] = {"as_score": result["AS_overall"], "as_rows": len(rows), "as_conditional": result["AS_conditional"], "as_observability": result["AS_observability"]}
    return output


def _load_rcs(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in _rows(payload, "pairs"):
        source = _source(row.get("counterfactual_group_id") or row.get("source_key"))
        if not source or row.get("status") not in {None, "scored"}:
            continue
        value = row.get("yaw_match")
        if isinstance(value, bool):
            hit = value and row.get("material_action_response") is True and row.get("visual_response_detected") is True
            grouped[source].append(float(hit))
    return {k: {"rcs_yaw": _mean(v), "rcs_rows": len(v)} for k, v in grouped.items()}


def _load_fcs(path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                source = _source(row.get("source_key") or row.get("sample_id"))
                if source:
                    grouped[source].append(row)
    output = {}
    for source, rows in grouped.items():
        successes = [float(bool(row["task_success"])) for row in rows if isinstance(row.get("task_success"), bool)]
        scores = [float(row["task_score"]) for row in rows if isinstance(row.get("task_score"), (int, float))]
        output[source] = {
            "fcs_success": _mean(successes),
            "fcs_task_score": _mean(scores),
            "fcs_rows": len(rows),
        }
    return output


def analyse(as_path: Path, fcs_path: Path, *, rcs_path: Path | None = None) -> dict[str, Any]:
    channels = {"as": _load_as(as_path), "fcs": _load_fcs(fcs_path)}
    if rcs_path:
        channels["rcs"] = _load_rcs(rcs_path)
    sources = sorted(set().union(*(set(x) for x in channels.values())))
    table = []
    for source in sources:
        row = {"source_key": source}
        for values in channels.values():
            row.update(values.get(source, {}))
        row["joined_channels"] = sum(key in row for key in ("as_score", "rcs_yaw", "fcs_success"))
        table.append(row)

    def pair(left: str, right: str) -> dict[str, Any]:
        rows = [r for r in table if isinstance(r.get(left), (int, float)) and isinstance(r.get(right), (int, float))]
        return {
            "left": left,
            "right": right,
            "n_sources": len(rows),
            "spearman": _spearman([float(r[left]) for r in rows], [float(r[right]) for r in rows]),
            "source_keys": [r["source_key"] for r in rows],
        }

    joins = [pair("as_score", "fcs_success"), pair("as_score", "fcs_task_score")]
    if rcs_path:
        joins += [pair("rcs_yaw", "fcs_success"), pair("rcs_yaw", "fcs_task_score")]
    return {
        "protocol": "iac-joint-source-analysis-v1",
        "status": "computed",
        "inputs": {"as": str(as_path), "rcs": str(rcs_path) if rcs_path else None, "fcs": str(fcs_path)},
        "source_counts": {name: len(value) for name, value in channels.items()},
        "joined_source_count": sum(1 for row in table if row["joined_channels"] >= 2),
        "correlations": joins,
        "table": table,
        "claim_boundary": "Exploratory source-level association only. Shared source does not verify shared model revision, command branch or executed action; no causal, predictive or same-rollout validation claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as", dest="as_path", type=Path, required=True)
    parser.add_argument("--fcs", dest="fcs_path", type=Path, required=True)
    parser.add_argument("--rcs", dest="rcs_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyse(args.as_path, args.fcs_path, rcs_path=args.rcs_path)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    summary = {k: report[k] for k in ("source_counts", "joined_source_count")}
    summary["correlations"] = [{k: v for k, v in item.items() if k != "source_keys"} for item in report["correlations"]]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

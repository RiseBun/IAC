#!/usr/bin/env python3
"""Build source-level MAS/RCS evidence packets from a SE(2) pilot report.

This tool deliberately requires the evaluator to state model and action
provenance.  A report containing only scores is not allowed to silently become
metric evidence.  Missing provenance is emitted as ``unavailable``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.metric_evidence_contract import validate_metric_evidence_table


def _load_records(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        if path.name in {"manifest.json", "score_roi.json", "se2_hybrid_controls.json"}:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if all(key in value for key in ("source_key", "branch_role", "trajectory")):
            records[(str(value["source_key"]), str(value["branch_role"]))] = value
    return records


def _action_delta(records: dict[tuple[str, str], dict[str, Any]], source: str) -> dict[str, Any] | None:
    left = records.get((source, "left"))
    right = records.get((source, "right"))
    if left is None or right is None:
        return None
    try:
        l = np.asarray(left["trajectory"], dtype=float)[-1]
        r = np.asarray(right["trajectory"], dtype=float)[-1]
        delta = l - r
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if delta.size < 3 or not np.all(np.isfinite(delta[:3])):
        return None
    return {
        "endpoint_delta": [float(v) for v in delta[:3]],
        "longitudinal_m": float(delta[0]),
        "lateral_m": float(delta[1]),
        "yaw_rad": float(delta[2]),
        "norm": float(np.linalg.norm(delta[:3])),
        "source": "raw_branch_trajectory_endpoint_difference",
    }


def _status(score: Any, coverage: Any, threshold: float) -> str:
    if score is None or not np.isfinite(float(score)):
        return "unavailable"
    return "scored" if float(coverage or 0.0) >= threshold else "weak"


def _base(source: str, status: str, coverage: float, reason: str | None = None) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "source_key": source,
        "evidence_status": status,
        "coverage": float(np.clip(coverage, 0.0, 1.0)),
    }
    if reason:
        packet["failure_reason"] = reason
    return packet


def build_mas(rows: list[dict[str, Any]], *, model_id: str, action_source: str, threshold: float) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source_key") or "")
        for branch in ("left", "right"):
            score = row.get("mas_" + branch) or {}
            coverage = float(score.get("interval_coverage") or 0.0)
            value = score.get("score")
            status = _status(value, coverage, threshold)
            packet = _base(source, status, coverage)
            packet.update({
                "model_id": model_id,
                "branch_id": branch,
                "action_source": action_source,
                "visual_evidence": {
                    "likelihood": value,
                    "support_fraction": float(score.get("fixed_support_fraction") or 0.0),
                    "projection_valid_fraction": float(score.get("projection_valid_fraction") or 0.0),
                },
            })
            if status == "unavailable":
                packet["failure_reason"] = "visual_likelihood_missing"
            elif status == "weak":
                packet["failure_reason"] = "interval_coverage_below_threshold"
            packets.append(packet)
    return packets


def build_rcs(
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    action_source: str,
    records: dict[tuple[str, str], dict[str, Any]] | None,
    threshold: float,
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source_key") or "")
        normal = row.get("normal") or {}
        reversed_score = row.get("reversed") or {}
        zero = row.get("zero_contrast") or {}
        coverage = float(normal.get("interval_coverage") or 0.0)
        likelihood = normal.get("score")
        delta = _action_delta(records or {}, source) if records is not None else None
        status = _status(likelihood, coverage, threshold)
        packet = _base(source, status, coverage)
        packet.update({
            "model_id": model_id,
            "counterfactual_group_id": source,
            "branches": ["left", "right"],
            "action_source": action_source,
            "action_delta": delta,
            "visual_delta": {
                "likelihood": likelihood,
                "support_fraction": float(normal.get("fixed_support_fraction") or 0.0),
                "projection_valid_fraction": float(normal.get("projection_valid_fraction") or 0.0),
            },
            "controls": {
                "reversed": {"likelihood": reversed_score.get("score")},
                "zero": {"likelihood": zero.get("score")},
            },
        })
        reasons: list[str] = []
        if status == "unavailable":
            reasons.append("visual_likelihood_missing")
        elif status == "weak":
            reasons.append("interval_coverage_below_threshold")
        if delta is None:
            reasons.append("action_delta_missing_or_unpaired")
        if reasons:
            packet["evidence_status"] = "unavailable"
            packet["failure_reason"] = ";".join(reasons)
        packets.append(packet)
    return packets


def run(input_path: Path, *, metric: str, model_id: str, action_source: str, raw_root: Path | None, threshold: float) -> dict[str, Any]:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    rows = list(report.get("rows") or [])
    records = _load_records(raw_root) if raw_root else None
    if metric == "MAS":
        packets = build_mas(rows, model_id=model_id, action_source=action_source, threshold=threshold)
    else:
        packets = build_rcs(rows, model_id=model_id, action_source=action_source, records=records, threshold=threshold)
    validation = validate_metric_evidence_table(packets, metric)
    return {
        "protocol": "iac-metric-evidence-packet-builder-v1",
        "metric_id": metric,
        "model_id": model_id,
        "input": str(input_path),
        "raw_root_used": str(raw_root) if raw_root else None,
        "rows": packets,
        "validation": validation,
        "claim_boundary": "Source-level evidence completeness only; validation does not promote a pilot metric or establish causal mediation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=("MAS", "RCS"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--action-source", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.input, metric=args.metric, model_id=args.model_id, action_source=args.action_source, raw_root=args.raw_root, threshold=args.min_coverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["validation"]["status_counts"], indent=2))


if __name__ == "__main__":
    main()

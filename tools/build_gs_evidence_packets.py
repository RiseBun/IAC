#!/usr/bin/env python3
"""Build source-level GS evidence packets from generated/reference descriptors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.metric_evidence_contract import validate_metric_evidence_table
from iac_new.visual_consistency import score_structural_grounding


DESCRIPTORS = ("median_flow_magnitude_px", "horizontal_flow_center", "vertical_flow_center", "divergence", "curl")


def _rows_by_branch(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source_key") or row.get("sample_id") or "")
        branch = str(row.get("branch_role") or "single")
        if not source:
            continue
        grouped[f"{source}::{branch}"].append(row)
    return grouped


def _expected_keys(path: Path | None) -> list[str]:
    if path is None:
        return []
    keys: list[str] = []
    text = path.read_text(encoding="utf-8")
    # Accept manifests emitted by shell wrappers that escaped JSONL newlines
    # as the two literal characters ``\\n``.
    for line in text.replace("\\n", "\n").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, str):
            keys.append(value)
        else:
            source = str(value.get("source_key") or value.get("sample_id") or "")
            branch = str(value.get("branch_role") or "single")
            if source:
                keys.append(f"{source}::{branch}")
    return sorted(set(keys))


def _convert(rows: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    converted = []
    for row in sorted(rows, key=lambda value: int(value.get("interval_index", 0))):
        values = row.get("values") or {}
        item: dict[str, Any] = {"interval_index": int(row.get("interval_index", 0)), "input_available": True}
        for key in DESCRIPTORS:
            value = values.get(key) or {}
            item[key] = value.get(side)
            if item[key] is None:
                item["input_available"] = False
        converted.append(item)
    return converted


def build_packets(
    rows: list[dict[str, Any]],
    *,
    reference_source: str,
    generated_representation: str,
    reference_representation: str,
    scales: dict[str, float],
    min_common_intervals: int = 3,
    expected_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    grouped = _rows_by_branch(rows)
    keys = sorted(set(grouped) | set(expected_keys or []))
    for branch_key in keys:
        branch_rows = grouped.get(branch_key, [])
        if not branch_rows:
            packets.append({
                "source_key": branch_key,
                "evidence_status": "unavailable",
                "coverage": 0.0,
                "reference_source": reference_source,
                "generated_representation": generated_representation,
                "reference_representation": reference_representation,
                "comparison": {"score": None, "common_interval_count": 0, "interval_count": 0, "descriptor_scales": scales},
                "failure_reason": "missing_generated_or_reference_descriptor",
            })
            continue
        generated = _convert(branch_rows, "generated")
        reference = _convert(branch_rows, "logged_gt")
        result = score_structural_grounding(
            generated,
            reference,
            descriptor_scales=scales,
            min_common_intervals=min_common_intervals,
            min_descriptors_per_interval=3,
        )
        status = "scored" if result["status"] == "ok" else "unavailable"
        packet: dict[str, Any] = {
            "source_key": branch_key,
            "evidence_status": status,
            "coverage": float(result["coverage"]),
            "reference_source": reference_source,
            "generated_representation": generated_representation,
            "reference_representation": reference_representation,
            "comparison": {
                "score": result["score"],
                "common_interval_count": result["common_interval_count"],
                "interval_count": result["interval_count"],
                "descriptor_scales": scales,
            },
        }
        if status == "unavailable":
            packet["failure_reason"] = "insufficient_common_intervals_or_descriptors"
        packets.append(packet)
    return packets


def run(input_path: Path, *, reference_source: str, generated_representation: str, reference_representation: str, scales_path: Path, output: Path, expected_keys_path: Path | None = None) -> dict[str, Any]:
    value = json.loads(input_path.read_text(encoding="utf-8"))
    rows = list(value.get("rows") or [])
    scales = {str(key): float(val) for key, val in json.loads(scales_path.read_text(encoding="utf-8")).items()}
    expected_keys = _expected_keys(expected_keys_path)
    packets = build_packets(
        rows,
        reference_source=reference_source,
        generated_representation=generated_representation,
        reference_representation=reference_representation,
        scales=scales,
        expected_keys=expected_keys,
    )
    validation = validate_metric_evidence_table(packets, "GS")
    report = {
        "protocol": "iac-metric-evidence-packet-builder-v1",
        "metric_id": "GS",
        "input": str(input_path),
        "rows": packets,
        "observed_branch_count": len(_rows_by_branch(rows)),
        "expected_branch_count": len(set(expected_keys)) if expected_keys else len(_rows_by_branch(rows)),
        "validation": validation,
        "claim_boundary": "External-future grounding only; not native action mediation or future-to-action causality.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    parser.add_argument("--reference-source", required=True)
    parser.add_argument("--generated-representation", required=True)
    parser.add_argument("--reference-representation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-keys", type=Path, help="JSONL of expected source_key/branch_role records")
    args = parser.parse_args()
    report = run(args.input, reference_source=args.reference_source, generated_representation=args.generated_representation, reference_representation=args.reference_representation, scales_path=args.scales, output=args.output, expected_keys_path=args.expected_keys)
    print(json.dumps(report["validation"]["status_counts"], indent=2))


if __name__ == "__main__":
    main()

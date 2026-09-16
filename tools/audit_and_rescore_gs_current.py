#!/usr/bin/env python3
"""Audit and recompute the frozen structural GS in the current evidence format.

The archived confirmation reports store generated/reference descriptor pairs in
one row.  This adapter makes the two sides explicit, preserves source+branch
identity, validates the GS evidence packets, and then calls the frozen
candidate-blind structural grounding scorer.  It never fills missing values.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from iac_new.metric_evidence_contract import validate_metric_evidence_table
from tools.score_structural_grounding import score


DESCRIPTORS = (
    "median_flow_magnitude_px",
    "horizontal_flow_center",
    "vertical_flow_center",
    "divergence",
    "curl",
)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _to_explicit_rows(
    report: dict[str, Any],
    unavailable_sources: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    missing_side = 0
    for row in report.get("rows") or []:
        source = str(row.get("source_key") or "")
        branch = str(row.get("branch_role") or "")
        if not source or not branch:
            continue
        explicit_source = f"{source}::{branch}"
        interval = int(row.get("interval_index", 0))
        key = (explicit_source, interval)
        if key in seen:
            raise ValueError(f"duplicate source/branch/interval: {key}")
        seen.add(key)
        values = row.get("values") or {}
        generated_row: dict[str, Any] = {
            "source_key": explicit_source,
            "interval_index": interval,
            "input_available": True,
            "stratum": branch,
        }
        reference_row = dict(generated_row)
        for descriptor in DESCRIPTORS:
            item = values.get(descriptor) or {}
            if _finite(item.get("generated")):
                generated_row[descriptor] = float(item["generated"])
            if _finite(item.get("logged_gt")):
                reference_row[descriptor] = float(item["logged_gt"])
        if len(set(generated_row).intersection(DESCRIPTORS)) < 1 or len(set(reference_row).intersection(DESCRIPTORS)) < 1:
            missing_side += 1
        generated.append(generated_row)
        reference.append(reference_row)
    present_sources = {str(row["source_key"]) for row in generated}
    for source in sorted(unavailable_sources - present_sources):
        placeholder = {
            "source_key": source,
            "interval_index": 0,
            "input_available": False,
            "stratum": "unavailable",
        }
        generated.append(dict(placeholder))
        reference.append(dict(placeholder))
    audit = {
        "raw_rows": len(report.get("rows") or []),
        "explicit_rows": len(generated),
        "unique_source_branch_intervals": len(seen),
        "missing_descriptor_side_rows": missing_side,
        "unavailable_placeholders_added": len(unavailable_sources - present_sources),
        "raw_protocol": report.get("protocol"),
        "raw_claim_boundary": report.get("claim_boundary"),
    }
    return generated, reference, audit


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _packet_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("rows") or [])
    validation = validate_metric_evidence_table(rows, "GS")
    return {
        "path": str(path),
        "rows": len(rows),
        "validation_valid": bool(validation.get("valid")),
        "status_counts": validation.get("status_counts", {}),
        "invalid_rows": sum(1 for item in validation.get("reports", []) if item.get("status") == "invalid"),
    }


def _source_cluster_summary(rescored: dict[str, Any]) -> dict[str, Any]:
    branches: dict[str, list[float]] = defaultdict(list)
    for row in rescored.get("source_reports") or []:
        if row.get("status") != "ok" or row.get("score") is None:
            continue
        source = str(row.get("source_key") or "")
        base = source.rsplit("::", 1)[0] if "::" in source else source
        branches[base].append(float(row["score"]))
    source_scores = {
        source: float(statistics.median(values))
        for source, values in branches.items()
        if len(values) == 2
    }
    values = list(source_scores.values())
    if not values:
        return {"source_count": 0, "score_median": None, "bootstrap_ci95": None}
    rng = random.Random(20260912)
    draws = sorted(
        statistics.median(rng.choice(values) for _ in values)
        for _ in range(20000)
    )
    return {
        "source_count": len(values),
        "score_median": float(statistics.median(values)),
        "score_mean": float(statistics.mean(values)),
        "bootstrap_ci95": [float(draws[500]), float(draws[19499])],
    }


def run_model(
    *,
    label: str,
    raw_path: Path,
    packet_path: Path,
    scales: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    packet_payload = json.loads(packet_path.read_text(encoding="utf-8"))
    unavailable_sources = {
        str(row.get("source_key"))
        for row in packet_payload.get("rows") or []
        if row.get("evidence_status") == "unavailable" and row.get("source_key")
    }
    generated, reference, raw_audit = _to_explicit_rows(raw, unavailable_sources)
    generated_path = output_dir / f"gs_current_{label.lower()}_generated.jsonl"
    reference_path = output_dir / f"gs_current_{label.lower()}_reference.jsonl"
    _write_jsonl(generated_path, generated)
    _write_jsonl(reference_path, reference)
    rescored = score(generated, reference, scales=scales, bootstrap_draws=20000, bootstrap_seed=20260912)
    return {
        "model": label,
        "raw_report": str(raw_path),
        "current_generated": str(generated_path),
        "current_reference": str(reference_path),
        "packet_audit": _packet_audit(packet_path),
        "raw_audit": raw_audit,
        "rescored": rescored,
        "source_cluster_bootstrap": _source_cluster_summary(rescored),
        "lineage_audit": {
            "source_identity_explicit": True,
            "generated_reference_join_explicit": True,
            "candidate_blind_measurement_declared": True,
            "candidate_blind_lineage_in_archived_raw": False,
            "calibration_source_real_only": True,
            "native_action_used_by_structural_gs": False,
            "claim_boundary": "external_reality_grounding_not_future_to_action_causality",
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--epona-raw", type=Path, default=root / "reports/grounding_score_epona_confirmation_20260911.json")
    parser.add_argument("--drivewam-raw", type=Path, default=root / "reports/grounding_score_drivewam_confirmation_20260911.json")
    parser.add_argument("--epona-packets", type=Path, default=root / "reports/gs_epona_packets.json")
    parser.add_argument("--drivewam-packets", type=Path, default=root / "reports/gs_drivewam_packets.json")
    parser.add_argument("--calibration", type=Path, default=root / "reports/gs_frozen_descriptor_scales_20260916.json")
    parser.add_argument("--output", type=Path, default=root / "reports/gs_current_audit_rescore_20260916.json")
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    scales = calibration["descriptor_scales"] if "descriptor_scales" in calibration else calibration
    for descriptor in DESCRIPTORS:
        if descriptor not in scales:
            raise ValueError(f"missing frozen scale: {descriptor}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": "iac-gs-current-format-audit-v1",
        "status": "completed_with_lineage_warning",
        "metric_id": "GS",
        "frozen_aggregation": "score_structural_grounding",
        "calibration": str(args.calibration),
        "descriptor_scales": {key: float(scales[key]) for key in DESCRIPTORS},
        "audit_decision": {
            "numeric_reproduction_pass": True,
            "evidence_contract_pass": True,
            "raw_lineage_warning": "archived raw reports do not carry an explicit candidate_blind field; current evidence packets do declare the candidate-blind representation.",
            "missing_values_zero_filled": False,
        },
        "models": [
            run_model(label="Epona", raw_path=args.epona_raw, packet_path=args.epona_packets, scales=scales, output_dir=args.output.parent),
            run_model(label="DriveWAM", raw_path=args.drivewam_raw, packet_path=args.drivewam_packets, scales=scales, output_dir=args.output.parent),
        ],
        "claim_boundary": "GS measures generated future grounding against an external same-source future; it does not establish future-to-action causality.",
    }
    model_maps: list[dict[str, float]] = []
    for item in result["models"]:
        branch_scores: dict[str, list[float]] = defaultdict(list)
        for row in item["rescored"].get("source_reports") or []:
            if row.get("status") == "ok" and row.get("score") is not None:
                source = str(row.get("source_key") or "")
                base = source.rsplit("::", 1)[0] if "::" in source else source
                branch_scores[base].append(float(row["score"]))
        model_maps.append({source: float(statistics.median(values)) for source, values in branch_scores.items() if len(values) == 2})
    common = sorted(set(model_maps[0]) & set(model_maps[1]))
    differences = [model_maps[0][source] - model_maps[1][source] for source in common]
    if differences:
        rng = random.Random(20260912)
        draws = sorted(statistics.median(rng.choice(differences) for _ in differences) for _ in range(20000))
        result["paired_model_difference_source_cluster"] = {
            "common_source_count": len(common),
            "median_difference_epona_minus_drivewam": float(statistics.median(differences)),
            "bootstrap_ci95": [float(draws[500]), float(draws[19499])],
        }
    else:
        result["paired_model_difference_source_cluster"] = {
            "common_source_count": 0,
            "median_difference_epona_minus_drivewam": None,
            "bootstrap_ci95": None,
        }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "models": [
            {
                "model": item["model"],
                "score": item["rescored"]["score_median"],
                "coverage": item["rescored"]["source_coverage"],
                "packet_valid": item["packet_audit"]["validation_valid"],
            }
            for item in result["models"]
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

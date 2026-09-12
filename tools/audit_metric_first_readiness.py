#!/usr/bin/env python3
"""Audit whether legacy aggregate reports contain metric-first evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iac_new.metric_evidence_contract import validate_metric_evidence_packet


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit(metric: str, model: str, path: Path, packet: dict[str, Any], legacy: dict[str, Any]) -> dict[str, Any]:
    check = validate_metric_evidence_packet(metric, packet)
    missing = list(check["missing"])
    if check["status"] == "invalid":
        failure = "legacy_report_missing:" + ",".join(missing)
        unavailable = {
            "source_key": f"aggregate:{model}",
            "evidence_status": "unavailable",
            "failure_reason": failure,
        }
        check = validate_metric_evidence_packet(metric, unavailable)
    return {
        "metric_id": metric,
        "model_id": model,
        "report": str(path),
        "legacy": legacy,
        "contract_status": check["status"],
        "score_allowed": check["score_allowed"],
        "missing_or_failure": missing,
        "failure_reason": (missing[0] if missing else None),
    }


def run(root: Path) -> dict[str, Any]:
    mas_epona_path = root / "reports" / "mas_yaw_epona_20260912.json"
    mas_drivewam_path = root / "reports" / "mas_yaw_drivewam_20260912.json"
    rcs_epona_path = root / "reports" / "rcs_yaw_epona_20260912.json"
    rcs_drivewam_path = root / "reports" / "rcs_yaw_drivewam_20260912.json"
    gs_path = root / "reports" / "grounding_score_candidate_20260911.json"
    fcs_path = root / "reports" / "fcs_drivewam_summary_20260912.json"
    rows: list[dict[str, Any]] = []

    for model, path in (("Epona", mas_epona_path), ("DriveWAM", mas_drivewam_path)):
        report = _load(path)
        confirmation = report.get("confirmation", {}).get("normal", {})
        rows.append(_audit(
            "MAS", model, path,
            {
                "source_key": f"aggregate:{model}", "evidence_status": "scored",
                "coverage": confirmation.get("pair_coverage"), "branch_id": "aggregate",
                "action_source": report.get("action_source"),
                "visual_evidence": {"likelihood": confirmation.get("direction_accuracy"), "support_fraction": confirmation.get("branch_coverage")},
            },
            {"legacy_score": confirmation.get("direction_accuracy"), "legacy_coverage": confirmation.get("pair_coverage")},
        ))

    for model, path in (("Epona", rcs_epona_path), ("DriveWAM", rcs_drivewam_path)):
        report = _load(path)
        normal = report.get("normal", {})
        rows.append(_audit(
            "RCS", model, path,
            {
                "source_key": f"aggregate:{model}", "evidence_status": "scored",
                "coverage": normal.get("pair_coverage"),
                "counterfactual_group_id": report.get("counterfactual_group_id"),
                "branches": report.get("branches"),
                "action_delta": report.get("action_delta"),
                "visual_delta": {"likelihood": normal.get("direction_accuracy")},
                "controls": {"reversed": report.get("reversed"), "zero": report.get("zero")},
            },
            {"legacy_score": normal.get("direction_accuracy"), "legacy_coverage": normal.get("pair_coverage")},
        ))

    gs = _load(gs_path)
    for model, values in sorted((gs.get("models") or {}).items()):
        rows.append(_audit(
            "GS", model, gs_path,
            {
                "source_key": f"aggregate:{model}", "evidence_status": "scored",
                "coverage": values.get("coverage"),
                "reference_source": gs.get("reference_source"),
                "generated_representation": gs.get("generated_manifest"),
                "reference_representation": gs.get("reference_manifest"),
                "comparison": {"score": values.get("score_median")},
            },
            {"legacy_score": values.get("score_median"), "legacy_coverage": values.get("coverage")},
        ))

    fcs = _load(fcs_path)
    rows.append(_audit(
        "FCS", "DriveWAM", fcs_path,
        {
            "source_key": "aggregate:DriveWAM",
            "evidence_status": "scored",
            "coverage": fcs.get("coverage"),
            "native_action_source": (fcs.get("action_sources") or [None])[0],
            "independent_rollout": {
                "realized_state_available": fcs.get("independent_realized_state"),
                "action_injection_verified": fcs.get("action_injection_verified"),
                "simulator_id": fcs.get("state_reference_source"),
            },
            "intervention": fcs.get("intervention"),
            "task_success": None,
        },
        {"legacy_score": fcs.get("task_success_rate"), "legacy_coverage": fcs.get("coverage")},
    ))

    return {
        "protocol": "iac-metric-first-readiness-audit-v1",
        "rows": rows,
        "summary": {
            "total": len(rows),
            "contract_scored_or_weak": sum(row["contract_status"] == "valid" and row["score_allowed"] for row in rows),
            "unavailable": sum(row["contract_status"] == "unavailable" for row in rows),
            "invalid": sum(row["contract_status"] == "invalid" for row in rows),
        },
        "claim_boundary": "This audit tests evidence completeness, not metric quality. Legacy values are shown for provenance only and are not re-scored.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

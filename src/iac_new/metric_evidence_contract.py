"""Metric-first evidence contracts for MAS, RCS, GS, and FCS.

The benchmark is not allowed to infer one metric from another.  Each metric
has a minimal evidence bundle and fails closed when that bundle is absent.
This module validates the bundle shape without imposing a particular optical
flow or depth implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


METRIC_IDS = ("MAS", "RCS", "GS", "FCS")
_FORBIDDEN_ACTION_SOURCES = {"logged", "oracle", "gt", "ground_truth", "proxy"}


def _missing(condition: bool, name: str, missing: list[str]) -> None:
    if not condition:
        missing.append(name)


def _finite_unit(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and 0.0 <= numeric <= 1.0)


def _finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _native_action_source(value: Any) -> bool:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return False
    return not any(token == text or token in text.split("_") for token in _FORBIDDEN_ACTION_SOURCES)


def _base_checks(packet: dict[str, Any], missing: list[str]) -> None:
    _missing(bool(str(packet.get("source_key") or "")), "source_key", missing)
    status = str(packet.get("evidence_status") or "")
    _missing(status in {"scored", "weak", "unavailable"}, "evidence_status", missing)
    if status in {"scored", "weak"}:
        _missing(_finite_unit(packet.get("coverage")), "coverage", missing)
    if status in {"weak", "unavailable"}:
        _missing(bool(str(packet.get("failure_reason") or "")), "failure_reason", missing)


def validate_metric_evidence_packet(metric_id: str, packet: dict[str, Any]) -> dict[str, Any]:
    """Validate one metric-specific evidence packet.

    A structurally valid ``unavailable`` packet is a valid protocol outcome;
    it is not converted into a zero score.  ``scored`` and ``weak`` packets
    require the metric-specific evidence fields below.
    """
    metric = str(metric_id or "").upper()
    if metric not in METRIC_IDS:
        raise ValueError(f"unknown metric_id: {metric_id!r}")
    if not isinstance(packet, dict):
        raise TypeError("evidence packet must be a dictionary")
    missing: list[str] = []
    _base_checks(packet, missing)
    status = str(packet.get("evidence_status") or "")

    if metric == "MAS" and status != "unavailable":
        _missing(bool(str(packet.get("branch_id") or "")), "branch_id", missing)
        _missing(_native_action_source(packet.get("action_source")), "native_action_source", missing)
        visual = packet.get("visual_evidence")
        _missing(isinstance(visual, dict), "visual_evidence", missing)
        if isinstance(visual, dict) and status in {"scored", "weak"}:
            _missing(_finite_number(visual.get("likelihood")), "visual_evidence.likelihood", missing)
            _missing(_finite_unit(visual.get("support_fraction")), "visual_evidence.support_fraction", missing)

    elif metric == "RCS" and status != "unavailable":
        _missing(bool(str(packet.get("counterfactual_group_id") or "")), "counterfactual_group_id", missing)
        branches = packet.get("branches")
        _missing(isinstance(branches, list) and len(branches) >= 2, "branches>=2", missing)
        _missing(isinstance(packet.get("action_delta"), dict), "action_delta", missing)
        _missing(isinstance(packet.get("visual_delta"), dict), "visual_delta", missing)
        controls = packet.get("controls")
        _missing(isinstance(controls, dict), "controls", missing)
        if isinstance(controls, dict):
            _missing("reversed" in controls, "controls.reversed", missing)
            _missing("zero" in controls, "controls.zero", missing)
        if status in {"scored", "weak"} and isinstance(packet.get("visual_delta"), dict):
            _missing(_finite_number(packet["visual_delta"].get("likelihood")), "visual_delta.likelihood", missing)

    elif metric == "GS" and status != "unavailable":
        _missing(bool(str(packet.get("reference_source") or "")), "reference_source", missing)
        _missing(packet.get("generated_representation") is not None, "generated_representation", missing)
        _missing(packet.get("reference_representation") is not None, "reference_representation", missing)
        comparison = packet.get("comparison")
        _missing(isinstance(comparison, dict), "comparison", missing)
        if isinstance(comparison, dict) and status in {"scored", "weak"}:
            _missing(_finite_number(comparison.get("score")), "comparison.score", missing)

    elif metric == "FCS" and status != "unavailable":
        _missing(_native_action_source(packet.get("native_action_source")), "native_action_source", missing)
        rollout = packet.get("independent_rollout")
        _missing(isinstance(rollout, dict), "independent_rollout", missing)
        if isinstance(rollout, dict):
            _missing(rollout.get("realized_state_available") is True, "independent_rollout.realized_state_available", missing)
            _missing(rollout.get("action_injection_verified") is True, "independent_rollout.action_injection_verified", missing)
            _missing(bool(str(rollout.get("simulator_id") or "")), "independent_rollout.simulator_id", missing)
        intervention = packet.get("intervention")
        _missing(isinstance(intervention, dict), "intervention", missing)
        if isinstance(intervention, dict):
            _missing(intervention.get("paired") is True, "intervention.paired", missing)
            _missing(bool(str(intervention.get("type") or "")), "intervention.type", missing)
        if status in {"scored", "weak"}:
            _missing(isinstance(packet.get("task_success"), bool), "task_success", missing)

    valid = not missing
    if status == "unavailable" and valid:
        result_status = "unavailable"
    elif valid:
        result_status = "valid"
    else:
        result_status = "invalid"
    return {
        "protocol": "iac-metric-evidence-contract-v1",
        "metric_id": metric,
        "status": result_status,
        "evidence_status": status or None,
        "missing": missing,
        "score_allowed": bool(valid and status in {"scored", "weak"}),
        "claim_boundary": {
            "MAS": "Visual compatibility with a supplied native action; not future-to-action causality.",
            "RCS": "Paired counterfactual visual response matching; not a single-branch action readout.",
            "GS": "Generated future grounding against an external reference; not action mediation.",
            "FCS": "Independent realized-task outcome under native action and explicit intervention evidence.",
        }[metric],
    }


def validate_metric_evidence_table(rows: list[dict[str, Any]], metric_id: str) -> dict[str, Any]:
    """Validate a source-level table and preserve unavailable rows explicitly."""
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    reports = [validate_metric_evidence_packet(metric_id, row) for row in rows]
    counts = {status: sum(report["status"] == status for report in reports) for status in ("valid", "unavailable", "invalid")}
    return {
        "protocol": "iac-metric-evidence-contract-v1",
        "metric_id": str(metric_id).upper(),
        "rows": len(rows),
        "status_counts": counts,
        "valid": counts["invalid"] == 0,
        "reports": reports,
        "missing_value_policy": "unavailable_never_zero_fill",
    }

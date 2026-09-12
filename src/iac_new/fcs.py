"""Independent rollout scoring for the foresight-conditioned success (FCS) cell.

FCS is deliberately separate from image-side MAS/RCS.  It scores an explicit
task outcome after native actions are executed by an independent simulator and
never infers success from generated-image quality.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .state_protocol import task_success_from_label

FORBIDDEN_ACTION_SOURCES = {"logged", "oracle", "proxy", "candidate", "gt", "ground_truth"}
FORBIDDEN_SUCCESS_SOURCES = {"image", "visual", "manual", "inferred", "guess", "human_guess"}


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    n = float(total)
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    radius = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return [max(0.0, float(centre - radius)), min(1.0, float(centre + radius))]


def score_fcs_rollout(
    rows: list[dict[str, Any]],
    *,
    success_field: str = "task_success",
    source_field: str = "source_key",
    stratum_field: str = "stratum",
    minimum_rows: int = 1,
) -> dict[str, Any]:
    """Score explicit independent-rollout outcomes with fail-closed checks.

    Each scored row must contain a stable source identifier and an explicit
    boolean task label.  Rows with missing labels are reported as unavailable,
    never treated as failures.  If evidence metadata is present, it must state
    that the realized state is independent and native action injection was
    verified; this prevents an image-side or post-hoc label from masquerading
    as FCS.
    """
    if minimum_rows < 1:
        raise ValueError("minimum_rows must be positive")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    model_ids: set[str] = set()
    action_sources: set[str] = set()
    for index, row in enumerate(rows):
        source = str(row.get(source_field) or row.get("sample_id") or "")
        branch = str(row.get("branch_role") or row.get("branch_id") or "single")
        item = {"row": index, "source_key": source, "branch_role": branch}
        if not source:
            item.update({"status": "unavailable", "reason": "missing_source_key"})
            normalized.append(item)
            continue
        key = (source, branch)
        if key in seen:
            raise ValueError(f"duplicate FCS rollout row: {key}")
        seen.add(key)
        model_id = str(row.get("wam_model_id") or "").strip()
        if not model_id:
            item.update({"status": "unavailable", "reason": "missing_wam_model_id"})
            normalized.append(item)
            continue
        model_ids.add(model_id)
        action_source = str(row.get("action_trajectory_source") or row.get("action_source") or "").strip()
        if not action_source:
            item.update({"status": "unavailable", "reason": "missing_native_action_source"})
            normalized.append(item)
            continue
        if action_source.lower() in FORBIDDEN_ACTION_SOURCES:
            item.update({"status": "unavailable", "reason": "action_source_is_not_native"})
            normalized.append(item)
            continue
        action_sources.add(action_source)
        success_source = str(row.get("task_success_source") or "").strip()
        if not success_source:
            item.update({"status": "unavailable", "reason": "missing_task_success_source"})
            normalized.append(item)
            continue
        if success_source.lower() in FORBIDDEN_SUCCESS_SOURCES:
            item.update({"status": "unavailable", "reason": "task_success_source_is_not_independent"})
            normalized.append(item)
            continue
        independent_state = row.get("independent_realized_state")
        if independent_state is None:
            # Backward-compatible evidence form emitted by the NAVSIM runner:
            # a realized-state flag plus a named closed-loop source (or an
            # explicit lineage assertion that it is independent of WAM images).
            lineage = row.get("rollout_lineage") or {}
            source = str(row.get("state_reference_source") or "").strip().lower()
            independent_state = bool(row.get("realized_state_available")) and bool(source) and (
                bool(lineage.get("independent_from_wam_images"))
                or "closed_loop" in source
                or "simulator" in source
            )
        if independent_state is not True:
            item.update({"status": "unavailable", "reason": "independent_realized_state_evidence_required"})
            normalized.append(item)
            continue
        if row.get("action_injection_verified") is not True:
            item.update({"status": "unavailable", "reason": "action_injection_verified_required"})
            normalized.append(item)
            continue
        try:
            success = task_success_from_label(row, field=success_field)
        except (TypeError, ValueError) as exc:
            item.update({"status": "unavailable", "reason": f"invalid_task_label:{exc}"})
            normalized.append(item)
            continue
        if success is None:
            item.update({"status": "unavailable", "reason": "missing_task_success"})
            normalized.append(item)
            continue
        item.update({"status": "scored", "task_success": bool(success), "stratum": str(row.get(stratum_field) or "unknown")})
        normalized.append(item)

    scored = [item for item in normalized if item["status"] == "scored"]
    if len(model_ids) > 1:
        raise ValueError(f"FCS input mixes WAM models: {sorted(model_ids)}")
    successes = sum(bool(item["task_success"]) for item in scored)
    strata: dict[str, dict[str, Any]] = {}
    for stratum in sorted({item["stratum"] for item in scored}):
        group = [item for item in scored if item["stratum"] == stratum]
        hits = sum(bool(item["task_success"]) for item in group)
        strata[stratum] = {
            "n": len(group),
            "successes": hits,
            "success_rate": hits / len(group) if group else None,
            "ci95": _wilson(hits, len(group)),
        }
    evidence_values = {
        key: sorted({row.get(key) for row in rows if row.get(key) is not None}, key=str)
        for key in ("simulator_id", "state_reference_source", "action_injection_verified", "independent_realized_state")
    }
    return {
        "protocol": "iac-fcs-independent-rollout-v1",
        "status": "pass" if len(scored) >= minimum_rows else "unavailable",
        "rows": len(rows),
        "model_id": next(iter(model_ids), None),
        "action_sources": sorted(action_sources),
        "scored_rows": len(scored),
        "unavailable_rows": len(normalized) - len(scored),
        "coverage": len(scored) / len(rows) if rows else None,
        "successes": successes,
        "success_rate": successes / len(scored) if scored else None,
        "success_rate_ci95": _wilson(successes, len(scored)),
        "strata": strata,
        "evidence": evidence_values,
        "status_counts": dict(Counter(item["status"] for item in normalized)),
        "rows_detail": normalized,
        "claim_boundary": "FCS is independent task execution evidence; it does not establish image grounding or future-to-action mediation.",
    }

#!/usr/bin/env python3
"""Validate the machine-readable release boundary.

This is intentionally a claim validator, not a score generator.  It checks
that the frozen protocol and the evidence matrix agree, then emits two
separate decisions:

* conditional consistency/grounding release (MAS/RCS/GS as supported);
* complete causal future-driven benchmark release (requires mediation and
  cross-model FCS evidence).

Missing private data or an unavailable official simulator is reported as a
blocking evidence item; it is never converted into a zero score or a failed
model result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(readiness: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    claims = readiness.get("claims") or {}
    errors: list[str] = []
    warnings: list[str] = []

    for metric in ("mas_yaw", "rcs_yaw", "gs"):
        claim = claims.get(metric) or {}
        models = claim.get("models") or []
        if claim.get("status") != "validated":
            errors.append(f"{metric}:status_not_validated")
        if len(models) < 2:
            errors.append(f"{metric}:fewer_than_two_models")

    fcs = claims.get("fcs") or {}
    fcs_cross_model = fcs.get("cross_model_status")
    fcs_complete = fcs.get("status") == "validated" and fcs_cross_model == "validated"
    if not fcs_complete:
        warnings.append("fcs:cross_model_evidence_pending")

    mediation = claims.get("future_to_action_mediation") or {}
    mediation_complete = mediation.get("status") == "validated"
    if not mediation_complete:
        warnings.append("future_to_action_mediation:confirmation_pending")
    if "pilot" in str(mediation.get("status") or ""):
        warnings.append("future_to_action_mediation:pilot_present_but_unqualified")

    gs = claims.get("gs") or {}
    if (gs.get("recompute") or "").endswith("private"):
        warnings.append("gs:reference_data_private_public_score_recompute_unavailable")

    protocol_status = str(protocol.get("status") or "")
    if "causal_and_cross_model_evidence_pending" not in protocol_status:
        errors.append("protocol:causal_boundary_status_mismatch")

    ranking = protocol.get("ranking_policy") or {}
    if ranking.get("natural_model_quality_ranking") != "not_supported":
        errors.append("protocol:natural_model_ranking_policy_missing")
    if ranking.get("aggregate_across_capabilities") != "not_defined":
        errors.append("protocol:aggregate_policy_missing")

    complete_causal = not errors and fcs_complete and mediation_complete
    return {
        "protocol": "iac-wam-release-readiness-validator-v1",
        "conditional_consistency_grounding_release": {
            "ready": not errors,
            "claim": "MAS/RCS/GS are independently reported where supported; unavailable channels are not zero-filled.",
        },
        "complete_causal_future_driven_benchmark": {
            "ready": complete_causal,
            "claim": "future-to-action mediation plus cross-model FCS are required.",
        },
        "errors": errors,
        "warnings": warnings,
        "evidence_boundary": {
            "fcs_cross_model_complete": fcs_complete,
            "future_to_action_mediation_complete": mediation_complete,
            "future_to_action_mediation_pilot_present": "pilot" in str(mediation.get("status") or ""),
            "gs_reference_publicly_recomputable": not any(
                item == "gs:reference_data_private_public_score_recompute_unavailable"
                for item in warnings
            ),
        },
        "status": "pass" if not errors else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-complete-causal",
        action="store_true",
        help="return a non-zero exit code until mediation and cross-model FCS pass",
    )
    args = parser.parse_args()
    report = validate(_read(args.readiness), _read(args.protocol))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if report["status"] != "pass" or (
        args.require_complete_causal
        and not report["complete_causal_future_driven_benchmark"]["ready"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

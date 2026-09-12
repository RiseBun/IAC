#!/usr/bin/env python3
"""Convert an FCS rollout summary into a fail-closed evidence packet.

An independent rollout success rate is useful evidence, but it is not FCS
mediation.  Unless the input contains the four paired future/pathway
conditions, this command emits ``unavailable`` and preserves the legacy rate
only as diagnostic provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from iac_new.metric_evidence_contract import validate_metric_evidence_packet


REQUIRED_CONDITIONS = (
    "baseline",
    "future_perturbed",
    "future_perturbed_pathway_blocked",
    "future_fixed_action_pathway_control",
)


def build(summary: dict[str, Any]) -> dict[str, Any]:
    model_id = str(summary.get("model_id") or summary.get("model") or "unknown")
    packet: dict[str, Any] = {
        "source_key": f"aggregate:{model_id}",
        "evidence_status": "unavailable",
        "coverage": float(summary.get("coverage") or 0.0),
        "failure_reason": "paired_future_only_intervention_missing",
        "native_action_source": (summary.get("action_sources") or [None])[0],
        "legacy_rollout_diagnostic": {
            "task_success_rate": summary.get("task_success_rate"),
            "scored_rows": summary.get("scored_rows", summary.get("scored_rows")),
            "independent_realized_state": summary.get("independent_realized_state"),
            "action_injection_verified": summary.get("action_injection_verified"),
        },
        "required_paired_conditions": list(REQUIRED_CONDITIONS),
        "intervention": {"paired": False, "type": "future_only_required"},
    }
    check = validate_metric_evidence_packet("FCS", packet)
    return {
        "protocol": "iac-fcs-evidence-packet-builder-v1",
        "metric_id": "FCS",
        "model_id": model_id,
        "packet": packet,
        "validation": check,
        "claim_boundary": "Independent rollout diagnostic only until paired future-only intervention is present.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["validation"]["status"], "missing": report["validation"]["missing"]}, indent=2))


if __name__ == "__main__":
    main()

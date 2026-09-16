#!/usr/bin/env python3
"""Audit raw WAM artifacts against the future-to-action mediation contract.

This is deliberately an audit, not a scorer.  Existing intervention outputs
must not be relabelled as the four preregistered conditions merely because
they contain a native action head and a future perturbation.  The audit
reports what is present and names the exact missing controls.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_CONDITIONS = (
    "baseline",
    "future_perturbed",
    "future_perturbed_pathway_blocked",
    "future_fixed_action_pathway_control",
)


def _load(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    raise ValueError(f"unsupported JSON payload: {path}")


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


def audit(
    native_rows: list[dict[str, Any]],
    level2_rows: list[dict[str, Any]],
    swap_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rows = native_rows + level2_rows + swap_rows
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        group = str(row.get("counterfactual_group_id") or row.get("source_key") or "")
        if group:
            groups[group].append(row)

    observed_condition_values = sorted({
        str(row["condition"])
        for row in all_rows
        if row.get("condition") not in (None, "")
    })
    observed_pathway_states = sorted({
        str(row["pathway_state"])
        for row in all_rows
        if row.get("pathway_state") not in (None, "")
    })
    observed_branch_roles = sorted({
        str(row["branch_role"])
        for row in all_rows
        if row.get("branch_role") not in (None, "")
    })
    source_keys = sorted({
        str(row.get("source_key"))
        for row in all_rows
        if row.get("source_key") not in (None, "")
    })

    native_provenance = {
        "native_action_head_recorded_true": sum(row.get("native_action_head_recorded") is True for row in native_rows),
        "formal_foresight_mediation_input_true": sum(row.get("formal_foresight_mediation_input") is True for row in native_rows),
        "native_unintervened_true": sum(row.get("native_unintervened") is True for row in native_rows),
        "action_origins": _counter(native_rows, "action_origin"),
        "intervention_types": _counter(native_rows, "intervention_type"),
        "intervention_targets": _counter(native_rows, "intervention_target"),
    }

    required_presence = {
        condition: condition in observed_condition_values
        for condition in REQUIRED_CONDITIONS
    }
    missing_controls = [condition for condition, present in required_presence.items() if not present]
    missing_fields = {
        field: sum(field not in row or row.get(field) in (None, "") for row in all_rows)
        for field in (
            "future_fingerprint",
            "pathway_state",
            "action_normalization_fingerprint",
            "action_normalization_scale",
            "command_fingerprint",
            "model_revision",
        )
    }
    native_source_disjoint = len({
        str(row.get("source_key"))
        for row in native_rows
        if row.get("source_key") not in (None, "")
    })

    # A raw artifact with only native/reverse roles is an intervention pilot,
    # not one of the four conditions.  In particular, image-swap controls do
    # not establish that the action pathway was blocked or fixed.
    status = "ready_for_formal_scoring" if not missing_controls and not any(missing_fields.values()) else "blocked_missing_controls"
    return {
        "protocol": "iac-future-to-action-mediation-v1",
        "audit_type": "raw_artifact_readiness",
        "status": status,
        "claim_enabled": False,
        "inputs": {
            "native_rows": len(native_rows),
            "level2_rows": len(level2_rows),
            "image_swap_rows": len(swap_rows),
            "combined_rows": len(all_rows),
            "counterfactual_groups": len(groups),
        },
        "observed": {
            "branch_roles": observed_branch_roles,
            "condition_values": observed_condition_values,
            "pathway_states": observed_pathway_states,
            "source_key_count": len(source_keys),
            "source_keys": source_keys,
            "native_provenance": native_provenance,
        },
        "required_conditions": list(REQUIRED_CONDITIONS),
        "required_condition_presence": required_presence,
        "missing_controls": missing_controls,
        "missing_required_fields": missing_fields,
        "source_disjoint_confirmation": {
            "source_key_count_observed": native_source_disjoint,
            "calibration_manifest_supplied": False,
            "verified": False,
        },
        "interpretation": {
            "what_is_supported": "The pilot contains a model-owned native action head and a future latent permutation intervention, so it is useful for an action-response diagnostic.",
            "what_is_not_supported": "It does not identify future-to-action mediation because no pathway-blocked and fixed-action controls are present, and no frozen action normalization/future fingerprints are recorded.",
            "image_swap_is_not_pathway_block": True,
        },
        "next_experiment": {
            "minimum_confirmation_sources": 30,
            "conditions": list(REQUIRED_CONDITIONS),
            "hold_fixed": ["source_key", "history", "command", "nuisance_seed", "model_revision", "wam_model_id", "native_action_source"],
            "calibration_rule": "fit action normalization on calibration sources only, freeze before confirmation, and provide a disjoint source manifest",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-actions", type=Path, required=True)
    parser.add_argument("--level2", type=Path, required=True)
    parser.add_argument("--level2-swap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(_load(args.native_actions), _load(args.level2), _load(args.level2_swap))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "groups": result["inputs"]["counterfactual_groups"],
        "missing_controls": result["missing_controls"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

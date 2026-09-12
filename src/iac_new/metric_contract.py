"""Runtime checks for cross-model directional-yaw metric comparability."""

from __future__ import annotations

import json
from typing import Any


EXPECTED_MEASUREMENT_DIMENSIONS = {
    "yaw_direction": "validated",
    "speed": "diagnostic_only",
    "longitudinal_distance": "diagnostic_only",
    "lateral_displacement": "diagnostic_only",
    "curvature": "diagnostic_only",
    "metric_trajectory": "unavailable",
}


def validate_directional_yaw_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the frozen, model-comparable contract of MAS/RCS yaw configs.

    A model adapter may only resolve coordinate handedness.  It must not alter
    the descriptor, deadbands, aggregation, promotion gates, or bootstrap unit.
    This function is intentionally dependency-free so it can run in CI and on
    an evaluation server before any private data is mounted.
    """
    metric_id = str(config.get("metric_id") or "")
    if metric_id not in {"MAS", "RCS"}:
        raise ValueError("metric_id must be MAS or RCS")
    representation = config.get("representation")
    adapter = config.get("adapter")
    contract = config.get("comparability_contract")
    aggregation = config.get("aggregation")
    promotion = config.get("promotion_criteria")
    if not all(isinstance(value, dict) for value in (representation, adapter, contract, aggregation, promotion)):
        raise ValueError("directional-yaw config requires representation, adapter, comparability_contract, aggregation, and promotion_criteria")
    if representation.get("candidate_blind") is not True:
        raise ValueError("directional-yaw metric must be candidate_blind")
    if representation.get("metric_reconstruction_used") is not False:
        raise ValueError("directional-yaw metric cannot use metric reconstruction")
    if representation.get("output_domain") != "ordinal_direction_only":
        raise ValueError("directional-yaw output_domain must be ordinal_direction_only")
    if config.get("measurement_dimensions") != EXPECTED_MEASUREMENT_DIMENSIONS:
        raise ValueError("directional-yaw measurement dimensions must keep non-yaw domains diagnostic or unavailable")
    if adapter.get("orientation") not in (-1, 1):
        raise ValueError("adapter orientation must be -1 or 1")
    if adapter.get("frozen_before_confirmation") is not True:
        raise ValueError("adapter must be frozen before confirmation")
    if not str(adapter.get("calibration_contract_id") or ""):
        raise ValueError("adapter calibration_contract_id is required")
    allowed = set(contract.get("model_specific_fields_allowed") or [])
    if allowed != {"orientation"}:
        raise ValueError("only orientation may be model-specific")
    forbidden = set(contract.get("model_specific_fields_forbidden") or [])
    if not {"promotion_criteria", "aggregation"}.issubset(forbidden):
        raise ValueError("promotion_criteria and aggregation must be model-invariant")
    if contract.get("calibration_must_be_source_disjoint_from_confirmation") is not True:
        raise ValueError("calibration must be source-disjoint from confirmation")
    if float(promotion.get("pair_coverage_min", 0.0)) < 0.9:
        raise ValueError("pair coverage gate must be at least 0.90")
    if int(promotion.get("minimum_models", 0)) < 2:
        raise ValueError("promotion requires at least two models")
    bootstrap = aggregation.get("bootstrap")
    if not isinstance(bootstrap, dict) or bootstrap.get("unit") != "source_key":
        raise ValueError("bootstrap unit must be source_key")
    if config.get("missing_value_policy") != "unavailable_never_zero_fill":
        raise ValueError("missing value policy must be unavailable_never_zero_fill")
    return {
        "metric_id": metric_id,
        "status": "valid",
        "adapter_model_specific_fields": ["orientation"],
        "shared_descriptor": contract.get("shared_descriptor"),
        "calibration_source_disjoint": True,
        "calibration_contract_id": str(adapter["calibration_contract_id"]),
    }


def validate_directional_yaw_config_set(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate both individual configs and their shared frozen contract.

    MAS and RCS are different estimands, so their action thresholds may differ,
    but the image descriptor, bootstrap unit, missing-value policy, calibration
    contract, and promotion policy must remain comparable.  This check catches
    a common failure mode where every adapter passes a local schema check while
    the published columns were calibrated under different rules.
    """
    if not configs:
        raise ValueError("at least one directional-yaw config is required")
    reports = [validate_directional_yaw_config(config) for config in configs]
    invariant_fields = {
        "shared_descriptor": [report["shared_descriptor"] for report in reports],
        "calibration_contract_id": [report["calibration_contract_id"] for report in reports],
    }
    mismatches = {
        name: values for name, values in invariant_fields.items() if len(set(values)) != 1
    }
    bootstrap = []
    promotion = []
    for config in configs:
        bootstrap.append(config["aggregation"]["bootstrap"])
        promotion.append(config["promotion_criteria"])
    if len({json.dumps(value, sort_keys=True) for value in bootstrap}) != 1:
        mismatches["bootstrap"] = bootstrap
    if len({json.dumps(value, sort_keys=True) for value in promotion}) != 1:
        mismatches["promotion_criteria"] = promotion
    if mismatches:
        raise ValueError(f"directional-yaw config set is not comparable: {mismatches}")
    return {
        "status": "valid",
        "config_count": len(configs),
        "shared_descriptor": reports[0]["shared_descriptor"],
        "calibration_contract_id": reports[0]["calibration_contract_id"],
        "measurement_dimensions": dict(EXPECTED_MEASUREMENT_DIMENSIONS),
        "bootstrap": bootstrap[0],
        "promotion_criteria": promotion[0],
    }

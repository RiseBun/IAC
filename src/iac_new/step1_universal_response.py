"""Contract checks for the experimental, model-broad Step 1 response layer.

This layer deliberately measures ordinal visual response signatures instead of
recovering metres or a full SE(2) trajectory.  It is a protocol contract only:
individual channels remain unpromoted until their controls and held-out model
gates pass.
"""

from __future__ import annotations

from typing import Any


REQUIRED_CONTROLS = {
    "normal_action",
    "reversed_action",
    "identity_swap",
    "zero_contrast",
}

REQUIRED_CHANNELS = {
    "yaw_direction",
    "longitudinal_order",
    "lateral_direction",
    "temporal_persistence",
}

ALLOWED_ADAPTER_FIELDS = {
    "orientation",
    "image_geometry",
    "flow_backend_fingerprint",
}


def validate_universal_response_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the fail-closed contract for the experimental Step 1 layer.

    The validator does not promote a channel.  It only guarantees that a
    future experiment cannot silently turn a metric reconstruction or a model
    specific calibration into a supposedly universal MAS/RCS score.
    """
    if config.get("protocol") != "iac-step1-universal-response-v1":
        raise ValueError("wrong universal-response protocol id")
    if config.get("status") != "experimental_contract_not_promoted":
        raise ValueError("universal-response config must remain experimental")
    representation = config.get("representation")
    if not isinstance(representation, dict):
        raise ValueError("representation is required")
    if representation.get("candidate_blind") is not True:
        raise ValueError("response descriptors must be candidate blind")
    if representation.get("metric_reconstruction_used") is not False:
        raise ValueError("universal response cannot use metric reconstruction")
    if representation.get("output_domain") != "ordinal_response_signature":
        raise ValueError("output domain must be ordinal_response_signature")

    channels = representation.get("channels")
    if not isinstance(channels, dict) or set(channels) != REQUIRED_CHANNELS:
        raise ValueError("all four response channels must be declared")
    for name, channel in channels.items():
        if not isinstance(channel, dict):
            raise ValueError(f"channel {name} must be an object")
        if channel.get("validation_status") not in {"validated", "candidate", "unvalidated"}:
            raise ValueError(f"channel {name} has invalid validation status")
        if not str(channel.get("descriptor") or ""):
            raise ValueError(f"channel {name} requires a descriptor")

    controls = set(config.get("controls") or [])
    if controls != REQUIRED_CONTROLS:
        raise ValueError("normal/reversed/identity/zero controls are required")

    mas = config.get("mas")
    rcs = config.get("rcs")
    if not isinstance(mas, dict) or not isinstance(rcs, dict):
        raise ValueError("MAS and RCS contracts are required")
    if mas.get("unit") != "branch" or rcs.get("unit") != "same_source_twin":
        raise ValueError("MAS must be branch-level and RCS must be twin-level")
    if mas.get("candidate_blind") is not True or rcs.get("candidate_blind") is not True:
        raise ValueError("MAS/RCS must be candidate blind")
    if rcs.get("common_mode_normalization") != "signed_difference_common_mode_cancellation":
        raise ValueError("RCS requires signed differences after common-mode cancellation")
    if rcs.get("temporal_requirement") != "signed_persistence_across_intervals":
        raise ValueError("RCS requires signed temporal persistence")

    adapter = config.get("adapter_policy")
    if not isinstance(adapter, dict):
        raise ValueError("adapter_policy is required")
    allowed = set(adapter.get("model_specific_fields_allowed") or [])
    if not allowed.issubset(ALLOWED_ADAPTER_FIELDS):
        raise ValueError("adapter contains an unapproved model-specific field")
    if adapter.get("calibration_source") != "logged_real_only":
        raise ValueError("calibration must use logged real videos only")
    if adapter.get("calibration_source_disjoint") is not True:
        raise ValueError("calibration must be source-disjoint")
    if adapter.get("frozen_before_confirmation") is not True:
        raise ValueError("adapter must be frozen before confirmation")

    gates = config.get("promotion_gates")
    if not isinstance(gates, dict):
        raise ValueError("promotion_gates are required")
    if float(gates.get("coverage_min", 0.0)) < 0.9:
        raise ValueError("coverage gate must be at least 0.90")
    if float(gates.get("direction_ci95_lower_min", 0.0)) < 0.75:
        raise ValueError("direction gate must be at least 0.75")
    if int(gates.get("minimum_architectures", 0)) < 3:
        raise ValueError("universal response requires at least three architectures")
    if gates.get("control_gate_required") is not True:
        raise ValueError("control gate is required")

    return {
        "status": "valid_experimental_contract",
        "channels": sorted(REQUIRED_CHANNELS),
        "controls": sorted(REQUIRED_CONTROLS),
        "model_specific_fields_allowed": sorted(allowed),
        "promotion_gates": dict(gates),
    }

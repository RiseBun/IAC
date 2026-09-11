"""Pairwise ordinal scoring for candidate-blind flow-structure measurements."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 3:
        return None
    first_scale = max(1.0, float(np.max(np.abs(first))))
    second_scale = max(1.0, float(np.max(np.abs(second))))
    # Producers commonly integrate float32 controls.  Nominally identical
    # interventions can therefore differ by ~1e-8 after serialization.  That
    # is not rank variation and must not produce a spurious Spearman value.
    if (
        np.ptp(first) <= 1e-6 * first_scale
        or np.ptp(second) <= 1e-6 * second_scale
    ):
        return None
    return float(np.corrcoef(_rankdata(first), _rankdata(second))[0, 1])


def _wilson(hits: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    probability = hits / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [float(center - half), float(center + half)]


def _bootstrap_spearman_ci(
    first: np.ndarray,
    second: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> list[float] | None:
    if _spearman(first, second) is None or draws <= 0:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        index = rng.integers(0, len(first), size=len(first))
        value = _spearman(first[index], second[index])
        if value is not None:
            values.append(value)
    if not values:
        return None
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _descriptor_values(record: dict[str, Any], descriptor: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for row in record["flow_structure"]["rows"]:
        if not row.get("input_available"):
            continue
        value = row.get(descriptor)
        if value is not None and np.isfinite(float(value)):
            result[int(row["interval_index"])] = float(value)
    return result


def _trajectory_path_length(trajectory: np.ndarray) -> float:
    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 2 or not len(value):
        raise ValueError("action trajectory must have shape [T,>=2]")
    if not np.all(np.isfinite(value[:, :2])):
        raise ValueError("action trajectory positions must be finite")
    positions = np.vstack([np.zeros((1, 2), dtype=np.float64), value[:, :2]])
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def score_flow_structure_pairs(
    measurements: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    *,
    descriptor: str = "horizontal_flow_center",
    orientation: int = 1,
    action_column: int = 1,
    action_reference: str = "endpoint_column",
    minimum_common_intervals: int = 2,
    minimum_action_delta: float = 0.05,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 6101,
) -> dict[str, Any]:
    """Score left/right response ordering without metric trajectory reconstruction."""
    if orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or 1")
    if action_column not in (0, 1, 2):
        raise ValueError("action_column must be 0, 1 or 2")
    if action_reference not in {"endpoint_column", "trajectory_path_length"}:
        raise ValueError("action_reference must be endpoint_column or trajectory_path_length")
    measured = {
        (str(row.get("source_key") or ""), str(row.get("branch_role") or "")): row
        for row in measurements
    }
    inputs = {
        (str(row.get("source_key") or ""), str(row.get("branch_role") or "")): row
        for row in manifests
    }
    sources = sorted(
        {source for source, role in measured if role == "left"}
        & {source for source, role in measured if role == "right"}
    )
    pairs: list[dict[str, Any]] = []
    for source in sources:
        left = measured[(source, "left")]
        right = measured[(source, "right")]
        if left.get("candidate_bank_used_by_measurement") is not False:
            raise ValueError(f"{source}: left measurement is not candidate-blind")
        if right.get("candidate_bank_used_by_measurement") is not False:
            raise ValueError(f"{source}: right measurement is not candidate-blind")
        left_values = _descriptor_values(left, descriptor)
        right_values = _descriptor_values(right, descriptor)
        common = sorted(set(left_values) & set(right_values))
        row: dict[str, Any] = {
            "source_key": source,
            "stratum": str(left.get("stratum") or "unknown"),
            "common_intervals": common,
            "common_interval_count": len(common),
        }
        if len(common) < int(minimum_common_intervals):
            row.update({"status": "unavailable", "reason": "insufficient_common_intervals"})
            pairs.append(row)
            continue
        left_action = np.asarray(inputs[(source, "left")]["action_trajectory"], dtype=np.float64)
        right_action = np.asarray(inputs[(source, "right")]["action_trajectory"], dtype=np.float64)
        flow_delta = float(np.median([
            left_values[index] - right_values[index] for index in common
        ]))
        if action_reference == "trajectory_path_length":
            action_delta = _trajectory_path_length(left_action) - _trajectory_path_length(right_action)
        else:
            action_delta = float(left_action[-1, action_column] - right_action[-1, action_column])
        row.update({
            "status": "scored",
            "flow_structure_delta": float(orientation * flow_delta),
            "action_delta": action_delta,
            "command_label_direction_match": bool(orientation * flow_delta > 0.0),
            "action_direction_comparable": bool(abs(action_delta) > minimum_action_delta),
            "action_direction_match": (
                bool(np.sign(orientation * flow_delta) == np.sign(action_delta))
                if abs(action_delta) > minimum_action_delta and abs(flow_delta) > 1e-12
                else None
            ),
        })
        pairs.append(row)
    scored = [row for row in pairs if row["status"] == "scored"]
    comparable = [row for row in scored if row["action_direction_match"] is not None]
    direction_hits = sum(bool(row["action_direction_match"]) for row in comparable)
    command_hits = sum(bool(row["command_label_direction_match"]) for row in scored)
    flow_delta = np.asarray([row["flow_structure_delta"] for row in scored], dtype=np.float64)
    action_delta = np.asarray([row["action_delta"] for row in scored], dtype=np.float64)
    rank_correlation = _spearman(flow_delta, action_delta)
    return {
        "protocol": "iac-flow-structure-alignment-v1",
        "metric_reconstruction_used": False,
        "descriptor": descriptor,
        "orientation": orientation,
        "action_column": action_column,
        "action_reference": action_reference,
        "minimum_common_intervals": int(minimum_common_intervals),
        "minimum_action_delta": float(minimum_action_delta),
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_seed": int(bootstrap_seed),
        "pair_count": len(pairs),
        "status_counts": dict(Counter(row["status"] for row in pairs)),
        "coverage": len(scored) / len(pairs) if pairs else None,
        "action_response_spearman": rank_correlation,
        "action_response_spearman_ci95": _bootstrap_spearman_ci(
            flow_delta,
            action_delta,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        ),
        "action_direction_pairs": len(comparable),
        "action_direction_hits": direction_hits,
        "action_direction_accuracy": direction_hits / len(comparable) if comparable else None,
        "action_direction_accuracy_ci95": _wilson(direction_hits, len(comparable)),
        "command_label_direction_hits": command_hits,
        "command_label_direction_accuracy": command_hits / len(scored) if scored else None,
        "command_label_direction_accuracy_ci95": _wilson(command_hits, len(scored)),
        "warning": "This scores command/action response, not logged-GT fidelity or policy quality.",
        "pairs": pairs,
    }


def score_counterfactual_structure_pairs(
    measurements: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    *,
    descriptor: str = "median_flow_magnitude_px",
    orientation: int = 1,
    action_reference: str = "trajectory_path_length",
    action_column: int = 2,
    minimum_common_intervals: int = 2,
    minimum_action_delta: float = 0.5,
    minimum_common_mode: float = 1e-6,
) -> dict[str, Any]:
    """Score same-source counterfactual motion differences.

    This is intentionally separate from :func:`score_flow_structure_pairs`:
    it reports a branch contrast rather than an absolute motion estimate.  The
    common-mode normalized contrast is bounded by the observed branch motion
    and is only used for diagnostics; direction and ordering are determined
    from the raw, candidate-blind descriptor difference.
    """
    if orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or 1")
    if action_reference not in {"endpoint_column", "trajectory_path_length"}:
        raise ValueError("action_reference must be endpoint_column or trajectory_path_length")
    if action_column not in (0, 1, 2):
        raise ValueError("action_column must be 0, 1 or 2")
    measured = {
        (str(row.get("source_key") or ""), str(row.get("branch_role") or "")): row
        for row in measurements
    }
    inputs = {
        (str(row.get("source_key") or ""), str(row.get("branch_role") or "")): row
        for row in manifests
    }
    sources = sorted(
        {source for source, role in measured if role == "left"}
        & {source for source, role in measured if role == "right"}
    )
    pairs: list[dict[str, Any]] = []
    for source in sources:
        left = measured[(source, "left")]
        right = measured[(source, "right")]
        if left.get("candidate_bank_used_by_measurement") is not False:
            raise ValueError(f"{source}: left measurement is not candidate-blind")
        if right.get("candidate_bank_used_by_measurement") is not False:
            raise ValueError(f"{source}: right measurement is not candidate-blind")
        left_values = _descriptor_values(left, descriptor)
        right_values = _descriptor_values(right, descriptor)
        common = sorted(set(left_values) & set(right_values))
        row: dict[str, Any] = {
            "source_key": source,
            "stratum": str(left.get("stratum") or "unknown"),
            "descriptor": descriptor,
            "common_intervals": common,
            "common_interval_count": len(common),
        }
        if len(common) < int(minimum_common_intervals):
            row.update({"status": "unavailable", "reason": "insufficient_common_intervals"})
            pairs.append(row)
            continue
        left_action = np.asarray(inputs[(source, "left")]["action_trajectory"], dtype=np.float64)
        right_action = np.asarray(inputs[(source, "right")]["action_trajectory"], dtype=np.float64)
        if action_reference == "trajectory_path_length":
            action_delta = _trajectory_path_length(left_action) - _trajectory_path_length(right_action)
        else:
            action_delta = float(left_action[-1, action_column] - right_action[-1, action_column])
        raw_delta = np.asarray(
            [orientation * (left_values[index] - right_values[index]) for index in common],
            dtype=np.float64,
        )
        common_mode = np.asarray(
            [0.5 * (left_values[index] + right_values[index]) for index in common],
            dtype=np.float64,
        )
        common_scale = np.maximum(np.abs(common_mode), float(minimum_common_mode))
        normalized_delta = raw_delta / common_scale
        aggregate_delta = float(np.median(raw_delta))
        aggregate_normalized = float(np.median(normalized_delta))
        action_comparable = bool(abs(action_delta) >= float(minimum_action_delta))
        nonzero = np.abs(raw_delta) > 1e-12
        signs = np.sign(raw_delta[nonzero])
        temporal_persistence = (
            float(max(np.mean(signs > 0), np.mean(signs < 0))) if len(signs) else None
        )
        row.update({
            "status": "scored",
            "flow_delta_by_interval": raw_delta.tolist(),
            "common_mode_by_interval": common_mode.tolist(),
            "normalized_delta_by_interval": normalized_delta.tolist(),
            "flow_delta": aggregate_delta,
            "normalized_flow_delta": aggregate_normalized,
            "action_delta": float(action_delta),
            "action_direction_comparable": action_comparable,
            "temporal_persistence": temporal_persistence,
            "action_direction_match": (
                bool(np.sign(aggregate_delta) == np.sign(action_delta))
                if action_comparable and abs(aggregate_delta) > 1e-12 else None
            ),
        })
        pairs.append(row)
    scored = [row for row in pairs if row["status"] == "scored"]
    comparable = [row for row in scored if row["action_direction_match"] is not None]
    hits = sum(bool(row["action_direction_match"]) for row in comparable)
    action = np.asarray([row["action_delta"] for row in scored], dtype=np.float64)
    response = np.asarray([row["flow_delta"] for row in scored], dtype=np.float64)
    normalized = np.asarray([row["normalized_flow_delta"] for row in scored], dtype=np.float64)
    return {
        "protocol": "iac-counterfactual-flow-structure-delta-v1",
        "metric_reconstruction_used": False,
        "candidate_blind": True,
        "descriptor": descriptor,
        "orientation": orientation,
        "action_reference": action_reference,
        "action_column": action_column,
        "minimum_common_intervals": int(minimum_common_intervals),
        "minimum_action_delta": float(minimum_action_delta),
        "pair_count": len(pairs),
        "status_counts": dict(Counter(row["status"] for row in pairs)),
        "coverage": len(scored) / len(pairs) if pairs else None,
        "action_response_spearman": _spearman(response, action),
        "normalized_action_response_spearman": _spearman(normalized, action),
        "action_direction_pairs": len(comparable),
        "action_direction_hits": hits,
        "action_direction_accuracy": hits / len(comparable) if comparable else None,
        "action_direction_accuracy_ci95": _wilson(hits, len(comparable)),
        "median_temporal_persistence": (
            float(np.median([row["temporal_persistence"] for row in scored if row["temporal_persistence"] is not None]))
            if any(row["temporal_persistence"] is not None for row in scored) else None
        ),
        "warning": "This is an exploratory counterfactual response score, not metric distance or logged-GT fidelity.",
        "pairs": pairs,
    }

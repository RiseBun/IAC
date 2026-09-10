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


def score_flow_structure_pairs(
    measurements: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    *,
    descriptor: str = "horizontal_flow_center",
    orientation: int = 1,
    action_column: int = 1,
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

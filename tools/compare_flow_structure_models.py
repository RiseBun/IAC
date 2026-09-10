#!/usr/bin/env python3
"""Compare two WAMs on matched flow-structure sources and actions."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr


def _rho(action: np.ndarray, response: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        value = float(spearmanr(action, response).statistic)
    return value


def compare_reports(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    first_name: str,
    second_name: str,
    minimum_action_delta: float = 0.01,
    bootstrap_draws: int = 10000,
    bootstrap_seed: int = 6102,
) -> dict[str, Any]:
    first_pairs = {row["source_key"]: row for row in first["pairs"]}
    second_pairs = {row["source_key"]: row for row in second["pairs"]}
    source_keys = sorted(set(first_pairs) & set(second_pairs))
    if not source_keys:
        raise ValueError("model reports have no common source_key")

    def action_values(rows: dict[str, dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                float(rows[key]["action_delta"])
                if rows[key].get("action_delta") is not None else np.nan
                for key in source_keys
            ],
            dtype=np.float64,
        )

    action = action_values(first_pairs)
    second_action = action_values(second_pairs)
    action_comparable = np.isfinite(action) & np.isfinite(second_action)
    if not np.allclose(
        action[action_comparable],
        second_action[action_comparable],
        rtol=0.0,
        atol=1e-6,
    ):
        mismatch = float(
            np.max(np.abs(action[action_comparable] - second_action[action_comparable]))
        )
        raise ValueError(f"matched model actions differ; max absolute delta={mismatch}")

    first_scored = np.asarray(
        [first_pairs[key]["status"] == "scored" for key in source_keys], dtype=bool
    )
    second_scored = np.asarray(
        [second_pairs[key]["status"] == "scored" for key in source_keys], dtype=bool
    )
    common_scored = first_scored & second_scored & action_comparable
    material = action_comparable & (np.abs(action) >= float(minimum_action_delta))
    common_material = common_scored & material
    if common_material.sum() < 2 or common_scored.sum() < 3:
        raise ValueError("matched reports contain too few common scored/material pairs")

    def hits(rows: dict[str, dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [bool(rows[key].get("action_direction_match")) for key in source_keys],
            dtype=bool,
        )

    first_hits, second_hits = hits(first_pairs), hits(second_pairs)
    def response_values(rows: dict[str, dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                float(rows[key]["flow_structure_delta"])
                if rows[key].get("flow_structure_delta") is not None else np.nan
                for key in source_keys
            ],
            dtype=np.float64,
        )

    first_response = response_values(first_pairs)
    second_response = response_values(second_pairs)
    material_index = np.flatnonzero(common_material)
    scored_index = np.flatnonzero(common_scored)
    first_rho = _rho(action[common_scored], first_response[common_scored])
    second_rho = _rho(action[common_scored], second_response[common_scored])

    rng = np.random.default_rng(bootstrap_seed)
    coverage_delta: list[float] = []
    direction_delta: list[float] = []
    rho_delta: list[float] = []
    for _ in range(bootstrap_draws):
        sampled = rng.integers(0, len(source_keys), len(source_keys))
        coverage_delta.append(
            float(np.mean(second_scored[sampled]) - np.mean(first_scored[sampled]))
        )
        sampled_material = material_index[
            rng.integers(0, len(material_index), len(material_index))
        ]
        direction_delta.append(
            float(
                np.mean(second_hits[sampled_material])
                - np.mean(first_hits[sampled_material])
            )
        )
        sampled_scored = scored_index[
            rng.integers(0, len(scored_index), len(scored_index))
        ]
        first_sample_rho = _rho(action[sampled_scored], first_response[sampled_scored])
        second_sample_rho = _rho(action[sampled_scored], second_response[sampled_scored])
        if np.isfinite(first_sample_rho) and np.isfinite(second_sample_rho):
            rho_delta.append(second_sample_rho - first_sample_rho)

    def interval(values: list[float]) -> list[float]:
        return np.quantile(np.asarray(values), [0.025, 0.975]).tolist()

    return {
        "protocol": "iac-flow-structure-matched-model-comparison-v1",
        "status": "descriptive_posthoc_not_promotion",
        "first_model": first_name,
        "second_model": second_name,
        "common_source_count": len(source_keys),
        "exact_action_match_atol": 1e-6,
        "exact_action_match_count": int(action_comparable.sum()),
        "minimum_action_delta": minimum_action_delta,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
        "coverage": {
            first_name: float(np.mean(first_scored)),
            second_name: float(np.mean(second_scored)),
            "second_minus_first": float(np.mean(second_scored) - np.mean(first_scored)),
            "delta_ci95": interval(coverage_delta),
        },
        "common_scored_pairs": int(common_scored.sum()),
        "common_material_pairs": int(common_material.sum()),
        "direction_accuracy": {
            first_name: float(np.mean(first_hits[common_material])),
            second_name: float(np.mean(second_hits[common_material])),
            "second_minus_first": float(
                np.mean(second_hits[common_material])
                - np.mean(first_hits[common_material])
            ),
            "delta_ci95": interval(direction_delta),
            "discordant_second_only": int(
                np.sum(second_hits[common_material] & ~first_hits[common_material])
            ),
            "discordant_first_only": int(
                np.sum(first_hits[common_material] & ~second_hits[common_material])
            ),
        },
        "spearman_common_scored": {
            first_name: first_rho,
            second_name: second_rho,
            "second_minus_first": second_rho - first_rho,
            "delta_ci95": interval(rho_delta),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-name", required=True)
    parser.add_argument("--second-name", required=True)
    parser.add_argument("--minimum-action-delta", type=float, default=0.01)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=6102)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_reports(
        json.loads(args.first.read_text(encoding="utf-8")),
        json.loads(args.second.read_text(encoding="utf-8")),
        first_name=args.first_name,
        second_name=args.second_name,
        minimum_action_delta=args.minimum_action_delta,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

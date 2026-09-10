#!/usr/bin/env python3
"""Merge rigid-fit audit shards with source-clustered uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, spearmanr


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_index(paths: set[str]) -> dict[str, dict[str, Any]]:
    index = {}
    for value in paths:
        path = Path(value)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                index[str(row["sample_id"])] = row
    return index


def _source_type(row: dict[str, Any]) -> str:
    return "real" if row.get("branch_role") == "real" else "generated"


def _cluster_bootstrap_mean(
    rows: list[dict[str, Any]], key: str, *, iterations: int = 4000
) -> list[float] | None:
    clusters: dict[str, list[float]] = {}
    for row in rows:
        value = float(row["audit"][key])
        clusters.setdefault(str(row.get("source_key") or row["sample_id"]), []).append(value)
    if not clusters:
        return None
    cluster_means = np.asarray([np.mean(values) for values in clusters.values()], dtype=np.float64)
    rng = np.random.default_rng(20260909)
    samples = rng.integers(0, len(cluster_means), size=(iterations, len(cluster_means)))
    means = np.mean(cluster_means[samples], axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _energy_summary(
    rows: list[dict[str, Any]], *, attempted: int, practical_margin: float
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted_rows": attempted,
        "available_rows": len(rows),
        "availability": len(rows) / attempted if attempted else None,
    }
    if not rows:
        return result
    fitted = np.asarray([row["audit"]["fitted_energy"] for row in rows], dtype=np.float64)
    zero = np.asarray([row["audit"]["zero_flow_energy"] for row in rows], dtype=np.float64)
    delta = fitted - zero
    delta[np.abs(delta) < 1e-9] = 0.0
    ci = _cluster_bootstrap_mean(rows, "delta_fitted_minus_zero")
    result.update({
        "source_clusters": len({str(row.get("source_key") or row["sample_id"]) for row in rows}),
        "fitted_energy": {
            "mean": float(np.mean(fitted)),
            "median": float(np.median(fitted)),
            "q25": float(np.quantile(fitted, 0.25)),
            "q75": float(np.quantile(fitted, 0.75)),
        },
        "zero_flow_energy": {
            "mean": float(np.mean(zero)),
            "median": float(np.median(zero)),
        },
        "delta_fitted_minus_zero": {
            "mean": float(np.mean(delta)),
            "median": float(np.median(delta)),
            "source_cluster_bootstrap_mean_95ci": ci,
        },
        "practical_margin": float(practical_margin),
        "meaningfully_better_rows": int(np.sum(delta < -practical_margin)),
        "meaningfully_better_fraction": float(np.mean(delta < -practical_margin)),
        "primary_success": bool(ci is not None and ci[1] < -practical_margin),
        "zero_motion_selected_rows": int(sum(
            np.max(np.abs(np.asarray(row["audit"]["trajectory"], dtype=np.float64))) < 1e-9
            for row in rows
        )),
    })
    return result


def _interval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    interval_ids = sorted({
        int(interval["interval_index"])
        for row in rows
        for interval in row["audit"]["by_interval"]
    })
    for interval_id in interval_ids:
        values = [
            interval
            for row in rows
            for interval in row["audit"]["by_interval"]
            if int(interval["interval_index"]) == interval_id
            and interval["fitted_energy"] is not None
        ]
        fitted = np.asarray([value["fitted_energy"] for value in values], dtype=np.float64)
        zero = np.asarray([value["zero_flow_energy"] for value in values], dtype=np.float64)
        projection = np.asarray([value["projected_weight_fraction"] for value in values], dtype=np.float64)
        result[str(interval_id)] = {
            "rows": len(values),
            "fitted_energy_mean": float(np.mean(fitted)) if len(values) else None,
            "zero_flow_energy_mean": float(np.mean(zero)) if len(values) else None,
            "delta_mean": float(np.mean(fitted - zero)) if len(values) else None,
            "projected_weight_fraction_mean": float(np.mean(projection)) if len(values) else None,
        }
    return result


def _all_intervals_have_source(row: dict[str, Any]) -> bool:
    intervals = row["audit"]["by_interval"]
    return bool(intervals) and all(float(value["source_weight"]) > 1e-6 for value in intervals)


def _pair_alignment(
    rows: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        role = row.get("branch_role")
        if role in {"left", "right"}:
            pairs.setdefault(str(row["source_key"]), {})[str(role)] = row
    definitions = {
        "lateral_y": (1, 0.05),
        "yaw": (2, 0.01),
        "longitudinal_x": (0, 0.10),
    }
    output: dict[str, Any] = {"complete_pairs": sum(set(pair) == {"left", "right"} for pair in pairs.values())}
    for name, (axis, minimum_action_delta) in definitions.items():
        action_values = []
        fitted_values = []
        for pair in pairs.values():
            if set(pair) != {"left", "right"}:
                continue
            left, right = pair["left"], pair["right"]
            left_action = np.asarray(manifest[str(left["sample_id"])]["action_trajectory"], dtype=np.float64)
            right_action = np.asarray(manifest[str(right["sample_id"])]["action_trajectory"], dtype=np.float64)
            action_delta = float(left_action[-1, axis] - right_action[-1, axis])
            if abs(action_delta) < minimum_action_delta:
                continue
            left_fitted = np.asarray(left["audit"]["trajectory"], dtype=np.float64)
            right_fitted = np.asarray(right["audit"]["trajectory"], dtype=np.float64)
            action_values.append(action_delta)
            fitted_values.append(float(left_fitted[-1, axis] - right_fitted[-1, axis]))
        action_array = np.asarray(action_values, dtype=np.float64)
        fitted_array = np.asarray(fitted_values, dtype=np.float64)
        ties = np.abs(fitted_array) < 1e-6
        agreement = (np.sign(action_array) == np.sign(fitted_array)) & ~ties
        if len(agreement):
            ci = binomtest(int(np.sum(agreement)), len(agreement), 0.5).proportion_ci(0.95)
            correlation = spearmanr(action_array, fitted_array).statistic if len(agreement) > 2 else None
            output[name] = {
                "minimum_action_delta": minimum_action_delta,
                "pairs": len(agreement),
                "direction_agreement": int(np.sum(agreement)),
                "direction_agreement_fraction_ties_as_failures": float(np.mean(agreement)),
                "exact_binomial_95ci": [float(ci.low), float(ci.high)],
                "fitted_ties": int(np.sum(ties)),
                "spearman": None if correlation is None else float(correlation),
                "median_abs_action_delta": float(np.median(np.abs(action_array))),
                "median_abs_fitted_delta": float(np.median(np.abs(fitted_array))),
            }
        else:
            output[name] = {"minimum_action_delta": minimum_action_delta, "pairs": 0}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--practical-margin", type=float, default=0.05)
    args = parser.parse_args()

    payloads = [_json(path) for path in args.inputs]
    manifest_paths = {value for payload in payloads for value in payload.get("manifests", [])}
    manifest = _manifest_index(manifest_paths)
    rows = [row for payload in payloads for row in payload.get("records", [])]
    errors = [error for payload in payloads for error in payload.get("errors", [])]
    attempted_rows = []
    for row in rows:
        attempted_rows.append({**manifest.get(str(row["sample_id"]), {}), **row})
    for error in errors:
        raw = manifest.get(str(error["sample_id"]), {})
        attempted_rows.append({**raw, **error, "unavailable": True})

    output: dict[str, Any] = {
        "protocol": "fixed-source-spatially-stratified-rigid-fit-v1-merged",
        "inputs": [str(path) for path in args.inputs],
        "forward_backward": sorted({bool(payload["forward_backward"]) for payload in payloads}),
        "practical_margin": float(args.practical_margin),
        "groups": {},
        "errors": errors,
    }
    for source_type in ("real", "generated"):
        group_rows = [row for row in rows if _source_type(row) == source_type]
        group_attempted = [row for row in attempted_rows if _source_type(row) == source_type]
        strata = sorted({(row.get("metadata") or {}).get("stratum") for row in group_attempted}, key=str)
        output["groups"][source_type] = {
            **_energy_summary(
                group_rows,
                attempted=len(group_attempted),
                practical_margin=float(args.practical_margin),
            ),
            "by_interval": _interval_summary(group_rows),
            "all_intervals_source_available": _energy_summary(
                [row for row in group_rows if _all_intervals_have_source(row)],
                attempted=len(group_attempted),
                practical_margin=float(args.practical_margin),
            ),
            "by_stratum": {
                str(stratum): _energy_summary(
                    [row for row in group_rows if row.get("stratum") == stratum],
                    attempted=sum(
                        (row.get("metadata") or {}).get("stratum") == stratum
                        for row in group_attempted
                    ),
                    practical_margin=float(args.practical_margin),
                )
                for stratum in strata
            },
        }
    generated_rows = [row for row in rows if _source_type(row) == "generated"]
    output["generated_pair_alignment_to_action"] = {
        "all": _pair_alignment(generated_rows, manifest),
        "by_stratum": {
            str(stratum): _pair_alignment(
                [row for row in generated_rows if row.get("stratum") == stratum], manifest
            )
            for stratum in sorted({row.get("stratum") for row in generated_rows}, key=str)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

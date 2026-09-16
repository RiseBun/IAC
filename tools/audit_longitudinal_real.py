"""Audit the frozen longitudinal visual probe on source-disjoint real NAVSIM.

This is an offline audit of existing AS probe outputs.  It never changes the
visual estimator or learns a new mapping.  It reports raw metric error,
ordinal confusion, tolerance sensitivity, quality-gate coverage, and
source-level bootstrap intervals.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


LABELS = ("stop", "very_short", "short", "medium", "long")


def _load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.extend(payload.get("rows", []))
    return rows


def _source(row: dict[str, Any]) -> str:
    if row.get("source_key"):
        return str(row["source_key"])
    return str(row.get("sample_id", "")).split("::", 1)[0]


def _bin_gap(predicted: str, reference: str) -> int | None:
    try:
        return abs(LABELS.index(predicted) - LABELS.index(reference))
    except ValueError:
        return None


def _intervals(rows: list[dict[str, Any]], manifest: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        metadata = (manifest or {}).get(str(row.get("sample_id")), {})
        stratum = (metadata.get("metadata") or {}).get("stratum", "unknown")
        reference_distances = row.get("progress", {}).get("action_interval_distance_m", [])
        for local_index, interval in enumerate(row.get("intervals", [])):
            quality = interval.get("geometry_quality") or {}
            raw_predicted = interval.get("distance_m_raw")
            raw_reference = interval.get("reference_distance_m")
            if raw_reference is None and local_index < len(reference_distances):
                raw_reference = reference_distances[local_index]
            if interval.get("status") == "uncertain":
                reasons = quality.get("reasons") or [interval.get("abstention_reason") or "unspecified"]
                output.append({
                    "source": _source(row), "stratum": stratum, "index": local_index,
                    "status": "uncertain", "reasons": reasons,
                    "predicted": float(raw_predicted) if raw_predicted is not None else None,
                    "reference": float(raw_reference) if raw_reference is not None else None,
                    "predicted_bin": interval.get("distance_bin"),
                    "reference_bin": interval.get("reference_distance_bin"),
                    "matches": quality.get("matches"),
                    "inlier_fraction": quality.get("inlier_fraction"),
                    "reprojection_px": quality.get("median_reprojection_error_px"),
                })
                continue
            predicted = raw_predicted
            reference = raw_reference
            # AS output stores the action interval only in the row-level
            # progress object; align it by interval index when necessary.
            if reference is None:
                distances = row.get("progress", {}).get("action_interval_distance_m", [])
                reference = reference_distances[local_index] if local_index < len(reference_distances) else None
            if predicted is None or reference is None:
                output.append({"source": _source(row), "stratum": stratum, "index": local_index, "status": "uncertain", "reasons": ["missing_raw_motion"]})
                continue
            output.append(
                {
                    "source": _source(row),
                    "stratum": stratum,
                    "index": local_index,
                    "status": str(interval.get("status", "scored")),
                    "predicted": float(predicted),
                    "reference": float(reference),
                    "predicted_bin": interval.get("distance_bin"),
                    "reference_bin": interval.get("reference_distance_bin"),
                    "abs_error": abs(float(predicted) - float(reference)),
                    "matches": (interval.get("geometry_quality") or {}).get("matches"),
                    "inlier_fraction": (interval.get("geometry_quality") or {}).get("inlier_fraction"),
                    "reprojection_px": (interval.get("geometry_quality") or {}).get("median_reprojection_error_px"),
                }
            )
    return output


def _accept(item: dict[str, Any], abs_tol: float, rel_tol: float, adjacent: int) -> bool:
    if item.get("status") == "uncertain":
        return False
    distance_ok = item["abs_error"] <= max(abs_tol, rel_tol * abs(item["reference"]))
    bin_ok = _bin_gap(item.get("predicted_bin"), item.get("reference_bin")) is not None and _bin_gap(item.get("predicted_bin"), item.get("reference_bin")) <= adjacent
    return bool(distance_ok or bin_ok)


def _bootstrap(values_by_source: dict[str, list[float]], seed: int, draws: int) -> list[float] | None:
    if not values_by_source:
        return None
    rng = np.random.default_rng(seed)
    sources = list(values_by_source)
    estimates: list[float] = []
    for _ in range(draws):
        chosen = rng.choice(sources, size=len(sources), replace=True)
        source_values = [value for source in chosen for value in values_by_source[source]]
        estimates.append(float(np.mean(source_values)) if source_values else float("nan"))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def audit(rows: list[dict[str, Any]], *, manifest: dict[str, dict[str, Any]] | None = None, seed: int = 20260916, draws: int = 5000) -> dict[str, Any]:
    items = _intervals(rows, manifest)
    scored = [item for item in items if item.get("status") != "uncertain"]
    errors = [item["abs_error"] for item in scored]
    predicted = np.asarray([item["predicted"] for item in scored], dtype=np.float64)
    reference = np.asarray([item["reference"] for item in scored], dtype=np.float64)
    spearman = _spearman(predicted, reference)
    confusion = Counter((item.get("reference_bin"), item.get("predicted_bin")) for item in scored)
    exact = sum(count for (a, b), count in confusion.items() if a == b)
    adjacent = sum(count for (a, b), count in confusion.items() if _bin_gap(a, b) is not None and _bin_gap(a, b) <= 1)
    by_stratum: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        subset = [item for item in scored if item.get("reference_bin") == label]
        by_stratum[label] = {
            "n_scored": len(subset),
            "exact_bin_accuracy": float(np.mean([item.get("predicted_bin") == label for item in subset])) if subset else None,
            "adjacent_bin_accuracy": (
                float(np.mean([
                    _bin_gap(item.get("predicted_bin"), label) is not None
                    and _bin_gap(item.get("predicted_bin"), label) <= 1
                    for item in subset
                ]))
                if subset else None
            ),
            "median_absolute_error_m": float(np.median([item["abs_error"] for item in subset])) if subset else None,
        }
    strata = {}
    for stratum in sorted({item.get("stratum", "unknown") for item in items}):
        subset = [item for item in items if item.get("stratum") == stratum]
        valid = [item for item in subset if item.get("status") != "uncertain"]
        strata[stratum] = {
            "n_intervals": len(subset),
            "n_scored": len(valid),
            "coverage": len(valid) / len(subset) if subset else None,
            "median_absolute_error_m": float(np.median([item["abs_error"] for item in valid])) if valid else None,
            "exact_bin_accuracy": float(np.mean([item.get("predicted_bin") == item.get("reference_bin") for item in valid])) if valid else None,
            "adjacent_bin_accuracy": float(np.mean([
                _bin_gap(item.get("predicted_bin"), item.get("reference_bin")) is not None
                and _bin_gap(item.get("predicted_bin"), item.get("reference_bin")) <= 1
                for item in valid
            ])) if valid else None,
        }
    tolerances = []
    for abs_tol, rel_tol, adj in ((0.0, 0.0, 0), (1.0, 0.25, 0), (2.0, 0.33, 1), (3.0, 0.50, 1), (4.0, 0.50, 1)):
        accepted = [_accept(item, abs_tol, rel_tol, adj) for item in scored]
        by_source: dict[str, list[float]] = defaultdict(list)
        for item, value in zip(scored, accepted):
            by_source[item["source"]].append(float(value))
        tolerances.append({
            "abs_tolerance_m": abs_tol,
            "relative_tolerance": rel_tol,
            "adjacent_bin_tolerance": adj,
            "accepted_rate_over_scored": float(np.mean(accepted)) if accepted else None,
            "accepted_rate_over_all_intervals": float(sum(accepted) / len(items)) if items else None,
            "source_bootstrap_ci95": _bootstrap(by_source, seed, draws),
        })
    source_errors: dict[str, list[float]] = defaultdict(list)
    for item in scored:
        source_errors[item["source"]].append(item["abs_error"])
    return {
        "protocol": "iac-longitudinal-real-audit-v1",
        "rows": len(rows),
        "source_count": len({_source(row) for row in rows}),
        "intervals_total": len(items),
        "intervals_scored": len(scored),
        "coverage": len(scored) / len(items) if items else None,
        "abstention_count": len(items) - len(scored),
        "abstention_reasons": dict(Counter(reason for item in items if item.get("status") == "uncertain" for reason in item.get("reasons", ["unspecified"]))),
        "mae_m": float(np.mean(errors)) if errors else None,
        "median_absolute_error_m": float(np.median(errors)) if errors else None,
        "spearman": spearman,
        "exact_bin_accuracy": exact / len(scored) if scored else None,
        "adjacent_bin_accuracy": adjacent / len(scored) if scored else None,
        "confusion": {f"{reference_bin}->{predicted_bin}": count for (reference_bin, predicted_bin), count in sorted(confusion.items())},
        "by_reference_bin": by_stratum,
        "by_scene_stratum": strata,
        "by_interval_index": {
            str(index): {
                "n_total": sum(item.get("index") == index for item in items),
                "n_scored": sum(item.get("index") == index and item.get("status") != "uncertain" for item in items),
                "coverage": sum(item.get("index") == index and item.get("status") != "uncertain" for item in items) / max(sum(item.get("index") == index for item in items), 1),
                "median_absolute_error_m": float(np.median([item["abs_error"] for item in scored if item.get("index") == index])) if any(item.get("index") == index for item in scored) else None,
            }
            for index in range(4)
        },
        "risk_coverage": [
            {
                "minimum_inlier_fraction": threshold,
                "coverage": len([item for item in items if item.get("predicted") is not None and item.get("reference") is not None and (item.get("inlier_fraction") or 0.0) >= threshold]) / len(items) if items else None,
                "median_absolute_error_m": (
                    float(np.median([
                        abs(item["predicted"] - item["reference"])
                        for item in items
                        if item.get("predicted") is not None and item.get("reference") is not None and (item.get("inlier_fraction") or 0.0) >= threshold
                    ]))
                    if any(item.get("predicted") is not None and item.get("reference") is not None and (item.get("inlier_fraction") or 0.0) >= threshold for item in items)
                    else None
                ),
            }
            for threshold in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
        ],
        "tolerance_sensitivity": tolerances,
        "source_bootstrap": {
            "mae_ci95": _bootstrap({key: values for key, values in source_errors.items()}, seed, draws),
            "unit": "source_key",
            "draws": draws,
        },
        "frozen_config": {"distance_abs_tolerance_m": 3.0, "distance_relative_tolerance": 0.5, "adjacent_bin_tolerance": 1},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="AS full4 JSON files or globs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", nargs="*", default=[], help="Optional JSONL manifests used to recover source strata")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    paths = sorted(path for pattern in args.input for path in glob.glob(pattern))
    if not paths:
        raise SystemExit("no input files matched")
    manifest: dict[str, dict[str, Any]] = {}
    for manifest_path in args.manifest:
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                manifest[str(item.get("sample_id"))] = item
    result = audit(_load_rows(paths), manifest=manifest, seed=args.seed, draws=args.draws)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("rows", "intervals_total", "intervals_scored", "coverage", "mae_m", "median_absolute_error_m", "spearman", "exact_bin_accuracy", "adjacent_bin_accuracy")}, indent=2))


if __name__ == "__main__":
    main()

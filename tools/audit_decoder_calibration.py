#!/usr/bin/env python3
"""Audit readout dispersion and lateral/yaw coupling on reference decoder runs.

The output is descriptive calibration evidence only.  It never fits or writes
calibration parameters and requires the reference manifest to match the
evaluated output sample ids.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


FIELDS = (("longitudinal", 0), ("lateral", 1), ("yaw", 2))


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_manifest_root", str(path.parent.resolve()))
            rows.append(row)
    return rows


def reference_trajectory(record: dict) -> np.ndarray:
    candidate_id = record.get("gt_candidate_id")
    for candidate in record.get("candidates", []):
        if str(candidate.get("candidate_id")) == str(candidate_id):
            return np.asarray(candidate["trajectory"], dtype=np.float64)
    raise KeyError(f"missing gt candidate for {record.get('sample_id')}")


def logged_gt_from_source_pickle(record: dict) -> np.ndarray:
    lineage = record.get("lineage") or {}
    source = Path(str(lineage.get("source_sample") or "")).expanduser()
    if not source.is_absolute():
        source = Path(record["_manifest_root"]) / source
    if not source.is_file():
        raise FileNotFoundError(f"missing lineage.source_sample for {record.get('sample_id')}: {source}")
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    future = payload.get("future_trajectory")
    if not isinstance(future, list) or not future:
        raise ValueError(f"{source}: future_trajectory is absent")
    trajectory = np.asarray([item["pose"] for item in future], dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] < 3 or not np.all(np.isfinite(trajectory[:, :3])):
        raise ValueError(f"{source}: future_trajectory poses must be finite [N,3]")
    metadata = payload.get("metadata") or {}
    realized = metadata.get("realized_future_ego_state")
    if realized is not None:
        realized_pose = np.asarray(realized, dtype=np.float64)
        if realized_pose.ndim != 2 or realized_pose.shape[0] != len(trajectory) or realized_pose.shape[1] < 3 or not np.allclose(
            realized_pose[:, :3], trajectory[:, :3], atol=1e-9, rtol=0.0
        ):
            raise ValueError(f"{source}: future_trajectory does not match realized future state")
    source_times = np.asarray(metadata.get("future_times_s") or [], dtype=np.float64)
    if source_times.shape != (len(trajectory),):
        source_times = 0.5 * np.arange(1, len(trajectory) + 1, dtype=np.float64)
    target_times = np.asarray(record.get("future_times_s") or [], dtype=np.float64)
    if target_times.ndim != 1 or not len(target_times):
        raise ValueError(f"{record.get('sample_id')}: future_times_s is required")
    timestamp_tolerance_s = 0.05
    if (
        target_times[0] < source_times[0] - timestamp_tolerance_s
        or target_times[-1] > source_times[-1] + timestamp_tolerance_s
    ):
        raise ValueError(f"{source}: logged GT does not cover requested future times")
    target_times = np.clip(target_times, source_times[0], source_times[-1])
    result = np.column_stack([
        np.interp(target_times, source_times, trajectory[:, 0]),
        np.interp(target_times, source_times, trajectory[:, 1]),
        np.interp(target_times, source_times, np.unwrap(trajectory[:, 2])),
    ])
    return result


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "iqr": None, "cv": None}
    array = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.quantile(array, [0.25, 0.50, 0.75])
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return {
        "n": int(len(array)),
        "median": float(median),
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
        "mean": mean,
        "sd": sd,
        "cv": float(sd / max(abs(mean), 1e-9)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
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


def signed_alignment(pairs: list[tuple[float, float]]) -> dict:
    if not pairs:
        return {"n": 0, "direction_accuracy": None, "spearman": None}
    values = np.asarray(pairs, dtype=np.float64)
    hits = int(np.sum(np.sign(values[:, 0]) == np.sign(values[:, 1])))
    z = 1.959963984540054
    probability = hits / len(values)
    denominator = 1.0 + z * z / len(values)
    center = (probability + z * z / (2.0 * len(values))) / denominator
    half = z * np.sqrt(
        probability * (1.0 - probability) / len(values)
        + z * z / (4.0 * len(values) * len(values))
    ) / denominator
    rho = None
    if len(values) >= 3 and np.ptp(values[:, 0]) > 0 and np.ptp(values[:, 1]) > 0:
        rho = float(np.corrcoef(rankdata(values[:, 0]), rankdata(values[:, 1]))[0, 1])
    return {
        "n": int(len(values)),
        "direction_hits": hits,
        "direction_accuracy": float(probability),
        "direction_accuracy_ci95": [float(center - half), float(center + half)],
        "spearman": rho,
    }


def error_summary(predicted: np.ndarray, reference: np.ndarray) -> dict[str, list[float]]:
    error = np.abs(predicted - reference)
    return {
        "mae": np.mean(error, axis=0).tolist(),
        "median_absolute_error": np.median(error, axis=0).tolist(),
    }


def stable_fold(source_key: str, folds: int = 5) -> int:
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", nargs="+", type=Path, required=True)
    parser.add_argument("--reference-manifest", nargs="+", type=Path)
    parser.add_argument("--sample-manifest", nargs="+", type=Path)
    parser.add_argument(
        "--reference-source",
        choices=("lineage_pickle", "manifest_gt_candidate"),
        default="lineage_pickle",
    )
    parser.add_argument("--join-key", choices=("sample_id", "source_key"), default="sample_id")
    parser.add_argument("--allow-missing-reference", action="store_true")
    parser.add_argument("--status", default="explained")
    parser.add_argument("--minimum-longitudinal-reference", type=float, default=1e-4)
    parser.add_argument("--minimum-lateral-reference", type=float, default=1e-4)
    parser.add_argument("--minimum-yaw-reference", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl([Path(path) for pattern in args.outputs for path in glob.glob(str(pattern))])
    if args.reference_source == "lineage_pickle":
        if not args.sample_manifest:
            parser.error("--sample-manifest is required for --reference-source lineage_pickle")
        reference_rows = read_jsonl(args.sample_manifest)
    else:
        if not args.reference_manifest:
            parser.error("--reference-manifest is required for manifest_gt_candidate")
        reference_rows = read_jsonl(args.reference_manifest)
    references = {str(row[args.join_key]): row for row in reference_rows}
    sample_sources = {}
    if args.sample_manifest:
        sample_sources = {
            str(row["sample_id"]): str(row.get("source_key") or row["sample_id"])
            for row in read_jsonl(args.sample_manifest)
        }
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    aligned: dict[str, list[tuple[float, float]]] = defaultdict(list)
    regression_x: list[list[float]] = []
    regression_y: list[list[float]] = []
    regression_strata: list[str] = []
    endpoints: list[tuple[np.ndarray, np.ndarray, int]] = []
    minimum_reference = {
        "longitudinal": float(args.minimum_longitudinal_reference),
        "lateral": float(args.minimum_lateral_reference),
        "yaw": float(args.minimum_yaw_reference),
    }
    missing = []
    for row in rows:
        if row.get("geometry_fit_status") != args.status:
            continue
        row_key = (
            sample_sources.get(str(row.get("sample_id")))
            if args.join_key == "source_key"
            else str(row.get("sample_id"))
        )
        reference = references.get(row_key)
        if reference is None:
            missing.append(row.get("sample_id"))
            continue
        gt = (
            logged_gt_from_source_pickle(reference)
            if args.reference_source == "lineage_pickle"
            else reference_trajectory(reference)
        )
        predicted = np.asarray(row["decoder"]["trajectory"], dtype=np.float64)
        stratum = str((reference.get("metadata") or {}).get("stratum", "unknown"))
        for name, index in FIELDS:
            denominator = abs(float(gt[-1, index]))
            if denominator >= minimum_reference[name]:
                grouped[stratum][name].append(abs(float(predicted[-1, index])) / denominator)
                aligned[name].append((
                    float(predicted[-1, index]),
                    float(gt[-1, index]),
                ))
        regression_x.append(gt[-1, [1, 2]].tolist())
        regression_y.append(predicted[-1, [1, 2]].tolist())
        regression_strata.append(stratum)
        endpoints.append((
            predicted[-1],
            gt[-1],
            stable_fold(str(reference.get("source_key", reference["sample_id"]))),
        ))

    if missing and not args.allow_missing_reference:
        raise SystemExit(f"{len(missing)} output ids are absent from the reference manifest")
    output: dict = {
        "protocol": "decoder-calibration-dispersion-audit-v1",
        "reference_source": args.reference_source,
        "reference_identity": (
            "lineage.source_sample pickle future_trajectory (NAVSIM logged realized ego trajectory)"
            if args.reference_source == "lineage_pickle"
            else "manifest gt_candidate_id"
        ),
        "status": args.status,
        "minimum_absolute_reference": minimum_reference,
        "n_rows": len(rows),
        "join_key": args.join_key,
        "matched_explained_rows": len(endpoints),
        "missing_reference_rows": len(missing),
        "groups": {
            stratum: {name: summarize(values) for name, values in fields.items()}
            for stratum, fields in sorted(grouped.items())
        },
    }
    all_values = defaultdict(list)
    for fields in grouped.values():
        for name, values in fields.items():
            all_values[name].extend(values)
    output["all"] = {name: summarize(values) for name, values in all_values.items()}
    output["signed_alignment"] = {
        name: signed_alignment(values) for name, values in aligned.items()
    }

    x = np.asarray(regression_x, dtype=np.float64)
    y = np.asarray(regression_y, dtype=np.float64)
    if len(x) >= 4:
        design = np.column_stack([np.ones(len(x)), x])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        rng = np.random.default_rng(7)
        bootstrap = []
        for _ in range(2000):
            indices = rng.integers(0, len(x), len(x))
            bootstrap.append(
                np.linalg.lstsq(design[indices], y[indices], rcond=None)[0][1:].T
            )
        output["lateral_yaw_regression"] = {
            "n": int(len(x)),
            "intercept": coefficients[0].tolist(),
            "matrix_rows_fitted_y_fitted_yaw_cols_true_y_true_yaw": coefficients[1:].T.tolist(),
            "matrix_bootstrap_95_ci": np.quantile(
                np.asarray(bootstrap), [0.025, 0.975], axis=0
            ).tolist(),
            "note": "descriptive OLS; off-diagonal terms are not a calibration fit",
        }
        by_stratum = {}
        for stratum in sorted(set(regression_strata)):
            mask = np.asarray([value == stratum for value in regression_strata])
            if int(mask.sum()) < 4:
                continue
            fit = np.linalg.lstsq(design[mask], y[mask], rcond=None)[0]
            by_stratum[stratum] = {
                "n": int(mask.sum()),
                "matrix_rows_fitted_y_fitted_yaw_cols_true_y_true_yaw": fit[1:].T.tolist(),
            }
        output["lateral_yaw_regression"]["by_stratum"] = by_stratum

    # Audit calibration only on held-out samples. The fitted parameters are
    # never exported to the runtime configuration by this tool.
    if len(endpoints) >= 10:
        predicted = np.asarray([item[0] for item in endpoints])
        reference = np.asarray([item[1] for item in endpoints])
        folds = np.asarray([item[2] for item in endpoints])
        held_out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        longitudinal_raw = []
        longitudinal_calibrated = []
        longitudinal_reference = []
        gains = []
        for fold in range(5):
            train = folds != fold
            test = folds == fold
            if not np.any(train) or not np.any(test):
                continue
            raw = predicted[test, 1:3]
            diagonal = np.zeros_like(raw)
            for index in range(2):
                train_design = np.column_stack([
                    np.ones(int(train.sum())), predicted[train, index + 1]
                ])
                fit = np.linalg.lstsq(
                    train_design, reference[train, index + 1], rcond=None
                )[0]
                diagonal[:, index] = np.column_stack([
                    np.ones(int(test.sum())), predicted[test, index + 1]
                ]) @ fit
            train_design = np.column_stack([
                np.ones(int(train.sum())), predicted[train, 1:3]
            ])
            fit = np.linalg.lstsq(train_design, reference[train, 1:3], rcond=None)[0]
            joint = np.column_stack([
                np.ones(int(test.sum())), predicted[test, 1:3]
            ]) @ fit
            for name, values in (("raw", raw), ("diagonal", diagonal), ("joint_2x2", joint)):
                held_out[name].append((values, reference[test, 1:3]))

            pred_x = predicted[train, 0]
            gain = float(pred_x @ reference[train, 0] / max(pred_x @ pred_x, 1e-9))
            gains.append(gain)
            longitudinal_raw.extend(predicted[test, 0].tolist())
            longitudinal_calibrated.extend((gain * predicted[test, 0]).tolist())
            longitudinal_reference.extend(reference[test, 0].tolist())
        output["five_fold_calibration_audit"] = {
            "fold_assignment": "sha256(source_key) modulo 5",
            "lateral_yaw": {
                name: error_summary(
                    np.concatenate([item[0] for item in values]),
                    np.concatenate([item[1] for item in values]),
                )
                for name, values in held_out.items()
            },
        }
        raw_x = np.asarray(longitudinal_raw)
        calibrated_x = np.asarray(longitudinal_calibrated)
        reference_x = np.asarray(longitudinal_reference)
        ratio = calibrated_x / np.maximum(np.abs(reference_x), 1e-4)
        output["five_fold_calibration_audit"]["longitudinal_through_origin"] = {
            "gain_range": [float(min(gains)), float(max(gains))],
            "raw_mae": float(np.mean(np.abs(raw_x - reference_x))),
            "calibrated_mae": float(np.mean(np.abs(calibrated_x - reference_x))),
            "calibrated_ratio_median": float(np.median(ratio)),
            "calibrated_ratio_iqr": float(np.quantile(ratio, 0.75) - np.quantile(ratio, 0.25)),
            "fraction_within_20_percent": float(np.mean((ratio >= 0.8) & (ratio <= 1.2))),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

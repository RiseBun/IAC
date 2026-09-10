"""Audit predicted and ground-truth projection support from frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.geometry import scale_intrinsics
from iac_new.trajectory_decode import _sparse_predicted_flows


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _support_rows(
    trajectory: np.ndarray,
    *,
    pixel_xy: np.ndarray,
    weights: np.ndarray,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    minimum_points: int,
    minimum_weight_fraction: float,
) -> list[dict[str, Any]]:
    predicted, projected = _sparse_predicted_flows(
        trajectory,
        camera_to_ego,
        intrinsics,
        pixel_xy,
        image_size=image_size,
        depths_m=None,
    )
    source = np.isfinite(weights) & (weights > 0.0)
    usable = source & projected & np.isfinite(predicted).all(axis=-1)
    rows = []
    for index in range(len(trajectory)):
        source_weight = float(np.where(source[index], weights[index], 0.0).sum())
        projected_weight = float(np.where(usable[index], weights[index], 0.0).sum())
        fraction = projected_weight / source_weight if source_weight > 1e-6 else 0.0
        source_points = int(source[index].sum())
        projected_points = int(usable[index].sum())
        passed = (
            source_points >= minimum_points
            and projected_points >= minimum_points
            and fraction >= minimum_weight_fraction
        )
        rows.append(
            {
                "interval_index": index,
                "source_points": source_points,
                "projected_points": projected_points,
                "source_weight": source_weight,
                "projected_weight": projected_weight,
                "projected_weight_fraction": fraction,
                "projection_supported": bool(passed),
            }
        )
    return rows


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _summarize(samples: list[dict[str, Any]], field: str) -> dict[str, Any]:
    intervals = [row for sample in samples for row in sample[field]]
    return {
        "samples": len(samples),
        "intervals": len(intervals),
        "supported_intervals": int(sum(row["projection_supported"] for row in intervals)),
        "interval_coverage": float(np.mean([row["projection_supported"] for row in intervals])),
        "all_intervals_supported_samples": int(
            sum(all(row["projection_supported"] for row in sample[field]) for sample in samples)
        ),
        "sample_coverage_all_intervals": float(
            np.mean([all(row["projection_supported"] for row in sample[field]) for sample in samples])
        ),
        "projected_weight_fraction": _quantiles(
            [row["projected_weight_fraction"] for row in intervals]
        ),
        "zero_projected_intervals": int(sum(row["projected_points"] == 0 for row in intervals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-points", type=int, default=30)
    parser.add_argument("--minimum-weight-fraction", type=float, default=0.5)
    args = parser.parse_args()

    evidence = args.package_root / "evidence" / "navsim_transfer100"
    records = {row["sample_id"]: row for row in _read_json(evidence / "input_records.json")}
    identities = {
        row["sample_id"]: row for row in _read_json(evidence / "input_identities.json")
    }
    config = _read_json(evidence / "effective_config.json")
    target_size = (int(config["image"]["width"]), int(config["image"]["height"]))
    truth = _read_json(evidence / "evaluation_only_truth.json")
    samples = []
    for sample_id, record in records.items():
        prediction = _read_json(evidence / "predictions" / f"{sample_id}_A.json")
        points = np.load(evidence / "predictions" / f"{sample_id}_A_points.npz")
        source_size = tuple(int(value) for value in identities[sample_id]["source_size"])
        common = {
            "pixel_xy": np.asarray(points["xy"], dtype=np.float64),
            "weights": np.asarray(points["weights"], dtype=np.float64),
            "camera_to_ego": np.asarray(record["camera_to_ego"], dtype=np.float64),
            "intrinsics": scale_intrinsics(
                np.asarray(record["intrinsics"], dtype=np.float64), source_size, target_size
            ),
            "image_size": target_size,
            "minimum_points": args.minimum_points,
            "minimum_weight_fraction": args.minimum_weight_fraction,
        }
        samples.append(
            {
                "sample_id": sample_id,
                "group": truth[sample_id]["group"],
                "frozen_prediction_support": prediction["projection_support"],
                "prediction_support": _support_rows(
                    np.asarray(prediction["scored"]["decoder"]["trajectory"], dtype=np.float64),
                    **common,
                ),
                "ground_truth_support": _support_rows(
                    np.asarray(truth[sample_id]["trajectory"], dtype=np.float64),
                    **common,
                ),
            }
        )

    groups = sorted({sample["group"] for sample in samples})
    reconstruction_mismatches = []
    for sample in samples:
        for frozen, reconstructed in zip(
            sample["frozen_prediction_support"], sample["prediction_support"]
        ):
            if frozen["projected_points"] != reconstructed["projected_points"]:
                reconstruction_mismatches.append(
                    {
                        "sample_id": sample["sample_id"],
                        "interval_index": reconstructed["interval_index"],
                        "frozen_projected_points": frozen["projected_points"],
                        "reconstructed_projected_points": reconstructed["projected_points"],
                    }
                )
    report = {
        "protocol": "frozen-projection-support-contract-audit-v1",
        "gate": {
            "minimum_source_points": args.minimum_points,
            "minimum_projected_points": args.minimum_points,
            "minimum_projected_weight_fraction": args.minimum_weight_fraction,
        },
        "population": (
            "NAVSIM transfer100 frozen regression set; this is distinct from the 530 generated-frame "
            "v3 rows and must not be presented as their coverage"
        ),
        "reconstruction_mismatches": reconstruction_mismatches,
        "overall": {
            "prediction_A": _summarize(samples, "prediction_support"),
            "ground_truth": _summarize(samples, "ground_truth_support"),
        },
        "by_group": {
            group: {
                "prediction_A": _summarize(
                    [sample for sample in samples if sample["group"] == group],
                    "prediction_support",
                ),
                "ground_truth": _summarize(
                    [sample for sample in samples if sample["group"] == group],
                    "ground_truth_support",
                ),
            }
            for group in groups
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()

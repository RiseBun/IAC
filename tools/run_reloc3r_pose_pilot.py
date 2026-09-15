#!/usr/bin/env python3
"""Zero-shot Reloc3r pilot for short real or generated driving clips.

The unit of prediction is an image interval, not a single frame.  Reloc3r
estimates the transform from the second view to the first view.  We conjugate
the rotation into the ego basis supplied by the manifest, sum adjacent yaw
increments, and keep forward/reverse cycle residuals as reliability evidence.

Translation is reported only as a direction.  Reloc3r's public demo explicitly
normalizes translation, so this script does not interpret its norm as distance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("manifest JSON must contain a list")
    return value


def _split(source: str) -> str:
    return "calibration" if hashlib.sha256(source.encode()).digest()[0] < 128 else "confirmation"


def _trajectory(record: dict[str, Any]) -> np.ndarray | None:
    value = record.get("action_trajectory")
    if not value:
        return None
    if isinstance(value[0], dict):
        value = [item.get("pose") for item in value]
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 3:
        return None
    return result


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((float(np.trace(rotation)) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(cosine))


def _safe_unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return vector / norm if np.isfinite(norm) and norm > 1e-8 else None


def _ego_rotation(record: dict[str, Any], rotation: np.ndarray) -> np.ndarray:
    camera_to_ego = np.asarray(record.get("camera_to_ego", np.eye(4)), dtype=np.float64)
    if camera_to_ego.shape != (4, 4):
        camera_to_ego = np.eye(4, dtype=np.float64)
    basis = camera_to_ego[:3, :3]
    return basis @ rotation @ basis.T


def _yaw_from_ego_rotation(rotation: np.ndarray) -> float:
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def _interval(record: dict[str, Any], forward: np.ndarray, reverse: np.ndarray) -> dict[str, Any]:
    rf, tf = forward[:3, :3], forward[:3, 3]
    rr, tr = reverse[:3, :3], reverse[:3, 3]
    rf_ego = _ego_rotation(record, rf)
    rr_ego = _ego_rotation(record, rr)
    tf_unit, tr_unit = _safe_unit(tf), _safe_unit(tr)
    inverse_translation_cosine = None
    if tf_unit is not None and tr_unit is not None:
        expected_reverse = -rf.T @ tf_unit
        inverse_translation_cosine = float(np.clip(expected_reverse @ tr_unit, -1.0, 1.0))
    camera_to_ego = np.asarray(record.get("camera_to_ego", np.eye(4)), dtype=np.float64)[:3, :3]
    translation_ego = camera_to_ego @ tf
    translation_ego_unit = _safe_unit(translation_ego)
    return {
        "yaw_rad": _yaw_from_ego_rotation(rf_ego),
        "reverse_yaw_rad": _yaw_from_ego_rotation(rr_ego),
        "yaw_antisymmetry_error_rad": abs(_yaw_from_ego_rotation(rf_ego) + _yaw_from_ego_rotation(rr_ego)),
        "rotation_cycle_error_rad": _rotation_angle(rf @ rr),
        "translation_inverse_cosine": inverse_translation_cosine,
        "translation_raw_norm": float(np.linalg.norm(tf)),
        "translation_direction_ego": translation_ego_unit.tolist() if translation_ego_unit is not None else None,
    }


def _class(value: float, deadband: float) -> str:
    if value < -deadband:
        return "left"
    if value > deadband:
        return "right"
    return "straight"


def _accuracy(rows: list[dict[str, Any]], orientation: int, pred_deadband: float, ref_deadband: float) -> dict[str, Any]:
    usable = [row for row in rows if row.get("reference_yaw_rad") is not None and row.get("predicted_yaw_rad") is not None]
    hits = 0
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    direction_hits = direction_total = 0
    for row in usable:
        reference = _class(float(row["reference_yaw_rad"]), ref_deadband)
        predicted = _class(orientation * float(row["predicted_yaw_rad"]), pred_deadband)
        matrix[reference][predicted] += 1
        hits += int(reference == predicted)
        if reference != "straight":
            direction_total += 1
            direction_hits += int(np.sign(float(row["reference_yaw_rad"])) == np.sign(orientation * float(row["predicted_yaw_rad"])))
    return {
        "records": len(rows),
        "finite": len(usable),
        "coverage": len(usable) / len(rows) if rows else 0.0,
        "three_class_accuracy": hits / len(usable) if usable else None,
        "turn_direction_accuracy": direction_hits / direction_total if direction_total else None,
        "turn_records": direction_total,
        "confusion": {key: dict(value) for key, value in matrix.items()},
        "median_rotation_cycle_error_deg": float(np.degrees(np.median([row["median_rotation_cycle_error_rad"] for row in usable]))) if usable else None,
    }


def _fit_adapter(rows: list[dict[str, Any]], ref_deadband: float) -> tuple[int, float]:
    turns = [row for row in rows if row.get("reference_yaw_rad") is not None and abs(float(row["reference_yaw_rad"])) >= ref_deadband]
    aligned = sum(np.sign(float(row["predicted_yaw_rad"])) == np.sign(float(row["reference_yaw_rad"])) for row in turns)
    orientation = 1 if aligned >= len(turns) / 2 else -1
    values = sorted({0.0, *[abs(float(row["predicted_yaw_rad"])) for row in rows if row.get("predicted_yaw_rad") is not None]})
    best_threshold, best_hits = 0.0, -1
    for threshold in values:
        hits = sum(
            _class(orientation * float(row["predicted_yaw_rad"]), threshold)
            == _class(float(row["reference_yaw_rad"]), ref_deadband)
            for row in rows
            if row.get("reference_yaw_rad") is not None and row.get("predicted_yaw_rad") is not None
        )
        if hits > best_hits:
            best_threshold, best_hits = float(threshold), hits
    return orientation, best_threshold


def _paired(rows: list[dict[str, Any]], orientation: int, ref_delta_deadband: float) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["source_key"])][str(row.get("branch_role", ""))] = row
    comparisons = []
    for source, branches in grouped.items():
        if not {"left", "right"}.issubset(branches):
            continue
        left, right = branches["left"], branches["right"]
        if left.get("reference_yaw_rad") is None or right.get("reference_yaw_rad") is None:
            continue
        reference_delta = float(left["reference_yaw_rad"]) - float(right["reference_yaw_rad"])
        predicted_delta = orientation * (float(left["predicted_yaw_rad"]) - float(right["predicted_yaw_rad"]))
        if abs(reference_delta) < ref_delta_deadband:
            continue
        comparisons.append({"source_key": source, "reference_delta": reference_delta, "predicted_delta": predicted_delta})
    normal_hits = sum(np.sign(item["reference_delta"]) == np.sign(item["predicted_delta"]) for item in comparisons)
    return {
        "source_pairs": len(grouped),
        "material_pairs": len(comparisons),
        "normal_accuracy": normal_hits / len(comparisons) if comparisons else None,
        "reversed_accuracy": (len(comparisons) - normal_hits) / len(comparisons) if comparisons else None,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reloc3r-root", type=Path, required=True)
    parser.add_argument("--resolution", choices=("224", "512"), default="224")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reference-yaw-deadband", type=float, default=0.012)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.reloc3r_root))
    from reloc3r.reloc3r_relpose import inference_relpose, setup_reloc3r_relpose_model
    from reloc3r.utils.device import to_numpy
    from reloc3r.utils.image import check_images_shape_format, load_images

    records = _records(args.manifest)
    if args.limit:
        records = records[: args.limit]
    model = setup_reloc3r_relpose_model(args.resolution, args.device)
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records, 1):
        paths = list(record.get("future_frame_paths") or record.get("future_images") or [])
        trajectory = _trajectory(record)
        if len(paths) < 2:
            continue
        try:
            images = check_images_shape_format(load_images(paths, size=int(args.resolution), verbose=False), args.device)
            intervals = []
            for index in range(len(images) - 1):
                forward = to_numpy(inference_relpose([images[index], images[index + 1]], model, args.device, use_amp=True)[0])
                reverse = to_numpy(inference_relpose([images[index + 1], images[index]], model, args.device, use_amp=True)[0])
                intervals.append(_interval(record, forward, reverse))
            yaws = [item["yaw_rad"] for item in intervals]
            row = {
                "source_key": str(record.get("source_key") or record.get("sample_id") or record_index),
                "sample_id": record.get("sample_id"),
                "branch_role": record.get("branch_role"),
                "split": _split(str(record.get("source_key") or record.get("sample_id") or record_index)),
                "reference_yaw_rad": float(trajectory[-1, 2]) if trajectory is not None else None,
                "reference_path_length": float(np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), trajectory[:, :2]]), axis=0), axis=1).sum()) if trajectory is not None else None,
                "predicted_yaw_rad": float(np.sum(yaws)),
                "median_abs_interval_yaw_rad": float(np.median(np.abs(yaws))),
                "median_rotation_cycle_error_rad": float(np.median([item["rotation_cycle_error_rad"] for item in intervals])),
                "median_yaw_antisymmetry_error_rad": float(np.median([item["yaw_antisymmetry_error_rad"] for item in intervals])),
                "median_translation_inverse_cosine": float(np.median([item["translation_inverse_cosine"] for item in intervals if item["translation_inverse_cosine"] is not None])),
                "intervals": intervals,
            }
            rows.append(row)
            status = "ok"
        except Exception as exc:  # keep failures visible as coverage failures
            rows.append({
                "source_key": str(record.get("source_key") or record.get("sample_id") or record_index),
                "sample_id": record.get("sample_id"),
                "branch_role": record.get("branch_role"),
                "split": _split(str(record.get("source_key") or record.get("sample_id") or record_index)),
                "reference_yaw_rad": float(trajectory[-1, 2]) if trajectory is not None else None,
                "predicted_yaw_rad": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
            status = "failed"
        print(f"[{record_index}/{len(records)}] {record.get('sample_id', record_index)} {status}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"incomplete": True, "rows_detail": rows}, indent=2) + "\n", encoding="utf-8")

    calibration = [row for row in rows if row["split"] == "calibration" and row.get("predicted_yaw_rad") is not None]
    confirmation = [row for row in rows if row["split"] == "confirmation" and row.get("predicted_yaw_rad") is not None]
    orientation, pred_deadband = _fit_adapter(calibration, args.reference_yaw_deadband)
    result = {
        "protocol": "iac-reloc3r-zero-shot-relative-pose-pilot-v1",
        "backend": {"model": f"siyan824/reloc3r-{args.resolution}", "trained_on_iac": False, "candidate_blind": True},
        "translation_claim": "direction only; raw translation norm is not interpreted as distance",
        "records": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "real_calibration_adapter": {"orientation": orientation, "reference_yaw_deadband_rad": args.reference_yaw_deadband, "predicted_yaw_deadband_rad": pred_deadband},
        "calibration": _accuracy(calibration, orientation, pred_deadband, args.reference_yaw_deadband),
        "confirmation": _accuracy(confirmation, orientation, pred_deadband, args.reference_yaw_deadband),
        "paired_direction": _paired(rows, orientation, args.reference_yaw_deadband),
        "rows_detail": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("records", "split_counts", "real_calibration_adapter", "calibration", "confirmation", "paired_direction")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

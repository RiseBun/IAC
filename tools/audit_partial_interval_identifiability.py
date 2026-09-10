#!/usr/bin/env python3
"""Audit whether terminal yaw survives candidate-blind missing-interval masks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EVALUATOR_PATH = ROOT / "scripts" / "evaluate_continuous_decoder.py"
SPEC = importlib.util.spec_from_file_location("iac_evaluator", EVALUATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"unable to load evaluator from {EVALUATOR_PATH}")
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


MASKS = (
    (0,), (1,), (2,), (3,),
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _logged_gt(record: dict[str, Any], manifest_root: Path) -> np.ndarray:
    source = Path(str((record.get("lineage") or {}).get("source_sample") or ""))
    if not source.is_absolute():
        source = manifest_root / source
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    future = payload.get("future_trajectory")
    if not isinstance(future, list) or not future:
        raise ValueError(f"{source}: future_trajectory is absent")
    trajectory = np.asarray([row["pose"][:3] for row in future], dtype=np.float64)
    metadata = payload.get("metadata") or {}
    source_times = np.asarray(metadata.get("future_times_s") or [], dtype=np.float64)
    if source_times.shape != (len(trajectory),):
        source_times = 0.5 * np.arange(1, len(trajectory) + 1, dtype=np.float64)
    target_times = np.asarray(record["future_times_s"], dtype=np.float64)
    if target_times[-1] > source_times[-1] + 0.05:
        raise ValueError(f"{source}: logged GT does not cover the requested horizon")
    target_times = np.clip(target_times, source_times[0], source_times[-1])
    return np.column_stack([
        np.interp(target_times, source_times, trajectory[:, 0]),
        np.interp(target_times, source_times, trajectory[:, 1]),
        np.interp(target_times, source_times, np.unwrap(trajectory[:, 2])),
    ])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 3 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return None
    return float(np.corrcoef(_rankdata(first), _rankdata(second))[0, 1])


def _summarize(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    usable = [row for row in records if row["full"]["motion_explanation_status"] == "explained"]
    masked = np.asarray([row[key]["terminal_yaw_rad"] for row in usable], dtype=np.float64)
    full = np.asarray([row["full"]["terminal_yaw_rad"] for row in usable], dtype=np.float64)
    gt = np.asarray([row["logged_gt_terminal_yaw_rad"] for row in usable], dtype=np.float64)
    material = np.abs(gt) >= 0.01
    full_hits = np.sign(full[material]) == np.sign(gt[material])
    masked_hits = np.sign(masked[material]) == np.sign(gt[material])
    return {
        "n_full_explained": len(usable),
        "masked_fit_improvement_ge_0_05": int(sum(row[key]["fit_improvement"] >= 0.05 for row in usable)),
        "terminal_yaw_abs_change_median_rad": float(np.median(np.abs(masked - full))),
        "terminal_yaw_abs_change_q95_rad": float(np.quantile(np.abs(masked - full), 0.95)),
        "spearman_masked_vs_full": _spearman(masked, full),
        "spearman_full_vs_gt": _spearman(full, gt),
        "spearman_masked_vs_gt": _spearman(masked, gt),
        "material_gt_n": int(np.sum(material)),
        "direction_accuracy_full_vs_gt": float(np.mean(full_hits)) if np.any(material) else None,
        "direction_accuracy_masked_vs_gt": float(np.mean(masked_hits)) if np.any(material) else None,
        "direction_accuracy_drop": (
            float(np.mean(full_hits) - np.mean(masked_hits)) if np.any(material) else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    config = evaluator._json(args.config)
    raw_rows = _read_jsonl(args.manifest)
    if args.max_samples is not None:
        raw_rows = raw_rows[: args.max_samples]
    records = evaluator.validate_manifest_records(raw_rows, config, args.manifest.parent)
    flow_cfg = config["flow"]
    checkpoint = Path(str(flow_cfg["checkpoint"]))
    if not checkpoint.is_absolute():
        checkpoint = args.config.parent.parent / checkpoint
    extractor = evaluator.RaftFlowExtractor(
        model_size=str(flow_cfg["model"]),
        device=args.device,
        updates=int(flow_cfg["updates"]),
        batch_size=int(flow_cfg["batch_size"]),
        forward_backward=bool(flow_cfg["forward_backward"]),
        fb_abs_threshold_px=float(flow_cfg["fb_abs_threshold_px"]),
        fb_relative_threshold=float(flow_cfg["fb_relative_threshold"]),
        checkpoint=str(checkpoint),
    )
    perception = evaluator.build_perception(config, device=args.device)
    original_decode = evaluator.decode_continuous_trajectory
    current: dict[str, Any] | None = None

    def wrapped_decode(**kwargs: Any) -> dict[str, Any]:
        nonlocal current
        full = original_decode(**kwargs)
        variants: dict[str, Any] = {"full": full}
        base_weights = np.asarray(kwargs["dynamic_weights"], dtype=np.float32)
        base_quality = np.asarray(kwargs["interval_observability"], dtype=np.float64)
        for missing in MASKS:
            masked_weights = base_weights.copy()
            masked_quality = base_quality.copy()
            masked_weights[list(missing)] = 0.0
            masked_quality[list(missing)] = 0.0
            variants["missing_" + "_".join(map(str, missing))] = original_decode(
                **{
                    **kwargs,
                    "dynamic_weights": masked_weights,
                    "interval_observability": masked_quality,
                }
            )
        current = variants
        return full

    evaluator.decode_continuous_trajectory = wrapped_decode
    output_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        for index, record in enumerate(records, start=1):
            try:
                current = None
                evaluator.evaluate_record(record, extractor, config, perception)
                if current is None:
                    raise RuntimeError("decoder wrapper was not called")
                gt = _logged_gt(raw_rows[index - 1], args.manifest.parent)
                result: dict[str, Any] = {
                    "sample_id": record["sample_id"],
                    "source_key": raw_rows[index - 1].get("source_key"),
                    "stratum": (raw_rows[index - 1].get("metadata") or {}).get("stratum"),
                    "logged_gt_terminal_yaw_rad": float(gt[-1, 2]),
                }
                for key, decoded in current.items():
                    result[key] = {
                        "terminal_yaw_rad": float(decoded["trajectory"][-1][2]),
                        "energy": float(decoded["energy"]),
                        "fit_improvement": float(decoded["fit_improvement"]),
                        "motion_explanation_status": decoded["motion_explanation_status"],
                    }
                output_rows.append(result)
            except Exception as error:
                errors.append({"sample_id": str(record.get("sample_id")), "error": str(error)})
            print(json.dumps({"completed": index, "total": len(records)}), flush=True)
    finally:
        evaluator.decode_continuous_trajectory = original_decode

    variant_keys = ["missing_" + "_".join(map(str, mask)) for mask in MASKS]
    output = {
        "protocol": "step1-partial-interval-identifiability-audit-v1",
        "fit_protocol": config.get("step1_measurement"),
        "missing_interval_patterns": [list(mask) for mask in MASKS],
        "num_input": len(records),
        "num_scored": len(output_rows),
        "errors": errors,
        "summary": {key: _summarize(output_rows, key) for key in variant_keys},
        "records": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

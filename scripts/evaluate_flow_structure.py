#!/usr/bin/env python3
"""Evaluate candidate-blind, decoder-free flow structure on image clips."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.flow import RaftFlowExtractor
from iac_new.sea_raft_flow import SeaRaftFlowExtractor
from iac_new.protocol import read_jsonl, validate_record, write_jsonl
from iac_new.road_structure import flow_structure_profile
from iac_new.scoring import polygon_mask


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_record(
    record: dict[str, Any],
    extractor: RaftFlowExtractor | SeaRaftFlowExtractor,
    config: dict[str, Any],
) -> dict[str, Any]:
    width = int(config["image"]["width"])
    height = int(config["image"]["height"])
    flow_cfg = config["flow"]
    observation = extractor.observe(
        record["frame_paths"],
        record["intrinsics"],
        record["distortion"],
        (width, height),
        inference_size=(
            tuple(flow_cfg["inference_size"])
            if flow_cfg.get("inference_size") is not None else None
        ),
        allow_mixed_source_sizes=bool(config.get("allow_mixed_source_sizes", False)),
        intrinsics_source_size=(
            record.get("intrinsics_source_size")
            or (
                tuple(config["intrinsics_source_size"])
                if config.get("intrinsics_source_size") is not None else None
            )
        ),
    )
    future_start = int(record["history_count"]) - 1
    flows = np.asarray(observation.forward[future_start:], dtype=np.float32)
    consistency = (
        np.ones(flows.shape[:-1], dtype=bool)
        if observation.consistency_masks is None
        else np.asarray(observation.consistency_masks[future_start:], dtype=bool)
    )
    roi = polygon_mask(height, width, config["mask"]["polygon_normalized"])
    weights = consistency.astype(np.float32)
    structure_cfg = config.get("flow_structure", {})
    profile = flow_structure_profile(
        flows,
        weights,
        roi,
        observation.intrinsics,
        min_flow_px=float(structure_cfg.get("min_flow_px", 0.5)),
        min_points=int(structure_cfg.get("min_points", 100)),
        min_spatial_cells=int(structure_cfg.get("min_spatial_cells", 6)),
        max_points=int(structure_cfg.get("max_points", 3000)),
        grid_rows=int(structure_cfg.get("grid_rows", 3)),
        grid_cols=int(structure_cfg.get("grid_cols", 6)),
    )
    for index, row in enumerate(profile["rows"]):
        finite = np.isfinite(flows[index]).all(axis=-1) & roi
        row["finite_flow_fraction"] = float(finite.sum() / max(roi.sum(), 1))
        row["forward_backward_fraction"] = float(
            (finite & consistency[index]).sum() / max(finite.sum(), 1)
        )
    return {
        "sample_id": record["sample_id"],
        "scene_id": record["scene_id"],
        "source_key": str(record.get("source_key") or record["sample_id"].split("::")[0]),
        "branch_role": record.get("branch_role"),
        "stratum": str((record.get("metadata") or {}).get("stratum", "unknown")),
        "future_times_s": np.asarray(record["future_times_s"], dtype=np.float64).tolist(),
        "source_frame_size": list(observation.source_size),
        "effective_intrinsics": np.asarray(observation.intrinsics, dtype=np.float64).tolist(),
        "flow_structure": profile,
        "candidate_bank_used_by_measurement": False,
        "metric_reconstruction_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    config = _json(args.config)
    raw_rows = read_jsonl(args.manifest)
    if args.max_samples is not None:
        raw_rows = raw_rows[: args.max_samples]
    records = []
    calibration_contract = config.get("calibration_contract") or {}
    expected_source_size = calibration_contract.get("expected_intrinsics_source_size")
    for raw in raw_rows:
        record = validate_record(
            raw,
            manifest_root=args.manifest.parent,
            require_candidates=False,
            require_intrinsics_source_size=bool(
                calibration_contract.get("require_explicit_intrinsics_source_size", False)
            ),
            expected_intrinsics_source_size=(
                tuple(int(value) for value in expected_source_size)
                if expected_source_size is not None else None
            ),
        )
        record["source_key"] = raw.get("source_key")
        record["branch_role"] = raw.get("branch_role")
        records.append(record)
    flow_cfg = config["flow"]
    backend = str(flow_cfg.get("backend", "torchvision_raft"))
    checkpoint = flow_cfg.get("checkpoint")
    if checkpoint:
        checkpoint_path = Path(str(checkpoint))
        if not checkpoint_path.is_absolute():
            checkpoint_path = args.config.parent.parent / checkpoint_path
        checkpoint = str(checkpoint_path)
    if backend == "torchvision_raft":
        extractor = RaftFlowExtractor(
            model_size=str(flow_cfg["model"]),
            device=args.device,
            updates=int(flow_cfg["updates"]),
            batch_size=int(flow_cfg["batch_size"]),
            forward_backward=bool(flow_cfg["forward_backward"]),
            fb_abs_threshold_px=float(flow_cfg["fb_abs_threshold_px"]),
            fb_relative_threshold=float(flow_cfg["fb_relative_threshold"]),
            checkpoint=checkpoint,
        )
    elif backend == "sea_raft":
        if not checkpoint:
            raise ValueError("SEA-RAFT requires flow.checkpoint")
        extractor = SeaRaftFlowExtractor(
            checkpoint=checkpoint,
            device=args.device,
            iters=int(flow_cfg.get("iters", 12)),
            forward_backward=bool(flow_cfg["forward_backward"]),
            fb_abs_threshold_px=float(flow_cfg["fb_abs_threshold_px"]),
            fb_relative_threshold=float(flow_cfg["fb_relative_threshold"]),
        )
    else:
        raise ValueError(f"unsupported flow backend: {backend}")
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        try:
            rows.append(evaluate_record(record, extractor, config))
        except Exception as error:
            errors.append({"sample_id": record["sample_id"], "error": str(error)})
        print(json.dumps({"completed": index, "total": len(records)}), flush=True)
    write_jsonl(args.output, rows)
    statuses = Counter(row["flow_structure"]["status"] for row in rows)
    intervals = [item for row in rows for item in row["flow_structure"]["rows"]]
    summary = {
        "protocol": "candidate-blind-flow-structure-v1",
        "manifest": str(args.manifest.resolve()),
        "config": str(args.config.resolve()),
        "num_input": len(records),
        "num_scored": len(rows),
        "num_error": len(errors),
        "status_counts": dict(statuses),
        "measurement_available_fraction": float(np.mean([
            row["flow_structure"]["measurement_available"] for row in rows
        ])) if rows else None,
        "input_available_interval_fraction": float(np.mean([
            item["input_available"] for item in intervals
        ])) if intervals else None,
        "mean_structure_confidence": float(np.mean([
            item["structure_confidence"] for item in intervals
        ])) if intervals else None,
        "elapsed_s": time.perf_counter() - started,
        "errors": errors,
    }
    args.output.with_name(f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

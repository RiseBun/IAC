#!/usr/bin/env python3
"""Extract candidate-blind dense future flow for forward-consistency audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iac_new.flow import RaftFlowExtractor
from iac_new.flow_structure_scoring import _manifest_action_trajectory
from iac_new.protocol import read_jsonl, validate_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract = config.get("calibration_contract") or {}
    expected_size = contract.get("expected_intrinsics_source_size")
    raw_rows = read_jsonl(args.manifest)
    if args.max_samples is not None:
        raw_rows = raw_rows[: args.max_samples]
    flow_cfg = config["flow"]
    checkpoint = flow_cfg.get("checkpoint")
    if checkpoint and not Path(str(checkpoint)).is_absolute():
        checkpoint = str(args.config.parent.parent / str(checkpoint))
    extractor = RaftFlowExtractor(
        model_size=str(flow_cfg["model"]), device=args.device,
        updates=int(flow_cfg["updates"]), batch_size=int(flow_cfg["batch_size"]),
        forward_backward=bool(flow_cfg["forward_backward"]),
        fb_abs_threshold_px=float(flow_cfg["fb_abs_threshold_px"]),
        fb_relative_threshold=float(flow_cfg["fb_relative_threshold"]),
        checkpoint=checkpoint,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, raw in enumerate(raw_rows):
        record = validate_record(
            raw, manifest_root=args.manifest.parent, require_candidates=False,
            require_intrinsics_source_size=bool(contract.get("require_explicit_intrinsics_source_size", False)),
            expected_intrinsics_source_size=(tuple(int(v) for v in expected_size) if expected_size else None),
        )
        observation = extractor.observe(
            record["frame_paths"], record["intrinsics"], record["distortion"],
            (int(config["image"]["width"]), int(config["image"]["height"])),
            inference_size=tuple(flow_cfg["inference_size"]),
            allow_mixed_source_sizes=bool(config.get("allow_mixed_source_sizes", False)),
            intrinsics_source_size=record.get("intrinsics_source_size"),
            image_geometry_adapter=record.get("image_geometry_adapter"),
        )
        start = int(record["history_count"]) - 1
        flow = np.asarray(observation.forward[start:], dtype=np.float32)
        if observation.consistency_masks is None:
            valid = np.ones(flow.shape[:-1], dtype=bool)
        else:
            valid = np.asarray(observation.consistency_masks[start:], dtype=bool)
        stem = f"{index:06d}"
        np.save(args.output_root / f"{stem}.flow.npy", flow)
        np.save(args.output_root / f"{stem}.valid.npy", valid)
        metadata = {
            "index": index, "sample_id": raw.get("sample_id", record["sample_id"]),
            "source_key": raw.get("source_key"), "branch_role": raw.get("branch_role"),
            "trajectory": _manifest_action_trajectory(raw).tolist(),
            "intrinsics": np.asarray(observation.intrinsics).tolist(),
            "camera_to_ego": np.asarray(record["camera_to_ego"]).tolist(),
            "future_times_s": np.asarray(record["future_times_s"]).tolist(),
            "flow_path": f"{stem}.flow.npy", "valid_path": f"{stem}.valid.npy",
            "candidate_blind": True,
        }
        (args.output_root / f"{stem}.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
        rows.append(metadata)
        print(json.dumps({"completed": index + 1, "total": len(raw_rows)}), flush=True)
    (args.output_root / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

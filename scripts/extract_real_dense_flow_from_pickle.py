#!/usr/bin/env python3
"""Extract four one-second dense-flow intervals from logged pickle images."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from iac_new.flow import RaftFlowExtractor
from iac_new.geometry import scale_intrinsics


def _resize_flow(flow: np.ndarray, source_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    width, height = target_size
    out = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out[..., 0] *= width / float(source_size[0])
    out[..., 1] *= height / float(source_size[1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-sources", type=int)
    args = parser.parse_args()
    raw_rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_source = {}
    for row in raw_rows:
        by_source.setdefault(str(row["source_key"]), row)
    source_rows = list(by_source.values())
    if args.max_sources is not None:
        source_rows = source_rows[: args.max_sources]
    extractor = RaftFlowExtractor(
        model_size="large", device=args.device, updates=32, batch_size=4,
        forward_backward=False, fb_abs_threshold_px=1.5,
        fb_relative_threshold=0.05, checkpoint=str(args.checkpoint),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for index, manifest_row in enumerate(source_rows):
        source_pkl = (manifest_row.get("metadata") or {}).get("source_sample")
        if not source_pkl:
            raise ValueError(f"{manifest_row['source_key']}: missing metadata.source_sample")
        with open(source_pkl, "rb") as handle:
            sample = pickle.load(handle)
        images = np.asarray(sample["images"])
        if images.ndim != 4 or images.shape[0] < 9:
            raise ValueError(f"{source_pkl}: expected current plus at least 8 future images")
        # The archived contract is current + eight 0.5s future frames.  Use
        # every other frame to obtain four one-second intervals.
        selected = [images[0], images[2], images[4], images[6], images[8]]
        inference_size = (512, 288)
        target_size = (448, 256)
        resized = [cv2.resize(frame, inference_size, interpolation=cv2.INTER_AREA) for frame in selected]
        forward, _ = extractor._infer_pairs(resized[:-1], resized[1:])
        flow = np.stack([_resize_flow(value, inference_size, target_size) for value in forward], axis=0)
        valid = np.isfinite(flow).all(axis=-1)
        trajectory = np.asarray(
            [row["pose"][:3] for row in sample["future_trajectory"]],
            dtype=np.float64,
        )[[1, 3, 5, 7]]
        stem = f"{index:06d}"
        np.save(args.output_root / f"{stem}.flow.npy", flow)
        np.save(args.output_root / f"{stem}.valid.npy", valid)
        effective_intrinsics = scale_intrinsics(
            np.asarray(manifest_row["intrinsics"], dtype=np.float64),
            (1920, 1080), target_size,
        )
        metadata = {
            "index": index, "source_key": str(manifest_row["source_key"]),
            "branch_role": "real", "sample_id": sample.get("metadata", {}).get("sample_id"),
            "trajectory": trajectory.tolist(),
            "intrinsics": effective_intrinsics.tolist(),
            "camera_to_ego": np.asarray(manifest_row["camera_to_ego"], dtype=np.float64).tolist(),
            "flow_path": f"{stem}.flow.npy", "valid_path": f"{stem}.valid.npy",
            "candidate_blind": True, "source_pkl": source_pkl,
        }
        (args.output_root / f"{stem}.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
        output_rows.append(metadata)
        print(json.dumps({"completed": index + 1, "total": len(source_rows)}), flush=True)
    (args.output_root / "manifest.json").write_text(json.dumps(output_rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

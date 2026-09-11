#!/usr/bin/env python3
"""Build candidate-blind flow records from DriveWAM generated manifests."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(inputs: list[Path], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in inputs:
        for raw in _read(path):
            sample_id = f"{raw['source_sample']}::{raw.get('branch_mode', 'unknown')}"
            if sample_id in seen:
                raise ValueError(f"duplicate branch: {sample_id}")
            seen.add(sample_id)
            source_sample = Path(str(raw["source_sample"]))
            speed_role = str(raw.get("speed_role") or ("fast" if "/fast/" in str(source_sample) else "slow" if "/slow/" in str(source_sample) else "unknown"))
            with source_sample.open("rb") as handle:
                sample = pickle.load(handle)
            metadata = sample.get("metadata") or {}
            history = list(metadata.get("image_paths") or [])[-4:]
            future = list(raw.get("future_images") or [])
            if len(history) != 4 or len(future) < 4:
                raise ValueError(f"{sample_id}: expected four history and four future images")
            row = dict(raw)
            row.update({
                "sample_id": sample_id,
                "source_key": metadata.get("source_key"),
                "counterfactual_group_id": metadata.get("source_key"),
                "branch_role": "left" if speed_role == "fast" else ("right" if speed_role == "slow" else raw.get("branch_mode")),
                "speed_role": speed_role,
                "scene_id": str(metadata.get("scene_token", "")),
                "history_frame_paths": history,
                "future_frame_paths": future[:4],
                "history_times_s": [-1.5, -1.0, -0.5, 0.0],
                "future_times_s": [1.0, 2.0, 3.0, 4.0],
                "intrinsics": metadata.get("camera_intrinsic"),
                "distortion": metadata.get("camera_distortion"),
                "camera_to_ego": metadata.get("camera_to_ego"),
                "intrinsics_source_size": [1920, 1080],
                "image_geometry_adapter": {
                    "schema": "iac-image-geometry-v1",
                    "adapter_id": "drivewam_native_output_v1",
                    "calibration_source_size": [1920, 1080],
                    "frame_groups": [
                        {"indices": [0, 1, 2, 3], "operation": "identity", "frame_size": [1920, 1080]},
                        # The decoder emits the model frame at this resolution;
                        # geometry is equivalent to a fixed source-to-frame resize.
                        {"indices": [4, 5, 6, 7], "operation": "direct_resize", "frame_size": [448, 256]},
                    ],
                    "provenance": {"candidate_dependent": False},
                },
                "frame_paths": history + future[:4],
                "history_count": 4,
                "future_count": 4,
                "candidate_bank_used_by_measurement": False,
                "metric_reconstruction_used": False,
            })
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    report = {"protocol": "iac-drivewam-flow-manifest-v1", "row_count": len(rows), "source_count": len({r.get('source_key') for r in rows}), "output": str(output)}
    output.with_name(f"{output.stem}_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

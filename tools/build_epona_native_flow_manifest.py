#!/usr/bin/env python3
"""Build a candidate-blind flow manifest for native Epona control output."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def build(input_manifest: Path, output: Path, sensor_root: Path, start_index: int) -> dict:
    rows = json.loads(input_manifest.read_text(encoding="utf-8"))
    output_rows = []
    for raw in rows:
        source = Path(str(raw["source_sample"]))
        frames = pickle.loads(source.read_bytes())
        start = int(raw.get("source_start_index", start_index + int(raw["sample_index"]) * 5))
        window = frames[start : start + 15]
        if len(window) < 15:
            raise ValueError(f"{source}: incomplete window at {start}")
        history = [str(sensor_root / frame["cams"]["CAM_F0"]["data_path"]) for frame in window[:10]][-4:]
        future = list(raw["future_images"])
        branch = str(raw.get("branch_mode", "unknown"))
        camera = window[9]["cams"]["CAM_F0"]
        intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64).tolist()
        distortion = np.asarray(camera.get("distortion", []), dtype=np.float64).tolist()
        camera_to_ego = np.eye(4, dtype=np.float64)
        camera_to_ego[:3, :3] = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
        camera_to_ego[:3, 3] = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
        row = dict(raw)
        row.update(
            {
                "sample_id": f"{raw['source_key']}::{branch}",
                "source_key": raw["source_key"],
                "counterfactual_group_id": raw["counterfactual_group_id"],
                "scene_id": raw["source_key"],
                "branch_role": branch,
                "speed_role": branch,
                "history_fingerprint": raw["source_key"],
                "nuisance_seed": int(raw.get("nuisance_seed", 9103000 + int(raw["sample_index"]))),
                "intervention_type": "action_trajectory_perturbation",
                "wam_model_id": "epona_nuplan",
                "history_frame_paths": history,
                "future_frame_paths": future,
                "history_times_s": [-1.5, -1.0, -0.5, 0.0],
                "future_times_s": [0.5, 1.0, 1.5, 2.0],
                "intrinsics": intrinsic,
                "intrinsics_source_size": [1920, 1080],
                "distortion": distortion,
                "camera_to_ego": camera_to_ego.tolist(),
                "frame_paths": history + future,
                "history_count": 4,
                "future_count": 4,
                "candidate_bank_used_by_measurement": False,
                "metric_reconstruction_used": False,
                "lineage": {
                    "same_history_seed": True,
                    "runner_branch": branch,
                    "source_sample": str(source),
                    "source_start_index": start,
                },
                "image_geometry_adapter": {
                    "schema": "iac-image-geometry-v1",
                    "adapter_id": "epona_native_control_v1",
                    "calibration_source_size": [1920, 1080],
                    "frame_groups": [
                        {"indices": [0, 1, 2, 3], "operation": "identity", "frame_size": [1920, 1080]},
                        {"indices": [4, 5, 6, 7], "operation": "direct_resize", "frame_size": [1024, 512]},
                    ],
                    "provenance": {"candidate_dependent": False},
                },
            }
        )
        output_rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output_rows), encoding="utf-8")
    return {"protocol": "iac-epona-native-flow-manifest-v1", "row_count": len(output_rows), "source_count": len({r['source_key'] for r in output_rows}), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, args.sensor_root, args.start_index), indent=2))


if __name__ == "__main__":
    main()

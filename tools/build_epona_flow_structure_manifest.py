#!/usr/bin/env python3
"""Convert the archived Epona five-group probe to the standard flow manifest."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _integrate_epona_controls(xy: np.ndarray, yaw_deg: np.ndarray) -> np.ndarray:
    pose = np.zeros(3, dtype=np.float64)
    rows = []
    for delta_xy, delta_yaw_deg in zip(xy, yaw_deg.reshape(-1)):
        cosine, sine = np.cos(pose[2]), np.sin(pose[2])
        dx = float(delta_xy[0])
        # Epona translation is right-positive while its yaw is left-positive;
        # convert each delta before standard left-positive SE(2) integration.
        dy = -float(delta_xy[1])
        pose[:2] += [cosine * dx - sine * dy, sine * dx + cosine * dy]
        pose[2] += np.deg2rad(float(delta_yaw_deg))
        rows.append(pose.copy())
    return np.asarray(rows, dtype=np.float64)


def _camera_matrix(frame: dict[str, Any]) -> tuple[list[list[float]], list[float], list[list[float]]]:
    camera = frame["cams"]["CAM_F0"]
    rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    translation = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    camera_to_ego = np.eye(4, dtype=np.float64)
    camera_to_ego[:3, :3] = rotation
    camera_to_ego[:3, 3] = translation
    return (
        np.asarray(camera["cam_intrinsic"], dtype=np.float64).tolist(),
        np.asarray(camera.get("distortion", []), dtype=np.float64).tolist(),
        camera_to_ego.tolist(),
    )


def _source_start(item: dict[str, Any]) -> int:
    if item.get("source_start_index") is not None:
        return int(item["source_start_index"])
    source_key_parts = str(item.get("source_key") or "").rsplit(":", 1)
    if len(source_key_parts) == 2 and source_key_parts[1].isdigit():
        return int(source_key_parts[1])
    # Compatibility only for the archived five-group probe, whose producer
    # used a fixed stride but did not persist the actual window start.
    return int(item["sample_index"]) * 5


def build_rows(
    source_pickle: Path,
    epona_manifest: Path,
    sensor_root: Path,
    *,
    future_cadence: str = "protocol_1hz",
) -> list[dict[str, Any]]:
    frames = pickle.loads(source_pickle.read_bytes())
    timestamps = np.asarray([frame["timestamp"] for frame in frames], dtype=np.float64)
    source_step_s = float(np.median(np.diff(timestamps)) / 1e6)
    if not 0.45 <= source_step_s <= 0.55:
        raise ValueError(f"unexpected Epona source cadence: {source_step_s:.6f}s")
    archived = json.loads(epona_manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in archived:
        index = int(item["sample_index"])
        start = _source_start(item)
        window = frames[start : start + 10]
        if len(window) != 10:
            raise ValueError(f"sample {index}: source pickle does not contain the history window")
        anchor = window[9]
        history = [
            str((sensor_root / frame["cams"]["CAM_F0"]["data_path"]).resolve())
            for frame in window[6:10]
        ]
        intrinsic, distortion, camera_to_ego = _camera_matrix(anchor)
        mode = str(item["branch_mode"])
        if mode not in {"left", "right"}:
            continue
        source_key = str(
            item.get("counterfactual_group_id")
            or item.get("source_key")
            or f"epona_native:{start}"
        )
        archived_images = [str(Path(value).resolve()) for value in item["future_images"]]
        archived_yaw_deg = np.asarray(item["action_yaw_deg"], dtype=np.float64).reshape(-1)
        if len(archived_yaw_deg) != len(archived_images):
            raise ValueError(f"sample {index}/{mode}: action yaw length does not match frames")
        # The archived probe incorrectly labeled each generated step as 0.2 s.
        # Its controls come from consecutive NAVSIM records, whose measured
        # cadence is 0.5 s.  Derive selection from source timestamps instead
        # of trusting that stale label.
        if future_cadence == "protocol_1hz":
            target_times = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
            selected = np.rint(target_times / source_step_s).astype(np.int64) - 1
            times = target_times
        else:
            native_count = int(round(4.0 / source_step_s))
            selected = np.arange(native_count, dtype=np.int64)
            times = source_step_s * (selected + 1)
        if len(archived_images) <= int(selected[-1]):
            raise ValueError(f"sample {index}/{mode}: expected 20 archived future frames")
        future_images = [archived_images[int(value)] for value in selected]
        yaw_deg = archived_yaw_deg[selected]
        archived_action_xy = np.asarray(item["action_trajectory"], dtype=np.float64)
        if (
            archived_action_xy.ndim != 2
            or archived_action_xy.shape[0] != len(archived_images)
            or archived_action_xy.shape[1] < 2
        ):
            raise ValueError(f"sample {index}/{mode}: action trajectory must be [frames,>=2]")
        action = _integrate_epona_controls(
            archived_action_xy[:, :2], archived_yaw_deg
        )[selected].tolist()
        rows.append({
            "sample_id": f"{source_key}::{mode}",
            "source_key": source_key,
            "counterfactual_group_id": source_key,
            "branch_role": mode,
            "scene_id": str(anchor.get("scene_token", source_key)),
            "history_frame_paths": history,
            "future_frame_paths": future_images,
            "history_times_s": [-1.5, -1.0, -0.5, 0.0],
            "future_times_s": times.tolist(),
            "intrinsics": intrinsic,
            "distortion": distortion,
            "camera_to_ego": camera_to_ego,
            "intrinsics_source_size": [1920, 1080],
            "image_geometry_adapter": {
                "schema": "iac-image-geometry-v1",
                "adapter_id": "epona_nuplan_direct_resize_v1",
                "calibration_source_size": [1920, 1080],
                "frame_groups": [
                    {
                        "indices": [0, 1, 2, 3],
                        "operation": "identity",
                        "frame_size": [1920, 1080],
                    },
                    {
                        "indices": list(range(4, 4 + len(future_images))),
                        "operation": "direct_resize",
                        "frame_size": [1024, 512],
                    },
                ],
                "provenance": {
                    "producer": "run_epona_action_control_native.py:_load_image",
                    "operation": "cv2.resize(source, (1024, 512), INTER_AREA)",
                    "candidate_dependent": False,
                },
            },
            "action_trajectory": action,
            "action_trajectory_source": "epona_conditioning_controls_integrated",
            "future_images_source": str(
                item.get("future_images_source") or "epona_generated"
            ),
            "wam_model_id": "epona_nuplan",
            "candidate_bank_used_by_decoder": False,
            "command_override": mode,
            "metadata": {
                "stratum": "epona_smoke",
                "protocol": (
                    "wam-native-epona-source-cadence"
                    if future_cadence == "native_cadence" else "iac-explicit-1hz"
                ),
                "action_label_source": "archived_epona_action_yaw_deg",
                "action_coordinate_adapter": "epona_y_right_to_iac_y_left_v1",
                "future_downsample": (
                    "first_4s_at_source_cadence"
                    if future_cadence == "native_cadence"
                    else "source_timestamp_derived_1_2_3_4s"
                ),
                "source_frame_step_s": source_step_s,
                "archived_future_step_s": 0.2,
                "archived_future_step_status": "rejected_inconsistent_with_source_timestamps",
                "source_pickle": str(source_pickle),
                "source_start_index": start,
                "randomness_contract": item.get("randomness_contract"),
            },
            "candidates": [],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pickle", type=Path, required=True)
    parser.add_argument("--epona-manifest", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--future-cadence",
        choices=("protocol_1hz", "native_cadence"),
        default="protocol_1hz",
    )
    args = parser.parse_args()
    rows = build_rows(
        args.source_pickle,
        args.epona_manifest,
        args.sensor_root,
        future_cadence=args.future_cadence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "groups": len({row["source_key"] for row in rows})}))


if __name__ == "__main__":
    main()

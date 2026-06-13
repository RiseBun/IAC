#!/usr/bin/env python3
"""Build an IAC consistency index from NAVSIM/OpenScene logs.

This is an adapter for NAVSIM-style data:

  navsim_logs/*.pkl
  sensor_blobs/<split>/<scene_or_log>/CAM_F0/*.jpg

It emits the same JSONL schema consumed by train.py:
history_images, future_images, ego_state, candidate_traj, consistency_label,
validity_label, source_type, ...

NAVSIM/OpenScene training splits often contain only history/current sensor
frames. In that case use --future-image-policy history_tail or repeat_current
to run an IAC-compatible NAVSIM experiment. Use the default future policy only
when future camera frames are actually present in sensor_blobs.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_consistency_index import (  # noqa: E402
    ConsistencyAnchor,
    compute_traj_scale_factors,
    count_source_types,
    serialize_split,
    split_scenes,
    wrap_angle,
    write_jsonl,
    yaw_from_quaternion,
)


@dataclass
class NavsimFrame:
    token: str
    timestamp: int
    filename_jpg: str
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    ax: float
    yaw_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an IAC JSONL index from NAVSIM/OpenScene logs",
    )
    parser.add_argument("--navsim-log-root", required=True, help="Directory containing NAVSIM .pkl log files")
    parser.add_argument("--sensor-root", required=True, help="Sensor blob root; image paths in logs are relative to this")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-channel", default="CAM_F0")
    parser.add_argument("--history-num-frames", type=int, default=4)
    parser.add_argument("--future-image-offsets", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--future-steps", type=int, default=8)
    parser.add_argument("--future-step-time-s", type=float, default=0.5)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-logs", type=int, default=0)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--max-anchors-per-log", type=int, default=0)
    parser.add_argument("--min-negative-index-gap", type=int, default=20)
    parser.add_argument("--perturb-lateral-range", type=float, nargs=2, default=[0.5, 2.0])
    parser.add_argument("--perturb-heading-range", type=float, nargs=2, default=[5.0, 15.0])
    parser.add_argument("--perturb-speed-range", type=float, nargs=2, default=[0.7, 1.3])
    parser.add_argument("--time-shift-future-steps", type=int, default=2)
    parser.add_argument("--add-reverse-traj", action="store_true")
    parser.add_argument(
        "--future-image-policy",
        choices=["future", "history_tail", "repeat_current"],
        default="future",
        help=(
            "future: require future camera frames. history_tail/repeat_current: "
            "IAC-compatible fallback for NAVSIM splits without future images."
        ),
    )
    return parser.parse_args()


def _iter_log_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.pkl") if path.is_file())


def _extract_frame_lists(obj: Any) -> List[List[Dict[str, Any]]]:
    """Return frame-list logs from common NAVSIM pickle shapes."""
    if isinstance(obj, list):
        if not obj:
            return []
        if all(isinstance(item, dict) for item in obj):
            return [obj]
        lists: List[List[Dict[str, Any]]] = []
        for item in obj:
            lists.extend(_extract_frame_lists(item))
        return lists
    if isinstance(obj, dict):
        for key in ("frames", "scene_frames", "data", "logs", "samples"):
            if key in obj:
                return _extract_frame_lists(obj[key])
    return []


def _camera_entry(cams: Dict[str, Any], channel: str) -> Optional[Dict[str, Any]]:
    if channel in cams:
        return cams[channel]
    upper = channel.upper()
    lower = channel.lower()
    for key, value in cams.items():
        if str(key).upper() == upper or str(key).lower() == lower:
            return value
    return None


def _frame_to_navsim(frame: Dict[str, Any], channel: str) -> Optional[NavsimFrame]:
    cams = frame.get("cams") or frame.get("camera_dict") or {}
    cam = _camera_entry(cams, channel)
    if not cam:
        return None
    filename = cam.get("data_path") or cam.get("filename_jpg") or cam.get("camera_path")
    if not filename:
        return None

    trans = frame.get("ego2global_translation")
    rot = frame.get("ego2global_rotation")
    if trans is None or rot is None:
        ego = frame.get("ego_status") or {}
        pose = ego.get("ego_pose") if isinstance(ego, dict) else None
        if pose is None or len(pose) < 3:
            return None
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    else:
        x, y = float(trans[0]), float(trans[1])
        yaw = yaw_from_quaternion(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))

    dyn = frame.get("ego_dynamic_state") or []
    vx = float(dyn[0]) if len(dyn) > 0 else 0.0
    vy = float(dyn[1]) if len(dyn) > 1 else 0.0
    ax = float(dyn[2]) if len(dyn) > 2 else 0.0
    yaw_rate = float(dyn[4]) if len(dyn) > 4 else 0.0
    token = str(frame.get("token") or frame.get("scene_token") or frame.get("timestamp") or "")
    timestamp = int(frame.get("timestamp") or 0)
    return NavsimFrame(
        token=token,
        timestamp=timestamp,
        filename_jpg=str(filename),
        x=x,
        y=y,
        yaw=yaw,
        vx=vx,
        vy=vy,
        ax=ax,
        yaw_rate=yaw_rate,
    )


def _relative_path(sensor_root: Path, image_path: str) -> str:
    # Keep the same convention as build_consistency_index.py: paths are
    # relative to sensor_root.parent and include sensor_root.name.
    return str(Path(sensor_root.name) / image_path)


def _trajectory_from_frames(current: NavsimFrame, futures: Sequence[NavsimFrame]) -> List[List[float]]:
    cos_yaw = math.cos(-current.yaw)
    sin_yaw = math.sin(-current.yaw)
    traj: List[List[float]] = []
    for pose in futures:
        dx_w = pose.x - current.x
        dy_w = pose.y - current.y
        dx_l = dx_w * cos_yaw - dy_w * sin_yaw
        dy_l = dx_w * sin_yaw + dy_w * cos_yaw
        dyaw = wrap_angle(pose.yaw - current.yaw)
        traj.append([dx_l, dy_l, dyaw])
    return traj


def _valid_image_paths(sensor_root: Path, frames: Sequence[NavsimFrame]) -> bool:
    return all((sensor_root / item.filename_jpg).exists() for item in frames)


def build_anchors_for_log(
    frames_raw: List[Dict[str, Any]],
    log_name: str,
    sensor_root: Path,
    camera_channel: str,
    history_num_frames: int,
    future_image_offsets: List[float],
    future_steps: int,
    future_step_time_s: float,
    sample_stride: int,
    future_image_policy: str,
    max_anchors: int,
) -> List[ConsistencyAnchor]:
    frames = [
        parsed for parsed in (_frame_to_navsim(frame, camera_channel) for frame in frames_raw)
        if parsed is not None
    ]
    if len(frames) < history_num_frames + future_steps:
        return []

    interval = max(float(future_step_time_s), 1e-6)
    future_image_deltas = [max(1, int(round(offset / interval))) for offset in future_image_offsets]
    anchors: List[ConsistencyAnchor] = []

    last_start = len(frames) - max(future_steps, max(future_image_deltas, default=1))
    for cur_idx in range(history_num_frames - 1, last_start, max(1, sample_stride)):
        history = frames[cur_idx - history_num_frames + 1: cur_idx + 1]
        current = frames[cur_idx]
        future_traj_frames = frames[cur_idx + 1: cur_idx + 1 + future_steps]

        if len(future_traj_frames) < future_steps:
            continue
        if not _valid_image_paths(sensor_root, history):
            continue

        if future_image_policy == "future":
            future_images_frames = [frames[cur_idx + delta] for delta in future_image_deltas]
            if not _valid_image_paths(sensor_root, future_images_frames):
                continue
        elif future_image_policy == "history_tail":
            future_images_frames = history[-len(future_image_deltas):]
            if len(future_images_frames) < len(future_image_deltas):
                future_images_frames = [history[0]] * (len(future_image_deltas) - len(future_images_frames)) + future_images_frames
        else:
            future_images_frames = [current for _ in future_image_deltas]

        sample_token = current.token or str(current.timestamp) or str(cur_idx)
        anchors.append(
            ConsistencyAnchor(
                sample_id=f"{log_name}__{sample_token}",
                scene_name=log_name,
                timestamp_us=current.timestamp,
                history_images=[_relative_path(sensor_root, item.filename_jpg) for item in history],
                future_images=[_relative_path(sensor_root, item.filename_jpg) for item in future_images_frames],
                ego_state=[current.vx, current.vy, current.yaw, current.ax, current.yaw_rate],
                candidate_traj=_trajectory_from_frames(current, future_traj_frames),
            )
        )
        if 0 < max_anchors <= len(anchors):
            break
    return anchors


def main() -> None:
    args = parse_args()
    log_root = Path(args.navsim_log_root)
    sensor_root = Path(args.sensor_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_files = _iter_log_files(log_root)
    if args.max_logs > 0:
        log_files = log_files[: args.max_logs]
    print(f"发现 NAVSIM log files: {len(log_files)}")

    all_anchors: Dict[str, List[ConsistencyAnchor]] = {}
    skipped: Dict[str, str] = {}
    total_anchors = 0

    for log_path in log_files:
        try:
            with log_path.open("rb") as f:
                obj = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            skipped[str(log_path)] = f"pickle load failed: {exc}"
            continue
        frame_lists = _extract_frame_lists(obj)
        if not frame_lists:
            skipped[str(log_path)] = "no frame list"
            continue

        log_anchor_count = 0
        for idx, frame_list in enumerate(frame_lists):
            log_name = log_path.stem if len(frame_lists) == 1 else f"{log_path.stem}_{idx:05d}"
            anchors = build_anchors_for_log(
                frames_raw=frame_list,
                log_name=log_name,
                sensor_root=sensor_root,
                camera_channel=args.camera_channel,
                history_num_frames=args.history_num_frames,
                future_image_offsets=args.future_image_offsets,
                future_steps=args.future_steps,
                future_step_time_s=args.future_step_time_s,
                sample_stride=args.sample_stride,
                future_image_policy=args.future_image_policy,
                max_anchors=args.max_anchors_per_log,
            )
            if anchors:
                all_anchors[log_name] = anchors
                log_anchor_count += len(anchors)
                total_anchors += len(anchors)
                print(f"[OK] {log_name}: anchors={len(anchors)}")
            if 0 < args.max_anchors <= total_anchors:
                break
        if log_anchor_count == 0:
            skipped[str(log_path)] = "no usable anchors"
        if 0 < args.max_anchors <= total_anchors:
            break

    usable_logs = sorted(all_anchors.keys())
    if not usable_logs:
        preview = dict(list(skipped.items())[:20])
        raise RuntimeError(f"未找到可用 NAVSIM anchors. skipped preview={preview}")

    train_logs, val_logs = split_scenes(usable_logs, val_ratio=args.val_ratio, seed=args.seed)
    train_anchors = [anchor for name in train_logs for anchor in all_anchors[name]]
    val_anchors = [anchor for name in val_logs for anchor in all_anchors[name]]
    traj_scale = compute_traj_scale_factors(train_anchors)
    print(f"训练集 traj_scale_factors: {traj_scale}")

    lat_range = tuple(args.perturb_lateral_range)
    hdg_range = tuple(args.perturb_heading_range)
    spd_range = tuple(args.perturb_speed_range)
    train_rows = serialize_split(
        train_anchors,
        seed=args.seed,
        min_gap=args.min_negative_index_gap,
        lateral_range=lat_range,
        heading_range=hdg_range,
        speed_range=spd_range,
        time_shift_future_steps=args.time_shift_future_steps,
        add_reverse_traj=args.add_reverse_traj,
    )
    val_rows = serialize_split(
        val_anchors,
        seed=args.seed + 1,
        min_gap=max(1, args.min_negative_index_gap // 2),
        lateral_range=lat_range,
        heading_range=hdg_range,
        speed_range=spd_range,
        time_shift_future_steps=args.time_shift_future_steps,
        add_reverse_traj=args.add_reverse_traj,
    )

    train_path = output_dir / "consistency_train.jsonl"
    val_path = output_dir / "consistency_val.jsonl"
    summary_path = output_dir / "consistency_index_summary.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "navsim",
                "navsim_log_root": str(log_root),
                "sensor_root": str(sensor_root),
                "camera_channel": args.camera_channel,
                "future_image_policy": args.future_image_policy,
                "num_train_anchors": len(train_anchors),
                "num_val_anchors": len(val_anchors),
                "num_train_rows": len(train_rows),
                "num_val_rows": len(val_rows),
                "traj_scale_factors": traj_scale,
                "train_source_types": count_source_types(train_rows),
                "val_source_types": count_source_types(val_rows),
                "skipped": skipped,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n写入训练索引: {train_path} ({len(train_rows)} 条)")
    print(f"写入验证索引: {val_path} ({len(val_rows)} 条)")
    print(f"写入摘要:     {summary_path}")


if __name__ == "__main__":
    main()

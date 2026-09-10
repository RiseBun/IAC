"""Strict JSONL contract for image-side evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .image_geometry import validate_image_geometry_adapter


def _json_default(value: Any) -> Any:
    """Serialize numpy scalars/arrays emitted by optional diagnostics."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=_json_default) + "\n")


def _matrix(value: Any, shape: tuple[int, int], field: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field} must be a finite {shape} matrix")
    return matrix


def validate_record(
    row: dict[str, Any],
    *,
    manifest_root: Path,
    require_candidates: bool = True,
    require_intrinsics_source_size: bool = False,
    expected_intrinsics_source_size: tuple[int, int] | None = None,
    require_trusted_gt_candidate: bool = False,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "")
    if not sample_id:
        raise ValueError("sample_id is required")
    explicit_history = row.get("history_frame_paths") is not None
    explicit_future = row.get("future_frame_paths") is not None
    if explicit_history or explicit_future:
        if not (explicit_history and explicit_future):
            raise ValueError(
                f"{sample_id}: history_frame_paths and future_frame_paths must be provided together"
            )
        history = [str(value) for value in row.get("history_frame_paths") or []]
        future = [str(value) for value in row.get("future_frame_paths") or []]
        native_protocol = str((row.get("metadata") or {}).get("protocol") or "").startswith("wam-native")
        if native_protocol:
            if len(history) < 2 or len(future) < 1:
                raise ValueError(f"{sample_id}: native protocol requires at least 2 history and 1 future frame")
        elif len(history) != 4 or len(future) not in (4, 8):
            raise ValueError(
                f"{sample_id}: the explicit protocol requires 4 history and either 4 or 8 future frames"
            )
        history_times = np.asarray(row.get("history_times_s"), dtype=np.float64)
        future_times = np.asarray(row.get("future_times_s"), dtype=np.float64)
        if history_times.shape != (len(history),) or future_times.shape != (len(future),):
            raise ValueError(f"{sample_id}: history/future timestamps must match their frame counts")
        if not np.all(np.diff(history_times) > 0.0) or not np.all(np.diff(future_times) > 0.0):
            raise ValueError(f"{sample_id}: history and future times must be strictly increasing")
        if future_times[0] <= history_times[-1]:
            raise ValueError(f"{sample_id}: future must start after the last history frame")
        frames = history + future
        timestamps = np.concatenate([history_times, future_times])
        history_count = len(history)
        protocol_variant = (
            f"native_history{len(history)}_future{len(future)}"
            if native_protocol
            else f"history4_future{len(future)}"
        )
    else:
        # Compatibility for the first exploratory manifests. New experiments
        # should use the explicit 4+4 fields above.
        frames = [str(value) for value in row.get("frame_paths") or []]
        if len(frames) < 2:
            raise ValueError(f"{sample_id}: at least two frame_paths are required")
        timestamps = np.asarray(row.get("frame_times_s"), dtype=np.float64)
        if timestamps.shape != (len(frames),) or not np.all(np.diff(timestamps) > 0.0):
            raise ValueError(f"{sample_id}: frame_times_s must be strictly increasing")
        history_count = 1
        protocol_variant = "legacy_anchor_future"
    resolved_frames = []
    for value in frames:
        path = Path(value).expanduser()
        resolved_frames.append(str(path if path.is_absolute() else manifest_root / path))
    intrinsics = _matrix(row.get("intrinsics"), (3, 3), "intrinsics")
    camera_to_ego = _matrix(row.get("camera_to_ego"), (4, 4), "camera_to_ego")
    candidates = list(row.get("candidates") or [])
    if require_candidates and len(candidates) < 2:
        raise ValueError(f"{sample_id}: at least two candidates are required")
    candidate_ids: set[str] = set()
    normalized_candidates = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError(f"{sample_id}: candidate IDs must be unique and non-empty")
        candidate_ids.add(candidate_id)
        trajectory = np.asarray(candidate.get("trajectory"), dtype=np.float64)
        future_count = len(frames) - history_count
        if trajectory.shape != (future_count, 3) or not np.all(np.isfinite(trajectory)):
            raise ValueError(
                f"{sample_id}/{candidate_id}: trajectory must have shape "
                f"[{future_count},3]"
            )
        prior = float(candidate.get("prior", 1.0))
        if not np.isfinite(prior) or prior <= 0.0:
            raise ValueError(f"{sample_id}/{candidate_id}: prior must be positive")
        normalized_candidate = {
            "candidate_id": candidate_id,
            "trajectory": trajectory,
            "prior": prior,
        }
        if candidate.get("counterfactual") is not None:
            normalized_candidate["counterfactual"] = dict(candidate["counterfactual"])
        if candidate.get("parent_candidate_id") is not None:
            normalized_candidate["parent_candidate_id"] = str(candidate["parent_candidate_id"])
        for field in (
            "feasibility_label", "support_label", "feasibility_reason",
            "trajectory_source",
        ):
            if candidate.get(field) is not None:
                normalized_candidate[field] = str(candidate[field])
        for field in ("offroad", "collision"):
            if candidate.get(field) is not None:
                normalized_candidate[field] = bool(candidate[field])
        normalized_candidates.append(normalized_candidate)
    gt_candidate_id = row.get("gt_candidate_id")
    if gt_candidate_id is not None and str(gt_candidate_id) not in candidate_ids:
        raise ValueError(f"{sample_id}: gt_candidate_id is absent from candidates")
    if gt_candidate_id is not None:
        gt_candidate = next(
            candidate
            for candidate in normalized_candidates
            if candidate["candidate_id"] == str(gt_candidate_id)
        )
        source = str(gt_candidate.get("trajectory_source") or "")
        if str(gt_candidate_id) == "wam_action_head" or source == "wam_action_head":
            raise ValueError(
                f"{sample_id}: wam_action_head is a model output and cannot be gt_candidate_id"
            )
        if require_trusted_gt_candidate and source not in {
            "navsim_logged_realized",
            "private_realized_gt",
        }:
            raise ValueError(
                f"{sample_id}: gt_candidate_id requires an explicit trusted trajectory_source"
            )
    metric_depth_path = row.get("metric_depth_path")
    if metric_depth_path is not None:
        depth_path = Path(str(metric_depth_path)).expanduser()
        metric_depth_path = str(
            depth_path if depth_path.is_absolute() else manifest_root / depth_path
        )
    metadata = dict(row.get("metadata") or {})
    intrinsics_source_size = row.get("intrinsics_source_size")
    if require_intrinsics_source_size and intrinsics_source_size is None:
        raise ValueError(
            f"{sample_id}: intrinsics_source_size is required by the calibration contract"
        )
    if intrinsics_source_size is not None:
        if (
            not isinstance(intrinsics_source_size, (list, tuple))
            or len(intrinsics_source_size) != 2
            or any(int(value) <= 0 for value in intrinsics_source_size)
        ):
            raise ValueError(f"{sample_id}: intrinsics_source_size must be [width,height]")
        intrinsics_source_size = tuple(int(value) for value in intrinsics_source_size)
        if (
            expected_intrinsics_source_size is not None
            and intrinsics_source_size != tuple(expected_intrinsics_source_size)
        ):
            raise ValueError(
                f"{sample_id}: intrinsics_source_size {intrinsics_source_size} does not "
                f"match frozen calibration size {tuple(expected_intrinsics_source_size)}"
            )
    image_geometry_adapter = validate_image_geometry_adapter(
        row.get("image_geometry_adapter"),
        frame_count=len(frames),
        intrinsics_source_size=intrinsics_source_size,
    )
    # Preserve native WAM fields without forcing every producer to duplicate
    # them under metadata.  They are optional for image-only experiments but
    # become available to realized-state evaluation when present.
    for field in (
        "source_key", "scene_name", "timestamp_us", "history_ego_state",
        "native_action_condition", "action_condition", "action_trajectory",
        "realized_future_ego_state", "task_success",
    ):
        if field in row and field not in metadata:
            metadata[field] = row[field]
    return {
        "sample_id": sample_id,
        "scene_id": str(row.get("scene_id") or sample_id),
        "frame_paths": resolved_frames,
        "frame_times_s": timestamps,
        "history_frame_paths": resolved_frames[:history_count],
        "future_frame_paths": resolved_frames[history_count:],
        "history_times_s": timestamps[:history_count],
        # Future knot times are expressed from the anchor (last history frame),
        # which makes speed and curvature summaries independent of dataset epoch.
        "future_times_s": timestamps[history_count:] - timestamps[history_count - 1],
        "anchor_time_s": float(timestamps[history_count - 1]),
        "history_count": history_count,
        "protocol_variant": protocol_variant,
        "intrinsics": intrinsics,
        "intrinsics_source_size": intrinsics_source_size,
        "image_geometry_adapter": image_geometry_adapter,
        "distortion": np.asarray(row.get("distortion") or [], dtype=np.float64),
        "camera_to_ego": camera_to_ego,
        "metric_depth_path": metric_depth_path,
        "metric_depth_source": (
            str(row["metric_depth_source"])
            if row.get("metric_depth_source") is not None
            else None
        ),
        "candidates": normalized_candidates,
        "gt_candidate_id": str(gt_candidate_id) if gt_candidate_id is not None else None,
        "metadata": metadata,
    }

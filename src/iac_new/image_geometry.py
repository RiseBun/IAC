"""Fail-closed image-geometry adapters for heterogeneous WAM renderers.

Adapters may normalize representation differences such as resize or crop, but
must never depend on a candidate, ground truth, or a fitted trajectory.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _size(value: Any, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(int(item) <= 0 for item in value)
    ):
        raise ValueError(f"{field} must be [width,height]")
    return tuple(int(item) for item in value)


def _source_to_frame(
    operation: str,
    source_size: tuple[int, int],
    frame_size: tuple[int, int],
    crop_xywh: Any = None,
) -> np.ndarray:
    source_width, source_height = source_size
    frame_width, frame_height = frame_size
    if operation in {"identity", "direct_resize"}:
        if operation == "identity" and frame_size != source_size:
            raise ValueError("identity image geometry requires equal source and frame sizes")
        return np.asarray([
            [frame_width / source_width, 0.0, 0.0],
            [0.0, frame_height / source_height, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    if operation == "crop_resize":
        if not isinstance(crop_xywh, (list, tuple)) or len(crop_xywh) != 4:
            raise ValueError("crop_resize requires crop_xywh=[x,y,width,height]")
        x, y, width, height = (float(item) for item in crop_xywh)
        if width <= 0.0 or height <= 0.0 or x < 0.0 or y < 0.0:
            raise ValueError("crop_xywh must describe a positive crop inside the source image")
        if x + width > source_width + 1e-6 or y + height > source_height + 1e-6:
            raise ValueError("crop_xywh exceeds the calibration source image")
        sx = frame_width / width
        sy = frame_height / height
        return np.asarray([
            [sx, 0.0, -sx * x],
            [0.0, sy, -sy * y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    raise ValueError(f"unsupported image geometry operation: {operation}")


def validate_image_geometry_adapter(
    value: Any,
    *,
    frame_count: int,
    intrinsics_source_size: tuple[int, int] | None,
) -> dict[str, Any] | None:
    """Validate and expand a candidate-blind piecewise image transform."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("image_geometry_adapter must be an object")
    adapter_id = str(value.get("adapter_id") or "")
    if not adapter_id:
        raise ValueError("image_geometry_adapter.adapter_id is required")
    if str(value.get("schema")) != "iac-image-geometry-v1":
        raise ValueError("image_geometry_adapter.schema must be iac-image-geometry-v1")
    source_size = _size(
        value.get("calibration_source_size"),
        "image_geometry_adapter.calibration_source_size",
    )
    if intrinsics_source_size is None:
        raise ValueError("image_geometry_adapter requires intrinsics_source_size")
    if source_size != tuple(intrinsics_source_size):
        raise ValueError(
            "image_geometry_adapter calibration_source_size does not match "
            "intrinsics_source_size"
        )
    groups = value.get("frame_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("image_geometry_adapter.frame_groups must be a non-empty list")
    transforms: list[dict[str, Any] | None] = [None] * frame_count
    normalized_groups = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError("image_geometry_adapter frame groups must be objects")
        indices = [int(item) for item in group.get("indices") or []]
        if not indices:
            raise ValueError(f"image geometry frame group {group_index} has no indices")
        operation = str(group.get("operation") or "")
        frame_size = _size(
            group.get("frame_size"),
            f"image_geometry_adapter.frame_groups[{group_index}].frame_size",
        )
        matrix = _source_to_frame(
            operation, source_size, frame_size, group.get("crop_xywh")
        )
        normalized_group = {
            "indices": indices,
            "operation": operation,
            "frame_size": frame_size,
            "source_to_frame": matrix,
        }
        if group.get("crop_xywh") is not None:
            normalized_group["crop_xywh"] = tuple(float(item) for item in group["crop_xywh"])
        normalized_groups.append(normalized_group)
        for index in indices:
            if index < 0 or index >= frame_count:
                raise ValueError(f"image geometry frame index {index} is out of range")
            if transforms[index] is not None:
                raise ValueError(f"image geometry frame index {index} is assigned more than once")
            transforms[index] = {
                "frame_size": frame_size,
                "source_to_frame": matrix,
                "operation": operation,
            }
    missing = [index for index, item in enumerate(transforms) if item is None]
    if missing:
        raise ValueError(f"image geometry adapter does not cover frame indices {missing}")
    return {
        "schema": "iac-image-geometry-v1",
        "adapter_id": adapter_id,
        "calibration_source_size": source_size,
        "frame_groups": normalized_groups,
        "frame_transforms": transforms,
        "provenance": dict(value.get("provenance") or {}),
    }

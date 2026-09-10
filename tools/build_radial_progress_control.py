#!/usr/bin/env python3
"""Build a validation-only radial-expansion control for progress measurements."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _warp_radial(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale < 1.0:
        raise ValueError("radial scale must be at least one")
    height, width = frame.shape[:2]
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    matrix = np.asarray(
        [[scale, 0.0, (1.0 - scale) * center_x],
         [0.0, scale, (1.0 - scale) * center_y]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def build_control_rows(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    source_role: str = "left",
    strong_expansion_per_step: float = 0.02,
    weak_expansion_per_step: float = 0.0,
    maximum_absolute_yaw_rad: float = 0.05,
) -> list[dict[str, Any]]:
    if not 0.0 <= weak_expansion_per_step < strong_expansion_per_step:
        raise ValueError("expansion rates must satisfy 0 <= weak < strong")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    seen = set()
    for row in rows:
        source = str(row.get("source_key") or "")
        if not source or row.get("branch_role") != source_role or source in seen:
            continue
        action = np.asarray(row["action_trajectory"], dtype=np.float64)
        if action.ndim != 2 or action.shape[1] < 3:
            raise ValueError(f"{source}: action_trajectory must have shape [T,>=3]")
        if float(np.max(np.abs(action[:, 2]))) > maximum_absolute_yaw_rad:
            continue
        seen.add(source)
        selected.append(row)

    output = []
    for index, original in enumerate(selected):
        source = str(original["source_key"])
        futures = list(original.get("future_frame_paths") or original.get("future_images") or [])
        if not futures:
            raise ValueError(f"{source}: future frames are required")
        for role, rate, action_scale in (
            ("left", strong_expansion_per_step, 1.25),
            ("right", weak_expansion_per_step, 0.75),
        ):
            branch_dir = output_dir / f"sample_{index:06d}" / role
            branch_dir.mkdir(parents=True, exist_ok=True)
            frame_paths = []
            for step, source_path in enumerate(futures, start=1):
                frame = np.asarray(Image.open(source_path).convert("RGB"))
                warped = _warp_radial(frame, 1.0 + rate * step)
                path = branch_dir / f"future_{step:02d}.png"
                Image.fromarray(warped).save(path)
                frame_paths.append(str(path.resolve()))

            row = copy.deepcopy(original)
            row["sample_id"] = f"{source}::{role}"
            row["counterfactual_group_id"] = source
            row["branch_role"] = role
            row["future_frame_paths"] = frame_paths
            row["future_images"] = frame_paths
            action = np.asarray(original["action_trajectory"], dtype=np.float64).copy()
            action[:, :2] *= action_scale
            row["action_trajectory"] = action.tolist()
            row["future_images_source"] = "controlled_radial_expansion"
            metadata = copy.deepcopy(row.get("metadata") or {})
            metadata["controlled_degradation"] = {
                "validation_only": True,
                "name": "radial_progress_expansion",
                "source_role": source_role,
                "expansion_per_step": rate,
                "maximum_absolute_yaw_rad": maximum_absolute_yaw_rad,
                "yaw_trajectory_unchanged": True,
            }
            row["metadata"] = metadata
            output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--source-role", default="left")
    parser.add_argument("--strong-expansion-per-step", type=float, default=0.02)
    parser.add_argument("--weak-expansion-per-step", type=float, default=0.0)
    parser.add_argument("--maximum-absolute-yaw-rad", type=float, default=0.05)
    args = parser.parse_args()
    output = build_control_rows(
        _read_jsonl(args.input),
        args.output_dir,
        source_role=args.source_role,
        strong_expansion_per_step=args.strong_expansion_per_step,
        weak_expansion_per_step=args.weak_expansion_per_step,
        maximum_absolute_yaw_rad=args.maximum_absolute_yaw_rad,
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "\n".join(json.dumps(row) for row in output) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(output), "pairs": len(output) // 2}))


if __name__ == "__main__":
    main()

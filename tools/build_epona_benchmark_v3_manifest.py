#!/usr/bin/env python3
"""Adapt Epona common-random left/right generations to Benchmark v3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def compose_relative_trajectory(relative: list[list[float]]) -> list[list[float]]:
    """Compose Epona [dx, dy, yaw_deg] increments into SE(2) poses in radians."""
    x = y = yaw = 0.0
    poses: list[list[float]] = []
    for item in relative:
        if len(item) < 3:
            raise ValueError("Epona action increment must contain dx, dy, yaw_deg")
        dx, dy, yaw_deg = map(float, item[:3])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        x += cosine * dx - sine * dy
        y += sine * dx + cosine * dy
        yaw += math.radians(yaw_deg)
        poses.append([x, y, yaw])
    return poses


def adapt_row(row: dict[str, Any]) -> dict[str, Any] | None:
    role = str(row.get("branch_mode") or "")
    if role not in {"left", "right"}:
        return None
    histories = list(row.get("history_images") or [])
    futures = list(row.get("future_images") or [])
    relative = list(row.get("action_trajectory") or [])
    if len(histories) < 4 or len(futures) != 8 or len(relative) != 8:
        raise ValueError(f"invalid Epona temporal contract for {row.get('source_key')}::{role}")
    group = str(row.get("counterfactual_group_id") or row.get("source_key") or "")
    randomness = row.get("randomness_contract") or {}
    if not group or randomness.get("common_random_numbers") is not True:
        raise ValueError(f"missing counterfactual identity/common randomness for {group!r}")
    trajectory = compose_relative_trajectory(relative)
    selected = [1, 3, 5, 7]
    return {
        "sample_id": f"{group}::{role}",
        "source_key": group,
        "counterfactual_group_id": group,
        "branch_role": role,
        "history_fingerprint": group,
        "nuisance_seed": randomness.get("seed"),
        "intervention_type": "action_trajectory_perturbation",
        "history_frame_paths": histories[-4:],
        "future_frame_paths": [futures[index] for index in selected],
        "history_times_s": [-1.5, -1.0, -0.5, 0.0],
        "future_times_s": [1.0, 2.0, 3.0, 4.0],
        "intrinsics": row.get("camera_intrinsic"),
        "intrinsics_source_size": [1920, 1120],
        "distortion": row.get("camera_distortion"),
        "camera_to_ego": row.get("camera_to_ego"),
        "action_trajectory": [trajectory[index] for index in selected],
        "action_trajectory_source": "epona_injected_relative_action",
        "future_images_source": "epona_generated_common_random",
        "wam_model_id": "epona_nuplan",
        "metadata": {
            "candidate_blind_image_branch": True,
            "action_waypoint_used_by_image_branch": True,
            "action_injection_verified": bool(row.get("action_injection_verified")),
            "intervention_variant": row.get("intervention_variant"),
        },
        "lineage": {
            "source_sample": row.get("source_sample"),
            "runner_branch": role,
            "same_history_seed": True,
            "native_future_times_s": row.get("future_times_s"),
            "native_selected_indices_zero_based": selected,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Epona manifest must be a JSON list")
    output = [adapted for row in payload if (adapted := adapt_row(row)) is not None]
    groups: dict[str, set[str]] = {}
    for row in output:
        groups.setdefault(row["counterfactual_group_id"], set()).add(row["branch_role"])
    incomplete = [group for group, roles in groups.items() if roles != {"left", "right"}]
    if incomplete:
        raise ValueError(f"{len(incomplete)} Epona groups lack a complete left/right pair")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"rows": len(output), "pairs": len(groups), "output": str(args.output)}))


if __name__ == "__main__":
    main()

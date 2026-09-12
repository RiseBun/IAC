#!/usr/bin/env python3
"""Build the shared Step1 evidence object from archived raw flow arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.road_structure import flow_structure_profile
from iac_new.step1_evidence import assemble_step1_evidence, response_channel_evidence
from iac_new.scoring import polygon_mask


def _records(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"manifest.json", "score_roi.json", "twin_controls.json"}:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if all(key in value for key in ("source_key", "branch_role", "trajectory", "flow_path", "valid_path", "intrinsics")):
            value["_root"] = str(path.parent)
            output.append(value)
    return output


def run(root: Path) -> dict[str, Any]:
    records = _records(root)
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["source_key"]), {})[str(record["branch_role"])] = record
    rows = []
    pair_responses = []
    for source, branches in sorted(grouped.items()):
        if not {"left", "right"}.issubset(branches):
            continue
        profiles: dict[str, dict[str, Any]] = {}
        for role, record in branches.items():
            root_path = Path(record["_root"])
            flow = np.load(root_path / record["flow_path"])
            valid = np.load(root_path / record["valid_path"]).astype(bool)
            height, width = flow.shape[1:3]
            roi = polygon_mask(height, width, [[0.08, 0.98], [0.92, 0.98], [0.63, 0.53], [0.37, 0.53]])
            profile = flow_structure_profile(
                flow,
                valid.astype(np.float32),
                roi,
                np.asarray(record["intrinsics"], dtype=float),
                min_flow_px=0.5,
                min_points=100,
                min_spatial_cells=6,
                max_points=3000,
            )
            profiles[role] = profile
        left, right = branches["left"], branches["right"]
        left_action = np.asarray(left["trajectory"], dtype=float)
        right_action = np.asarray(right["trajectory"], dtype=float)
        action_deltas = {
            "yaw_direction": float(left_action[-1, 2] - right_action[-1, 2]),
            "lateral_direction": float(left_action[-1, 1] - right_action[-1, 1]),
            "longitudinal_order": float(left_action[-1, 0] - right_action[-1, 0]),
        }
        response = response_channel_evidence(profiles["left"], profiles["right"], action_deltas)
        pair_responses.append(response)
        for role in ("left", "right"):
            evidence = assemble_step1_evidence(
                branch_profile=profiles[role],
                left_profile=profiles["left"],
                right_profile=profiles["right"],
                action_delta=action_deltas["yaw_direction"],
                flow_support_fraction=float(np.mean(np.load(Path(branches[role]["_root"]) / branches[role]["valid_path"]).astype(bool))),
            )
            evidence["source_key"] = source
            evidence["branch_role"] = role
            evidence["counterfactual_pair_evidence"] = response
            rows.append(evidence)
    temporal = [row["channels"]["temporal_motion"] for row in rows]
    response = pair_responses
    return {
        "protocol": "iac-step1-evidence-layer-raw-flow-run-v1",
        "raw_video_rerun": True,
        "root": str(root),
        "branch_count": len(rows),
        "pair_count": len(response),
        "temporal_scored_fraction": float(np.mean([row["status"] == "scored" for row in temporal])) if temporal else None,
        "response_scored_fraction": {
            channel: float(np.mean([row[channel]["status"] == "scored" for row in response]))
            if response else None
            for channel in ("yaw_direction", "lateral_direction", "longitudinal_order")
        },
        "grounding_status": "unavailable_without_external_reference_profile",
        "depth_status": "not_provided_optional_adapter",
        "rows": rows,
        "claim_boundary": "This run validates the shared evidence interface on raw flows. It is not a formal MAS/RCS/GS promotion result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("branch_count", "pair_count", "temporal_scored_fraction", "response_scored_fraction", "grounding_status", "depth_status")}, indent=2))


if __name__ == "__main__":
    main()

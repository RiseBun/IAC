#!/usr/bin/env python3
"""Build standard flow-structure measurements from precomputed flow arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iac_new.road_structure import flow_structure_profile
from iac_new.scoring import polygon_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-flow-px", type=float, default=0.5)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--min-spatial-cells", type=int, default=6)
    parser.add_argument("--max-points", type=int, default=3000)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    output: list[dict[str, Any]] = []
    for record in rows:
        flow_path = args.manifest.parent / str(record["flow_path"])
        valid_path = args.manifest.parent / str(record["valid_path"])
        flow = np.asarray(np.load(flow_path), dtype=np.float64)
        valid = np.asarray(np.load(valid_path), dtype=bool)
        if valid.shape != flow.shape[:-1]:
            raise ValueError(f"{record.get('sample_id')}: valid mask shape mismatch")
        height, width = flow.shape[1:3]
        roi = polygon_mask(
            height,
            width,
            [[0.08, 0.98], [0.92, 0.98], [0.63, 0.53], [0.37, 0.53]],
        )
        profile = flow_structure_profile(
            flow,
            valid.astype(np.float64),
            roi,
            np.asarray(record["intrinsics"], dtype=np.float64),
            min_flow_px=args.min_flow_px,
            min_points=args.min_points,
            min_spatial_cells=args.min_spatial_cells,
            max_points=args.max_points,
        )
        output.append({
            "sample_id": record.get("sample_id"),
            "scene_id": record.get("scene_id"),
            "source_key": record.get("source_key"),
            "branch_role": record.get("branch_role"),
            "future_times_s": record.get("future_times_s"),
            "candidate_bank_used_by_measurement": False,
            "flow_structure": profile,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(output), "output": str(args.output)}))


if __name__ == "__main__":
    main()

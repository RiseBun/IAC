#!/usr/bin/env python3
"""Remove the legacy false-GT label from WAM-derived manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def repair_row(
    row: dict[str, Any],
    *,
    intrinsics_source_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "<unknown>")
    if row.get("future_images_source") not in {"wam_generated", "logged_real"}:
        raise ValueError(
            f"{sample_id}: only WAM generated/real-counterpart rows may be repaired"
        )
    if row.get("action_trajectory_source") != "wam_action_head":
        raise ValueError(f"{sample_id}: action source is not the WAM action head")
    if row.get("gt_candidate_id") != "wam_action_head":
        raise ValueError(
            f"{sample_id}: expected legacy gt_candidate_id=wam_action_head"
        )
    candidates = list(row.get("candidates") or [])
    action_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_id") == "wam_action_head"
    ]
    if len(action_candidates) != 1:
        raise ValueError(f"{sample_id}: expected exactly one wam_action_head candidate")
    repaired = dict(row)
    repaired_candidates = [dict(candidate) for candidate in candidates]
    for candidate in repaired_candidates:
        if candidate.get("candidate_id") == "wam_action_head":
            candidate["trajectory_source"] = "wam_action_head"
            candidate.setdefault("support_label", "independent_action_head_reference")
    repaired["candidates"] = repaired_candidates
    repaired["gt_candidate_id"] = None
    existing_source_size = repaired.get("intrinsics_source_size")
    if intrinsics_source_size is not None:
        expected = [int(value) for value in intrinsics_source_size]
        if existing_source_size is not None and list(existing_source_size) != expected:
            raise ValueError(
                f"{sample_id}: existing intrinsics_source_size "
                f"{existing_source_size} conflicts with {expected}"
            )
        repaired["intrinsics_source_size"] = expected
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--intrinsics-source-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Explicitly attach the verified calibration coordinate size.",
    )
    args = parser.parse_args()
    rows = [
        repair_row(
            json.loads(line),
            intrinsics_source_size=(
                tuple(args.intrinsics_source_size)
                if args.intrinsics_source_size is not None
                else None
            ),
        )
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()

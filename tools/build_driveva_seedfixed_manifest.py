#!/usr/bin/env python3
"""Join seed-controlled DriveVA outputs to their candidate-blind NAVSIM inputs."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def read_many(patterns: list[str]) -> list[dict]:
    rows: list[dict] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--generated", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()

    base_rows = read_many([args.base])
    base_by_branch = {str(row["branch_id"]): row for row in base_rows}
    generated_rows = read_many(args.generated)
    output = []
    seen: set[str] = set()
    for generated in generated_rows:
        branch_id = str(generated["branch_id"])
        if branch_id in seen:
            raise ValueError(f"duplicate generated branch_id: {branch_id}")
        seen.add(branch_id)
        base = base_by_branch.get(branch_id)
        if base is None:
            raise ValueError(f"no base row for branch_id: {branch_id}")
        if int(generated["actual_generation_seed"]) != int(base["nuisance_seed"]):
            raise ValueError(f"seed mismatch: {branch_id}")
        future = list(generated["future_images"])
        if len(future) != 8:
            raise ValueError(f"{branch_id}: expected 8 generated frames")
        if args.check_files:
            missing = [path for path in future if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
        row = dict(base)
        row.update(
            {
                "sample_id": str(generated["sample_id"]),
                "future_images": future,
                "future_frame_paths": future,
                "future_times_s": list(generated["future_times_s"]),
                "future_images_source": "wam_generated",
                "action_trajectory": list(generated["action_trajectory"]),
                "action_trajectory_source": "wam_native_action_head",
                "wam_model_id": "driveva_navsim_seedfixed",
                "nuisance_seed": int(generated["actual_generation_seed"]),
                "actual_generation_seed": int(generated["actual_generation_seed"]),
                "intrinsics_source_size": [1920, 1080],
            }
        )
        row.pop("realized_future_ego_state", None)
        output.append(row)

    expected = set(base_by_branch)
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"generated set incomplete: {len(seen)}/{len(expected)}; first missing={missing[:1]}")
    groups: dict[str, list[dict]] = {}
    for row in output:
        groups.setdefault(str(row["counterfactual_group_id"]), []).append(row)
    if len(groups) * 2 != len(output):
        raise ValueError("every counterfactual group must contain exactly two branches")
    for group, rows in groups.items():
        if {row["branch_role"] for row in rows} != {"left", "right"}:
            raise ValueError(f"{group}: missing left/right")
        if len({row["actual_generation_seed"] for row in rows}) != 1:
            raise ValueError(f"{group}: left/right generation seed differs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(output),
                "pairs": len(groups),
                "seed_contract": "pass",
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()

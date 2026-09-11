#!/usr/bin/env python3
"""Annotate already generated Epona manifests with the pure-speed protocol.

This is useful when a legacy runner produced correct images but used its old
matched-DriveWAM labels.  It changes metadata only and never changes image
paths or trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _roles(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.glob("shard_*/manifest.json")):
        for row in _read(path):
            result[str(row["source_key"])] = str(row.get("speed_role") or "unknown")
    return result


def annotate(output_root: Path, fast_root: Path, slow_root: Path) -> int:
    roles = {**_roles(slow_root), **_roles(fast_root)}
    changed = 0
    for path in sorted(output_root.glob("shard_*/manifest.json")):
        rows = _read(path)
        for row in rows:
            source = str(row.get("source_key") or "")
            branch = str(row.get("branch_mode") or row.get("branch_role") or "")
            role = {"left": "fast", "right": "slow"}.get(branch, roles.get(source))
            if role is None:
                continue
            row["protocol"] = "iac-pure-speed-twin-v1"
            row["intervention_type"] = "pure_speed"
            row["speed_role"] = role
            row["future_images_source"] = "epona_generated_pure_speed_action"
            row["action_trajectory_source"] = "validation_only_scaled_action"
            row.setdefault("metadata", {})["protocol"] = "iac-pure-speed-twin-v1"
            row["metadata"]["intervention_type"] = "pure_speed"
            row["metadata"]["speed_role"] = role
            changed += 1
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in sorted(output_root.glob("shard_*/flow_manifest.jsonl")):
        rows = _read(path)
        for row in rows:
            source = str(row.get("source_key") or "")
            branch = str(row.get("branch_role") or row.get("branch_mode") or "")
            role = {"left": "fast", "right": "slow"}.get(branch, roles.get(source))
            if role is None:
                continue
            row["protocol"] = "iac-pure-speed-twin-v1"
            row["intervention_type"] = "pure_speed"
            row["speed_role"] = role
            row["future_images_source"] = "epona_generated_pure_speed_action"
            row["action_trajectory_source"] = "validation_only_scaled_action"
            row.setdefault("metadata", {})["protocol"] = "iac-pure-speed-twin-v1"
            row["metadata"]["intervention_type"] = "pure_speed"
            row["metadata"]["speed_role"] = role
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fast-root", type=Path, required=True)
    parser.add_argument("--slow-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"annotated_rows": annotate(args.output_root, args.fast_root, args.slow_root)}))


if __name__ == "__main__":
    main()

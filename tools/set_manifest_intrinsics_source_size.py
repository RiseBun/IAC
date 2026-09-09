#!/usr/bin/env python3
"""Write a new manifest with an explicit camera-calibration image size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must differ; archived manifests are immutable")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    target = [int(args.width), int(args.height)]
    rows = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        existing = row.get("intrinsics_source_size")
        if existing is not None and list(existing) != target:
            raise ValueError(
                f"{row.get('sample_id')}: existing intrinsics_source_size {existing} conflicts with {target}"
            )
        rows.append({**row, "intrinsics_source_size": target})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "intrinsics_source_size": target}))


if __name__ == "__main__":
    main()

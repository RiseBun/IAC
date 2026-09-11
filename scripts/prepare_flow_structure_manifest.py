#!/usr/bin/env python3
"""Normalize generated-flow manifests for the candidate-blind flow scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(inputs: list[Path], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in inputs:
        for raw in _read(path):
            sample_id = str(raw.get("sample_id") or "")
            if not sample_id or sample_id in seen:
                raise ValueError(f"missing or duplicate sample_id: {sample_id}")
            history = list(raw.get("history_frame_paths") or [])
            future = list(raw.get("future_frame_paths") or [])
            if len(history) < 2 or not future:
                raise ValueError(f"{sample_id}: requires history and future frame paths")
            row = dict(raw)
            row["frame_paths"] = history + future
            row["history_count"] = len(history)
            row["future_count"] = len(future)
            row["candidate_bank_used_by_measurement"] = False
            rows.append(row)
            seen.add(sample_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    report = {
        "protocol": "iac-flow-structure-manifest-v1",
        "input_manifests": [str(path) for path in inputs],
        "row_count": len(rows),
        "branch_count": len(rows),
        "source_count": len({str(row.get("source_key") or "") for row in rows}),
        "output": str(output),
    }
    output.with_name(f"{output.stem}_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

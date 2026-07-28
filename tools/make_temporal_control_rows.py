#!/usr/bin/env python3
"""Create deterministic raw-frame reversal or shuffle control indices."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ordered_motion_common import load_rows, sha256, write_json, write_jsonl  # noqa: E402


def _sample_seed(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{sample_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _controlled_sequence(
    history: Sequence[Any],
    future: Sequence[Any],
    *,
    control: str,
    seed: int,
) -> tuple[List[Any], List[Any]]:
    history_count = len(history)
    combined = [*history, *future]
    if control == "reverse":
        controlled = list(reversed(combined))
    elif control == "shuffle":
        controlled = list(combined)
        random.Random(seed).shuffle(controlled)
    else:
        raise ValueError(f"unknown control: {control}")
    return controlled[:history_count], controlled[history_count:]


def transform(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_rows(Path(args.input_rows))
    output: List[Dict[str, Any]] = []
    empty_sequences = 0
    for raw in rows:
        row = dict(raw)
        history = list(row.get("history_images") or [])
        future = list(row.get("future_images") or [])
        if not history and not future:
            empty_sequences += 1
            output.append(row)
            continue
        controlled_history, controlled_future = _controlled_sequence(
            history,
            future,
            control=args.control,
            seed=_sample_seed(
                str(row.get("sample_id", "")),
                int(args.seed),
            ),
        )
        row["history_images"] = controlled_history
        row["future_images"] = controlled_future
        output.append(row)

    output_path = Path(args.output_rows)
    write_jsonl(output_path, output)
    summary = {
        "kind": "raw_frame_temporal_control_rows",
        "control": args.control,
        "seed": int(args.seed),
        "rows": len(output),
        "empty_sequences": empty_sequences,
        "sample_ids_preserved": True,
        "input_rows": str(args.input_rows),
        "output_rows": str(output_path),
        "output_sha256": sha256(output_path),
    }
    if args.output_summary:
        write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-rows", required=True)
    parser.add_argument("--output-rows", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--control", choices=("reverse", "shuffle"), required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    transform(parse_args())


if __name__ == "__main__":
    main()

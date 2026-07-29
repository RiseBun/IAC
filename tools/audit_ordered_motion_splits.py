#!/usr/bin/env python3
"""Audit group, scene/log and image overlap across ordered-motion splits."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ordered_motion_common import group_id, load_rows, sha256, write_json  # noqa: E402


def _named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--split must use NAME=PATH")
    name, raw = value.split("=", 1)
    if not name.strip() or not raw.strip():
        raise ValueError("--split must use non-empty NAME=PATH")
    return name.strip(), Path(raw)


def _scene(row: Mapping[str, Any]) -> str:
    for key in ("scene_name", "log_name", "log_id", "scene_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _images(row: Mapping[str, Any]) -> Iterable[str]:
    for key in ("history_images", "future_images", "images"):
        values = row.get(key, [])
        if isinstance(values, (list, tuple)):
            for value in values:
                if value is not None:
                    yield str(value)


def _sets(rows: Sequence[Mapping[str, Any]]) -> Dict[str, set[str]]:
    return {
        "sample_ids": {
            str(row.get("sample_id", ""))
            for row in rows
            if row.get("sample_id") is not None
        },
        "group_ids": {group_id(row) for row in rows},
        "scenes": {value for row in rows if (value := _scene(row))},
        "images": {value for row in rows for value in _images(row)},
    }


def _representatives(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = group_id(row)
        if key not in result or str(row.get("source_type", "")) == "gt_pos":
            result[key] = row
    return result


def _right_group_overlap(
    left_sets: Dict[str, set[str]],
    right_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    scene_hits = 0
    image_hits = 0
    for row in _representatives(right_rows).values():
        scene_hits += int(bool(_scene(row) and _scene(row) in left_sets["scenes"]))
        image_hits += int(bool(set(_images(row)) & left_sets["images"]))
    return {
        "right_groups_with_scene_seen_on_left": scene_hits,
        "right_groups_with_exact_image_seen_on_left": image_hits,
    }


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    named = [_named_path(value) for value in args.split]
    if len(named) < 2:
        raise ValueError("provide at least two --split NAME=PATH entries")
    rows_by_name = {name: load_rows(path) for name, path in named}
    sets_by_name = {
        name: _sets(rows)
        for name, rows in rows_by_name.items()
    }
    split_summary: Dict[str, Any] = {}
    for name, path in named:
        values = sets_by_name[name]
        split_summary[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": len(rows_by_name[name]),
            "groups": len(values["group_ids"]),
            "scenes": len(values["scenes"]),
            "images": len(values["images"]),
        }
    pairs: Dict[str, Any] = {}
    all_disjoint = True
    for left, right in itertools.combinations([name for name, _ in named], 2):
        left_sets = sets_by_name[left]
        right_sets = sets_by_name[right]
        overlap = {
            "sample_id_overlap": len(
                left_sets["sample_ids"] & right_sets["sample_ids"]
            ),
            "group_id_overlap": len(
                left_sets["group_ids"] & right_sets["group_ids"]
            ),
            "scene_overlap": len(
                left_sets["scenes"] & right_sets["scenes"]
            ),
            "image_path_overlap": len(
                left_sets["images"] & right_sets["images"]
            ),
            **_right_group_overlap(
                left_sets,
                rows_by_name[right],
            ),
        }
        overlap["strict_scene_and_image_disjoint"] = (
            overlap["sample_id_overlap"] == 0
            and overlap["group_id_overlap"] == 0
            and overlap["scene_overlap"] == 0
            and overlap["image_path_overlap"] == 0
        )
        all_disjoint = all_disjoint and bool(
            overlap["strict_scene_and_image_disjoint"]
        )
        pairs[f"{left}__vs__{right}"] = overlap
    summary = {
        "kind": "ordered_motion_split_independence_audit",
        "splits": split_summary,
        "pairs": pairs,
        "all_pairs_strict_scene_and_image_disjoint": all_disjoint,
    }
    write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if bool(args.require_strict_disjoint) and not all_disjoint:
        raise SystemExit(2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--require-strict-disjoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    audit(parse_args())


if __name__ == "__main__":
    main()

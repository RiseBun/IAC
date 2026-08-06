#!/usr/bin/env python3
"""Repair two grouped splits by assigning shared-image components atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_id(row: Mapping[str, Any]) -> str:
    explicit = row.get("group_id")
    if explicit not in (None, ""):
        return str(explicit)
    sample_id = str(row.get("sample_id", ""))
    if "__" in sample_id:
        group, _ = sample_id.rsplit("__", 1)
        if group:
            return group
    raise ValueError(f"cannot derive group_id from sample_id={sample_id!r}")


def _images(row: Mapping[str, Any]) -> set[str]:
    return {
        str(value)
        for key in ("history_images", "future_images", "images")
        for value in row.get(key, [])
        if value not in (None, "")
    }


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            self.parent[second] = first


def repartition(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    target_left_groups: int | None = None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any]]:
    combined = list(left_rows) + list(right_rows)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    origin: dict[str, str] = {}
    sample_ids: set[str] = set()
    for side, rows in (("left", left_rows), ("right", right_rows)):
        for row in rows:
            group = _group_id(row)
            if group in origin and origin[group] != side:
                raise ValueError(f"group {group!r} already appears in the other split")
            origin[group] = side
            grouped[group].append(row)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in sample_ids:
                raise ValueError(f"missing or duplicate sample_id {sample_id!r}")
            sample_ids.add(sample_id)

    target = target_left_groups if target_left_groups is not None else sum(
        side == "left" for side in origin.values()
    )
    if not 0 < target < len(grouped):
        raise ValueError("target_left_groups must leave at least one group per split")

    union_find = _UnionFind(sorted(grouped))
    image_owner: dict[str, str] = {}
    for group in sorted(grouped):
        for row in grouped[group]:
            for image in sorted(_images(row)):
                owner = image_owner.setdefault(image, group)
                union_find.union(group, owner)

    component_groups: dict[str, list[str]] = defaultdict(list)
    for group in sorted(grouped):
        component_groups[union_find.find(group)].append(group)
    components = sorted(
        (tuple(sorted(values)) for values in component_groups.values()),
        key=lambda values: (-len(values), values),
    )

    # State: left group count -> (moved groups, component assignment bits).
    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for component in components:
        size = len(component)
        left_count = sum(origin[group] == "left" for group in component)
        right_count = size - left_count
        updated: dict[int, tuple[int, tuple[int, ...]]] = {}
        for count, (cost, choices) in states.items():
            candidates = (
                (count + size, cost + right_count, choices + (1,)),
                (count, cost + left_count, choices + (0,)),
            )
            for new_count, new_cost, new_choices in candidates:
                if new_count > target:
                    continue
                previous = updated.get(new_count)
                candidate = (new_cost, new_choices)
                if previous is None or candidate < previous:
                    updated[new_count] = candidate
        states = updated
    if target not in states:
        reachable = sorted(states)
        raise ValueError(
            f"no image-disjoint assignment reaches {target} left groups; "
            f"reachable counts end at {reachable[-10:]}"
        )

    moved_count, choices = states[target]
    assignment: dict[str, str] = {}
    for component, choose_left in zip(components, choices):
        side = "left" if choose_left else "right"
        assignment.update({group: side for group in component})
    output_left = [row for row in combined if assignment[_group_id(row)] == "left"]
    output_right = [row for row in combined if assignment[_group_id(row)] == "right"]
    left_images = set().union(*(_images(row) for row in output_left))
    right_images = set().union(*(_images(row) for row in output_right))
    overlap = left_images & right_images
    if overlap:
        raise AssertionError(f"repartition left {len(overlap)} shared images")

    moved = sorted(group for group in grouped if assignment[group] != origin[group])
    cross_components = sum(
        any(origin[group] == "left" for group in component)
        and any(origin[group] == "right" for group in component)
        for component in components
    )
    summary = {
        "kind": "iac_image_component_repartition_v1",
        "input_groups": len(grouped),
        "target_left_groups": target,
        "output_left_groups": sum(side == "left" for side in assignment.values()),
        "output_right_groups": sum(side == "right" for side in assignment.values()),
        "image_components": len(components),
        "cross_origin_components": cross_components,
        "largest_component_groups": max(map(len, components)),
        "moved_groups": moved_count,
        "moved_group_ids": moved,
        "post_repartition_image_overlap": 0,
    }
    return output_left, output_right, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-rows", required=True)
    parser.add_argument("--right-rows", required=True)
    parser.add_argument("--output-left-rows", required=True)
    parser.add_argument("--output-right-rows", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--target-left-groups", type=int)
    args = parser.parse_args()
    left_path, right_path = Path(args.left_rows), Path(args.right_rows)
    output_left, output_right, summary = repartition(
        _read_jsonl(left_path),
        _read_jsonl(right_path),
        target_left_groups=args.target_left_groups,
    )
    output_left_path = Path(args.output_left_rows)
    output_right_path = Path(args.output_right_rows)
    _write_jsonl(output_left_path, output_left)
    _write_jsonl(output_right_path, output_right)
    summary.update(
        {
            "left_input": str(left_path),
            "left_input_sha256": _sha256(left_path),
            "right_input": str(right_path),
            "right_input_sha256": _sha256(right_path),
            "left_output": str(output_left_path),
            "left_output_sha256": _sha256(output_left_path),
            "right_output": str(output_right_path),
            "right_output_sha256": _sha256(output_right_path),
        }
    )
    _write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

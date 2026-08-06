#!/usr/bin/env python3
"""Audit whether IAC train/validation/evaluation splits support formal claims."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import _pathfix  # noqa: F401

from multi_horizon_protocol import get_horizon, validate_row_for_horizon  # noqa: E402


DEFAULT_REQUIRED_SOURCES = (
    "gt_pos",
    "image_swap",
    "perturb_heading",
    "perturb_lateral",
    "perturb_speed",
    "time_shift_future",
    "traj_swap",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--split must use NAME=PATH")
    name, raw = value.split("=", 1)
    if not name.strip() or not raw.strip():
        raise ValueError("--split must use non-empty NAME=PATH")
    return name.strip(), Path(raw)


def _group_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("group_id")
    if value not in (None, ""):
        return str(value)
    sample_id = row.get("sample_id")
    if sample_id not in (None, "") and "__" in str(sample_id):
        derived, _ = str(sample_id).rsplit("__", 1)
        return derived or None
    return None


def _scene_id(row: Mapping[str, Any]) -> str | None:
    for key in ("scene_name", "log_name", "log_id", "scene_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _images(row: Mapping[str, Any]) -> Iterable[str]:
    for key in ("history_images", "future_images", "images"):
        values = row.get(key, [])
        if isinstance(values, (list, tuple)):
            for value in values:
                if value not in (None, ""):
                    yield str(value)


def _split_facts(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_sources: Sequence[str],
    horizon: str | None,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    missing_group_id = 0
    derived_group_id = 0
    missing_sample_id = 0
    missing_scene_id = 0
    missing_images = 0
    horizon_invalid = 0
    horizon_examples: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        group = _group_id(row)
        if group is None:
            missing_group_id += 1
        else:
            grouped[group].append(row)
            if row.get("group_id") in (None, ""):
                derived_group_id += 1
        if row.get("sample_id") in (None, ""):
            missing_sample_id += 1
        if _scene_id(row) is None:
            missing_scene_id += 1
        if not any(_images(row)):
            missing_images += 1
        if horizon is not None:
            errors = validate_row_for_horizon(row, get_horizon(horizon))
            if errors:
                horizon_invalid += 1
                if len(horizon_examples) < 10:
                    horizon_examples.append({"row": index, "errors": errors})

    incomplete_groups = 0
    duplicate_required_sources = 0
    incomplete_examples: list[dict[str, Any]] = []
    required = set(required_sources)
    for group, values in sorted(grouped.items()):
        counts = Counter(str(row.get("source_type", "")) for row in values)
        missing = sorted(required - set(counts))
        duplicated = sorted(
            source for source in required if counts.get(source, 0) != 1
        )
        if missing or duplicated:
            incomplete_groups += 1
            duplicate_required_sources += sum(
                max(counts.get(source, 0) - 1, 0) for source in required
            )
            if len(incomplete_examples) < 10:
                incomplete_examples.append(
                    {
                        "group_id": group,
                        "missing_sources": missing,
                        "non_singleton_required_sources": duplicated,
                    }
                )

    sample_ids = {
        str(row["sample_id"])
        for row in rows
        if row.get("sample_id") not in (None, "")
    }
    scenes = {
        scene for row in rows if (scene := _scene_id(row)) is not None
    }
    images = {image for row in rows for image in _images(row)}
    schema_complete = not any(
        (missing_group_id, missing_sample_id, missing_scene_id, missing_images)
    )
    groups_complete = incomplete_groups == 0 and bool(grouped)
    horizon_complete = horizon_invalid == 0
    return {
        "rows": len(rows),
        "groups": len(grouped),
        "scenes": len(scenes),
        "images": len(images),
        "sample_ids": sample_ids,
        "group_ids": set(grouped),
        "scene_ids": scenes,
        "image_paths": images,
        "schema": {
            "missing_group_id_rows": missing_group_id,
            "derived_group_id_rows": derived_group_id,
            "missing_sample_id_rows": missing_sample_id,
            "missing_scene_id_rows": missing_scene_id,
            "missing_image_rows": missing_images,
            "complete": schema_complete,
        },
        "group_protocol": {
            "required_sources": list(required_sources),
            "incomplete_groups": incomplete_groups,
            "duplicate_required_sources": duplicate_required_sources,
            "complete": groups_complete,
            "examples": incomplete_examples,
        },
        "horizon_protocol": {
            "horizon": horizon,
            "invalid_rows": horizon_invalid,
            "complete": horizon_complete,
            "examples": horizon_examples,
        },
        "formal_split_ready": (
            schema_complete and groups_complete and horizon_complete
        ),
    }


def _public_split_summary(
    facts: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "rows": facts["rows"],
        "groups": facts["groups"],
        "scenes": facts["scenes"],
        "images": facts["images"],
        "schema": facts["schema"],
        "group_protocol": facts["group_protocol"],
        "horizon_protocol": facts["horizon_protocol"],
        "formal_split_ready": facts["formal_split_ready"],
    }


def audit_splits(
    named_paths: Sequence[tuple[str, Path]],
    *,
    required_sources: Sequence[str] = DEFAULT_REQUIRED_SOURCES,
    horizon: str | None = None,
) -> dict[str, Any]:
    if len(named_paths) < 2:
        raise ValueError("provide at least two named splits")
    names = [name for name, _ in named_paths]
    if len(names) != len(set(names)):
        raise ValueError("split names must be unique")
    if horizon is not None:
        get_horizon(horizon).validate()

    rows_by_name = {
        name: _read_jsonl(path) for name, path in named_paths
    }
    facts_by_name = {
        name: _split_facts(
            rows_by_name[name],
            required_sources=required_sources,
            horizon=horizon,
        )
        for name, _ in named_paths
    }
    split_summary = {
        name: _public_split_summary(facts_by_name[name], path)
        for name, path in named_paths
    }

    pairs: dict[str, Any] = {}
    all_disjoint = True
    for left, right in itertools.combinations(names, 2):
        left_facts = facts_by_name[left]
        right_facts = facts_by_name[right]
        overlap = {
            "sample_id_overlap": len(
                left_facts["sample_ids"] & right_facts["sample_ids"]
            ),
            "group_id_overlap": len(
                left_facts["group_ids"] & right_facts["group_ids"]
            ),
            "scene_id_overlap": len(
                left_facts["scene_ids"] & right_facts["scene_ids"]
            ),
            "image_path_overlap": len(
                left_facts["image_paths"] & right_facts["image_paths"]
            ),
        }
        overlap["strict_scene_and_image_disjoint"] = bool(
            left_facts["formal_split_ready"]
            and right_facts["formal_split_ready"]
            and all(value == 0 for value in overlap.values())
        )
        all_disjoint = all_disjoint and bool(
            overlap["strict_scene_and_image_disjoint"]
        )
        pairs[f"{left}__vs__{right}"] = overlap

    all_splits_ready = all(
        bool(summary["formal_split_ready"])
        for summary in split_summary.values()
    )
    return {
        "kind": "iac_formal_split_audit_v1",
        "inference_or_training_performed": False,
        "horizon": horizon,
        "required_sources": list(required_sources),
        "splits": split_summary,
        "pairs": pairs,
        "all_splits_schema_group_and_horizon_ready": all_splits_ready,
        "all_pairs_strict_scene_and_image_disjoint": all_disjoint,
        "formal_evidence_ready": all_splits_ready and all_disjoint,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--horizon", choices=("2s", "4s", "6s", "8s"))
    parser.add_argument(
        "--required-sources",
        default=",".join(DEFAULT_REQUIRED_SOURCES),
    )
    parser.add_argument("--require-formal-ready", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_sources = tuple(
        value.strip()
        for value in args.required_sources.split(",")
        if value.strip()
    )
    summary = audit_splits(
        [_named_path(value) for value in args.split],
        required_sources=required_sources,
        horizon=args.horizon,
    )
    _write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.require_formal_ready and not summary["formal_evidence_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

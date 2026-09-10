#!/usr/bin/env python3
"""Build a validation-only manifest with attenuated left/right pixel contrast."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _pair_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        source_key = str(row["source_key"])
        role = str(row["branch_role"])
        if role not in {"left", "right"}:
            continue
        if role in pairs.setdefault(source_key, {}):
            raise ValueError(f"duplicate {source_key}/{role}")
        pairs[source_key][role] = row
    return {
        key: pair for key, pair in pairs.items()
        if set(pair) == {"left", "right"}
    }


def _future_paths(row: dict[str, Any]) -> list[str]:
    paths = row.get("future_frame_paths") or row.get("future_images")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{row.get('source_key')}: future frame paths are missing")
    return [str(value) for value in paths]


def _blend(left: np.ndarray, right: np.ndarray, strength: float) -> tuple[np.ndarray, np.ndarray]:
    if left.shape != right.shape:
        raise ValueError(f"paired image sizes differ: {left.shape} versus {right.shape}")
    first_weight = 1.0 - strength / 2.0
    second_weight = strength / 2.0
    left_out = np.rint(first_weight * left + second_weight * right)
    right_out = np.rint(second_weight * left + first_weight * right)
    return (
        np.clip(left_out, 0, 255).astype(np.uint8),
        np.clip(right_out, 0, 255).astype(np.uint8),
    )


def build_attenuated_rows(
    rows: list[dict[str, Any]],
    *,
    reference_source_keys: set[str],
    strength: float,
    image_root: Path,
) -> list[dict[str, Any]]:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    pairs = _pair_rows(rows)
    source_keys = sorted(set(pairs) & reference_source_keys)
    if source_keys != sorted(reference_source_keys):
        missing = sorted(reference_source_keys - set(pairs))
        raise ValueError(f"input manifest misses {len(missing)} reference pairs")

    output: list[dict[str, Any]] = []
    for source_key in source_keys:
        pair = pairs[source_key]
        left_paths, right_paths = _future_paths(pair["left"]), _future_paths(pair["right"])
        if len(left_paths) != len(right_paths):
            raise ValueError(f"{source_key}: paired clips have different lengths")
        updated = {role: copy.deepcopy(pair[role]) for role in ("left", "right")}
        if strength == 0.0:
            new_paths = {"left": left_paths, "right": right_paths}
        else:
            source_dir = image_root / hashlib.sha256(source_key.encode()).hexdigest()[:16]
            source_dir.mkdir(parents=True, exist_ok=True)
            new_paths = {"left": [], "right": []}
            for index, (left_path, right_path) in enumerate(zip(left_paths, right_paths)):
                left = cv2.imread(left_path, cv2.IMREAD_COLOR)
                right = cv2.imread(right_path, cv2.IMREAD_COLOR)
                if left is None or right is None:
                    raise FileNotFoundError(f"cannot read paired frames: {left_path}, {right_path}")
                left_out, right_out = _blend(left, right, strength)
                if strength == 1.0:
                    path = source_dir / f"shared_{index:02d}.png"
                    if not cv2.imwrite(str(path), left_out):
                        raise OSError(f"failed to write {path}")
                    new_paths["left"].append(str(path.resolve()))
                    new_paths["right"].append(str(path.resolve()))
                else:
                    for role, image in (("left", left_out), ("right", right_out)):
                        path = source_dir / f"{role}_{index:02d}.png"
                        if not cv2.imwrite(str(path), image):
                            raise OSError(f"failed to write {path}")
                        new_paths[role].append(str(path.resolve()))

        for role in ("left", "right"):
            updated[role]["future_frame_paths"] = new_paths[role]
            if "future_images" in updated[role]:
                updated[role]["future_images"] = new_paths[role]
            updated[role]["future_images_source"] = "controlled_pair_contrast_attenuation"
            metadata = dict(updated[role].get("metadata") or {})
            metadata["controlled_degradation"] = {
                "validation_only": True,
                "name": "counterfactual_pixel_contrast_attenuation",
                "strength": strength,
                "candidate_dependent": True,
                "formula": "L'=(1-s/2)L+(s/2)R; R'=(s/2)L+(1-s/2)R",
            }
            updated[role]["metadata"] = metadata
            output.append(updated[role])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--reference-manifest", type=Path, action="append", required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _read_jsonl(args.manifest)
    reference = _pair_rows(_read_jsonl(args.reference_manifest))
    output = build_attenuated_rows(
        rows,
        reference_source_keys=set(reference),
        strength=args.strength,
        image_root=args.image_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(output), "pairs": len(output) // 2, "strength": args.strength}))


if __name__ == "__main__":
    main()

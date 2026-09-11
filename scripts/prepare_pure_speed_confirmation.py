#!/usr/bin/env python3
"""Prepare scene-disjoint pure-speed twin action roots for WAM generation.

This tool does not generate images and does not score a model.  It constructs a
validation-only intervention in which the same source action is scaled in XY
for a fast and a slow branch while yaw is preserved exactly.  The resulting
manifest is deliberately explicit about its pending image generation state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _trajectory(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).squeeze()
    if result.ndim != 2 or result.shape[1] < 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"action trajectory must be finite [T,>=3], got {result.shape}")
    return result[:, :3]


def _source_sample(row: dict[str, Any]) -> Path | None:
    for value in (
        row.get("source_sample"),
        (row.get("lineage") or {}).get("source_sample"),
        (row.get("lineage") or {}).get("reused_from"),
    ):
        if value and Path(str(value)).is_file():
            return Path(str(value))
    return None


def _stratum(row: dict[str, Any]) -> str:
    if row.get("stratum"):
        return str(row["stratum"])
    sample = _source_sample(row)
    if sample is not None:
        try:
            payload = pickle.loads(sample.read_bytes())
            return str((payload.get("metadata") or {}).get("stratum", "unknown"))
        except Exception:
            pass
    return "unknown"


def _read_strata(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    result: dict[str, str] = {}
    for row in _read_jsonl(path):
        source = str(row.get("source_key") or row.get("sample_id") or "")
        if source and row.get("stratum"):
            result[source] = str(row["stratum"])
    return result


def _scene_group(row: dict[str, Any]) -> str:
    if row.get("scene_group"):
        return str(row["scene_group"])
    source = str(row.get("source_key") or row.get("sample_id") or "")
    return source.rsplit(":", 1)[0]


def _choose_split(scene_group: str, seed: str, confirmation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{scene_group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "confirmation" if value < confirmation_fraction else "calibration"


def prepare(
    rows: list[dict[str, Any]],
    *,
    output_root: Path,
    fast_scale: float,
    slow_scale: float,
    confirmation_fraction: float,
    selection_seed: str,
    model_ids: list[str],
    stratum_by_source: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not 0.0 < slow_scale < fast_scale:
        raise ValueError("scales must satisfy 0 < slow < fast")
    if not 0.0 < confirmation_fraction < 1.0:
        raise ValueError("confirmation_fraction must be in (0,1)")
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        source = str(row.get("source_key") or row.get("sample_id") or "")
        if not source:
            raise ValueError("every row requires source_key or sample_id")
        if source in unique:
            raise ValueError(f"duplicate source_key: {source}")
        unique[source] = row

    split_rows: dict[str, list[dict[str, Any]]] = {"calibration": [], "confirmation": []}
    strata: Counter[str] = Counter()
    scenes: dict[str, str] = {}
    for source in sorted(unique):
        row = unique[source]
        action_value = row.get("action_trajectory") or row.get("predicted_action_trajectory")
        if action_value is None:
            raise ValueError(f"{source}: missing action_trajectory")
        base = _trajectory(action_value)
        scene = _scene_group(row)
        split = _choose_split(scene, selection_seed, confirmation_fraction)
        stratum = str((stratum_by_source or {}).get(source) or _stratum(row))
        strata[stratum] += 1
        scenes[scene] = split
        for role, scale in (("left", fast_scale), ("right", slow_scale)):
            action = base.copy()
            action[:, :2] *= scale
            split_rows[split].append(
                {
                    "protocol": "iac-pure-speed-twin-v1",
                    "source_key": source,
                    "counterfactual_group_id": source,
                    "twin_id": f"pure-speed:{source}",
                    "branch_role": role,
                    "speed_role": "fast" if role == "left" else "slow",
                    "intervention_type": "pure_speed",
                    "intervention_axis": "XY_translation_scale",
                    "fast_scale": float(fast_scale),
                    "slow_scale": float(slow_scale),
                    "yaw_identical_by_construction": True,
                    "history_fingerprint": str(row.get("history_fingerprint") or source),
                    "nuisance_seed": row.get("nuisance_seed", row.get("seed", 0)),
                    "scene_group": scene,
                    "stratum": stratum,
                    "model_ids": list(model_ids),
                    "future_images_source": "wam_generated_pending",
                    "action_trajectory_source": "validation_only_scaled_action",
                    "action_trajectory": action.tolist(),
                    "source_action_trajectory": base.tolist(),
                    "source_record": str(row.get("sample_id") or source),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    for split, values in split_rows.items():
        (output_root / split).mkdir(parents=True, exist_ok=True)
        (output_root / split / "manifest.jsonl").write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
            encoding="utf-8",
        )
    report = {
        "protocol": "iac-pure-speed-twin-preparation-v1",
        "status": "action_roots_ready_images_pending",
        "model_ids": list(model_ids),
        "source_count": len(unique),
        "twin_count": len(unique),
        "branch_count": sum(len(value) for value in split_rows.values()),
        "split_source_counts": {
            split: len(values) // 2 for split, values in split_rows.items()
        },
        "split_scene_counts": {
            split: sum(value == split for value in scenes.values())
            for split in ("calibration", "confirmation")
        },
        "stratum_counts": dict(strata),
        "fast_scale": float(fast_scale),
        "slow_scale": float(slow_scale),
        "yaw_identical_by_construction": True,
        "selection_seed": selection_seed,
        "confirmation_fraction": float(confirmation_fraction),
        "required_before_confirmation": [
            "generate both branches for every model",
            "verify same history/timestamps/calibration/nuisance",
            "run normal/reversed/identity/zero controls",
            "score only on confirmation scenes after calibration is frozen",
        ],
    }
    (output_root / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", action="append", required=True)
    parser.add_argument("--fast-scale", type=float, default=1.25)
    parser.add_argument("--slow-scale", type=float, default=0.75)
    parser.add_argument("--confirmation-fraction", type=float, default=0.4)
    parser.add_argument("--selection-seed", default="iac-pure-speed-confirm-v1")
    parser.add_argument("--stratum-manifest", type=Path)
    args = parser.parse_args()
    report = prepare(
        _read_jsonl(args.input_manifest),
        output_root=args.output_root,
        fast_scale=args.fast_scale,
        slow_scale=args.slow_scale,
        confirmation_fraction=args.confirmation_fraction,
        selection_seed=args.selection_seed,
        model_ids=args.model_id,
        stratum_by_source=_read_strata(args.stratum_manifest),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

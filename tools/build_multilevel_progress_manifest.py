#!/usr/bin/env python3
"""Build stop/slow/normal/fast action roots without claiming images exist.

The input is the real validation trajectory for each source.  Only the
longitudinal displacement (x, axis 0) is scaled; lateral path and yaw remain
unchanged.  The resulting rows are generation inputs, not evaluation rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SCALES = {"stop": 0.0, "slow": 0.5, "normal": 1.0, "fast": 1.5}


def _read(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else list(payload.get("rows") or payload.get("data") or [])


def _trajectory(row: dict[str, Any]) -> list[list[float]]:
    value = row.get("action_trajectory") or row.get("predicted_action_trajectory")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
        value = value[0]
    if not isinstance(value, list) or not value or not all(isinstance(p, list) and len(p) >= 3 for p in value):
        raise ValueError(f"{row.get('source_key')}: expected trajectory with [x,y,yaw] points")
    return [[float(v) for v in point[:3]] for point in value]


def build(rows: list[dict[str, Any]], output: Path, scales: dict[str, float] | None = None) -> dict[str, Any]:
    scales = dict(scales or DEFAULT_SCALES)
    required = ("stop", "slow", "normal", "fast")
    if set(scales) != set(required) or any(value < 0 for value in scales.values()):
        raise ValueError("scales must define non-negative stop/slow/normal/fast values")
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        source = str(row.get("source_key") or row.get("counterfactual_group_id") or "")
        if not source or source in unique:
            raise ValueError(f"missing or duplicate source_key: {source}")
        unique[source] = row

    output.parent.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    for source in sorted(unique):
        row = unique[source]
        base = _trajectory(row)
        x0 = base[0][0]
        for role in required:
            scale = float(scales[role])
            action = [
                [x0 + scale * (point[0] - x0), point[1], point[2]]
                for point in base
            ]
            generated.append({
                "protocol": "iac-pure-speed-multilevel-v1",
                "status": "images_pending",
                "source_key": source,
                "counterfactual_group_id": source,
                "twin_id": f"pure-speed-multilevel:{source}",
                "speed_role": role,
                "intervention_type": "pure_speed",
                "longitudinal_scale": scale,
                "yaw_identical_by_construction": True,
                "history_fingerprint": row.get("history_fingerprint") or source,
                "nuisance_seed": row.get("nuisance_seed", row.get("seed", 0)),
                "source_sample": row.get("source_sample"),
                "action_trajectory_source": "validation_only_scaled_longitudinal",
                "future_images_source": "wam_generated_pending",
                "action_trajectory": action,
                "source_action_trajectory": base,
                "model_id": row.get("model_id") or row.get("wam_model_id"),
            })
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated), encoding="utf-8")
    report = {
        "protocol": "iac-pure-speed-multilevel-v1",
        "status": "images_pending",
        "source_count": len(unique),
        "branch_count": len(generated),
        "roles": required,
        "scales": scales,
        "output": str(output),
    }
    output.with_name(output.stem + "_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-scale", type=float, default=0.0)
    parser.add_argument("--slow-scale", type=float, default=0.5)
    parser.add_argument("--normal-scale", type=float, default=1.0)
    parser.add_argument("--fast-scale", type=float, default=1.5)
    args = parser.parse_args()
    report = build(_read(args.input), args.output, {
        "stop": args.stop_scale, "slow": args.slow_scale,
        "normal": args.normal_scale, "fast": args.fast_scale,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

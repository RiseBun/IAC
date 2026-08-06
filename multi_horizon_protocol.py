"""Shared protocol for horizon-specific IAC evaluation.

The protocol deliberately keeps horizon as an explicit part of the result.
This prevents a short-horizon model, a long-horizon model, and an old
checkpoint evaluated with truncated/padded inputs from being compared as if
they were the same experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import math


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    seconds: float
    future_num_frames: int
    trajectory_steps: int
    step_time_s: float = 0.5

    @property
    def expected_seconds(self) -> float:
        return self.future_num_frames * self.step_time_s

    def validate(self, *, tolerance_s: float = 1e-6) -> None:
        if self.seconds <= 0:
            raise ValueError(f"{self.name}: seconds must be positive")
        if self.future_num_frames <= 0 or self.trajectory_steps <= 0:
            raise ValueError(f"{self.name}: frame/trajectory counts must be positive")
        if self.step_time_s <= 0:
            raise ValueError(f"{self.name}: step_time_s must be positive")
        if not math.isclose(self.seconds, self.expected_seconds, abs_tol=tolerance_s):
            raise ValueError(
                f"{self.name}: seconds={self.seconds} does not match "
                f"future_num_frames*step_time_s={self.expected_seconds}"
            )


HORIZONS: dict[str, HorizonSpec] = {
    "2s": HorizonSpec("2s", 2.0, future_num_frames=4, trajectory_steps=4),
    "4s": HorizonSpec("4s", 4.0, future_num_frames=8, trajectory_steps=8),
    "6s": HorizonSpec("6s", 6.0, future_num_frames=12, trajectory_steps=12),
    "8s": HorizonSpec("8s", 8.0, future_num_frames=16, trajectory_steps=16),
}


def get_horizon(name: str) -> HorizonSpec:
    key = name.strip().lower()
    if key not in HORIZONS:
        raise KeyError(f"Unknown horizon {name!r}; expected one of {sorted(HORIZONS)}")
    spec = HORIZONS[key]
    spec.validate()
    return spec


def validate_config_for_horizon(config: Mapping[str, Any], horizon: str | HorizonSpec) -> HorizonSpec:
    spec = get_horizon(horizon) if isinstance(horizon, str) else horizon
    spec.validate()
    checks = {
        "future_num_frames": spec.future_num_frames,
        "consistency_traj_steps": spec.trajectory_steps,
    }
    mismatches: list[str] = []
    for key, expected in checks.items():
        if key in config and int(config[key]) != expected:
            mismatches.append(f"{key}={config[key]} (expected {expected})")
    if "future_step_time_s" in config and not math.isclose(
        float(config["future_step_time_s"]), spec.step_time_s, abs_tol=1e-6
    ):
        mismatches.append(
            f"future_step_time_s={config['future_step_time_s']} "
            f"(expected {spec.step_time_s})"
        )
    if mismatches:
        raise ValueError(f"{spec.name} config mismatch: " + "; ".join(mismatches))
    return spec


def _as_int(row: Mapping[str, Any], key: str) -> int | None:
    value = row.get(key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def validate_row_for_horizon(row: Mapping[str, Any], horizon: str | HorizonSpec) -> list[str]:
    """Return human-readable row-level mismatches; do not raise."""
    spec = get_horizon(horizon) if isinstance(horizon, str) else horizon
    errors: list[str] = []
    future = row.get("future_images") or row.get("future_frames")
    traj = row.get("candidate_traj") or row.get("trajectory")
    if future is not None and len(future) != spec.future_num_frames:
        errors.append(f"future_frames={len(future)} expected={spec.future_num_frames}")
    if traj is not None and len(traj) != spec.trajectory_steps:
        errors.append(f"trajectory_steps={len(traj)} expected={spec.trajectory_steps}")
    declared = row.get("horizon_seconds", row.get("horizon_s"))
    if declared is not None:
        try:
            if not math.isclose(float(declared), spec.seconds, abs_tol=1e-6):
                errors.append(f"horizon_seconds={declared} expected={spec.seconds}")
        except (TypeError, ValueError):
            errors.append(f"horizon_seconds={declared!r} is not numeric")
    return errors


def summarize_jsonl(path: str | Path, horizon: str | HorizonSpec) -> dict[str, Any]:
    spec = get_horizon(horizon) if isinstance(horizon, str) else horizon
    rows = 0
    valid = 0
    invalid_examples: list[dict[str, Any]] = []
    groups: dict[str, int] = {}
    group_sizes: dict[str, int] = {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            errors = validate_row_for_horizon(row, spec)
            if errors:
                if len(invalid_examples) < 10:
                    invalid_examples.append({"line": line_no, "errors": errors})
                continue
            valid += 1
            group = str(
                row.get("group_id")
                or row.get("scene_id")
                or row.get("sample_id")
                or "__ungrouped__"
            )
            groups[group] = groups.get(group, 0) + 1
    for size in groups.values():
        key = str(size)
        group_sizes[key] = group_sizes.get(key, 0) + 1
    return {
        "protocol": "iac_multi_horizon_v1",
        "horizon": asdict(spec),
        "rows": rows,
        "valid_rows": valid,
        "invalid_rows": rows - valid,
        "groups": len(groups),
        "group_sizes": group_sizes,
        "invalid_examples": invalid_examples,
        "source": str(path),
    }


def write_protocol_manifest(out_path: str | Path) -> None:
    payload = {
        "protocol": "iac_multi_horizon_v1",
        "main_horizon": "4s",
        "horizons": {name: asdict(spec) for name, spec in HORIZONS.items()},
        "rules": {
            "same_horizon_for_frames_and_trajectory": True,
            "old_checkpoint_long_horizon_eval_is_not_formal_training": True,
            "report_each_horizon_separately": True,
            "required_metrics": [
                "strict_gt_top1",
                "acceptable_top1",
                "hard_mismatch_top1",
                "mrr",
                "auc",
                "path_causal",
            ],
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


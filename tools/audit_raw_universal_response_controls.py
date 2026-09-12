#!/usr/bin/env python3
"""Apply the preregistered universal-response gates to raw-flow reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(paths: dict[str, Path], *, coverage_min: float = 0.90, direction_ci_lower_min: float = 0.75, negative_control_fp_max: float = 0.25) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for model, path in sorted(paths.items()):
        report = json.loads(path.read_text(encoding="utf-8"))
        normal = report.get("normal", {})
        controls = report.get("controls", {})
        ci = normal.get("direction_ci95")
        reversed_fp = controls.get("reversed_action", {}).get("positive_false_positive_rate")
        identity_fp = controls.get("identity_swap", {}).get("positive_false_positive_rate")
        zero = controls.get("zero_contrast", {})
        gates = {
            "coverage": float(normal.get("coverage") or 0.0) >= coverage_min,
            "direction_ci95_lower": isinstance(ci, list) and len(ci) == 2 and float(ci[0]) >= direction_ci_lower_min,
            "reversed_negative_control": reversed_fp is not None and float(reversed_fp) <= negative_control_fp_max,
            "identity_negative_control": identity_fp is not None and float(identity_fp) <= negative_control_fp_max,
            "zero_contrast_unavailable": zero.get("directional_estimand") == "unavailable" and float(zero.get("median_observed_delta_px") or 0.0) <= 1e-9,
        }
        models.append({
            "model_id": model,
            "source_report": str(path),
            "pair_count": report.get("pair_count"),
            "normal": normal,
            "controls": controls,
            "promotion_gates": gates,
            "promotion": bool(all(gates.values())),
            "status": "pass_candidate" if all(gates.values()) else "diagnostic_incomplete_or_failed",
        })
    promoted = [row for row in models if row["promotion"]]
    return {
        "protocol": "iac-step1-universal-response-raw-control-audit-v1",
        "raw_video_rerun": True,
        "required_controls": ["normal_action", "reversed_action", "identity_swap", "zero_contrast"],
        "gates": {
            "coverage_min": coverage_min,
            "direction_ci95_lower_min": direction_ci_lower_min,
            "negative_control_false_positive_max": negative_control_fp_max,
            "zero_contrast_must_be_unavailable": True,
            "minimum_architectures": 3,
        },
        "models": models,
        "promoted_model_count": len(promoted),
        "universal_channel_promotion": len(promoted) >= 3,
        "claim_boundary": "Raw same-source twin control rerun. The new channel remains experimental unless at least three architectures pass every gate on held-out data.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", nargs=2, metavar=("MODEL", "REPORT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-control-fp-max", type=float, default=0.25)
    args = parser.parse_args()
    report = audit({model: Path(path) for model, path in args.model}, negative_control_fp_max=args.negative_control_fp_max)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"promoted_model_count": report["promoted_model_count"], "universal_channel_promotion": report["universal_channel_promotion"], "models": [{"model_id": row["model_id"], "promotion": row["promotion"], "gates": row["promotion_gates"]} for row in report["models"]]}, indent=2))


if __name__ == "__main__":
    main()

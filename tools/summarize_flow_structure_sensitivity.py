#!/usr/bin/env python3
"""Audit a preregistered flow-structure sensitivity run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pair(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        source_key = str(row["source_key"])
        role = str(row["branch_role"])
        pairs.setdefault(source_key, {})[role] = row
    invalid = [key for key, pair in pairs.items() if set(pair) != {"left", "right"}]
    if invalid:
        raise ValueError(f"{len(invalid)} source keys do not have one left/right pair")
    return pairs


def _immutable_fingerprint(row: dict[str, Any]) -> str:
    stable = copy.deepcopy(row)
    for key in ("future_frame_paths", "future_images", "future_images_source"):
        stable.pop(key, None)
    metadata = dict(stable.get("metadata") or {})
    metadata.pop("controlled_degradation", None)
    stable["metadata"] = metadata
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _full_pixel_identity(pairs: dict[str, dict[str, dict[str, Any]]]) -> bool:
    for pair in pairs.values():
        left = pair["left"].get("future_frame_paths") or pair["left"].get("future_images")
        right = pair["right"].get("future_frame_paths") or pair["right"].get("future_images")
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        for left_path, right_path in zip(left, right):
            if Path(left_path).read_bytes() != Path(right_path).read_bytes():
                return False
    return True


def summarize(
    *,
    preregistration: dict[str, Any],
    run_root: Path,
    model_directories: dict[str, str],
) -> dict[str, Any]:
    criteria = preregistration["acceptance_criteria"]
    strengths = [float(value) for value in preregistration["degradation"]["strengths"]]
    expected_count = int(preregistration["source_contract"]["expected_source_count"])
    coverage_min = float(criteria["coverage_at_every_strength_min"])
    ratio_max = float(criteria["full_attenuation_response_ratio_max"])

    model_reports: dict[str, Any] = {}
    for model, directory in model_directories.items():
        runs: list[dict[str, Any]] = []
        fingerprints: dict[tuple[str, str], str] | None = None
        full_pairs: dict[str, dict[str, dict[str, Any]]] | None = None
        for strength in strengths:
            label = str(int(round(strength * 100)))
            path = run_root / directory / f"s_{label}"
            score = json.loads((path / "score.json").read_text(encoding="utf-8"))
            pairs = _pair(_read_jsonl(path / "manifest.jsonl"))
            current_fingerprints = {
                (source_key, role): _immutable_fingerprint(row)
                for source_key, pair in pairs.items()
                for role, row in pair.items()
            }
            if fingerprints is None:
                fingerprints = current_fingerprints
            elif fingerprints != current_fingerprints:
                raise ValueError(f"{model}: non-video inputs changed at strength {strength}")
            if strength == 1.0:
                full_pairs = pairs

            responses = [
                abs(float(pair["flow_structure_delta"]))
                for pair in score.get("pairs", [])
                if pair.get("status") == "scored"
            ]
            runs.append(
                {
                    "strength": strength,
                    "source_count": len(pairs),
                    "coverage": float(score["coverage"]),
                    "median_absolute_response": statistics.median(responses),
                    "spearman": score.get("action_response_spearman"),
                    "direction_accuracy": score.get("action_direction_accuracy"),
                    "direction_pairs": int(score.get("action_direction_pairs", 0)),
                }
            )

        medians = [run["median_absolute_response"] for run in runs]
        baseline = medians[0]
        full_ratio = medians[-1] / baseline if baseline > 0.0 else float("inf")
        full_spearman = runs[-1]["spearman"]
        checks = {
            "source_count": all(run["source_count"] == expected_count for run in runs),
            "coverage": all(run["coverage"] >= coverage_min for run in runs),
            "strict_response_decrease": all(a > b for a, b in zip(medians, medians[1:])),
            "full_response_ratio": full_ratio <= ratio_max,
            "full_spearman": full_spearman is None or abs(float(full_spearman)) <= 0.1,
            "full_pixel_identity": full_pairs is not None and _full_pixel_identity(full_pairs),
            "non_video_inputs_unchanged": True,
        }
        model_reports[model] = {
            "runs": runs,
            "full_attenuation_response_ratio": full_ratio,
            "checks": checks,
            "passed": all(checks.values()),
        }

    return {
        "protocol": preregistration["protocol"],
        "preregistered_status": preregistration["status"],
        "models": model_reports,
        "overall_pass": all(report["passed"] for report in model_reports.values()),
        "interpretation_boundary": preregistration["interpretation_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=DIRECTORY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    models = dict(value.split("=", 1) for value in args.model)
    report = summarize(
        preregistration=json.loads(args.preregister.read_text(encoding="utf-8")),
        run_root=args.run_root,
        model_directories=models,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

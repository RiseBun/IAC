#!/usr/bin/env python3
"""Audit metric credibility separately from evidence completeness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _direction_row(path: Path, metric: str) -> dict[str, Any]:
    report = _load(path)
    row = report.get("confirmation", {}).get("normal", {}) if metric == "MAS" else report.get("normal", {})
    coverage = float(row.get("pair_coverage") or 0.0)
    ci = list(row.get("source_bootstrap_ci95") or row.get("direction_ci95") or [])
    lower = float(ci[0]) if ci else None
    return {
        "metric": metric,
        "model": path.stem.replace("mas_yaw_", "").replace("rcs_yaw_", ""),
        "coverage": coverage,
        "direction_accuracy": row.get("direction_accuracy"),
        "direction_ci95": ci,
        "coverage_gate": coverage >= 0.90,
        "direction_gate": lower is not None and lower >= 0.75,
        "credible_for_declared_yaw_scope": coverage >= 0.90 and lower is not None and lower >= 0.75,
    }


def run(root: Path) -> dict[str, Any]:
    rows = []
    for metric, names in (("MAS", ("mas_yaw_epona_20260912.json", "mas_yaw_drivewam_20260912.json")), ("RCS", ("rcs_yaw_epona_20260912.json", "rcs_yaw_drivewam_20260912.json"))):
        rows.append(_direction_row(root / "reports" / names[0], metric))
        rows.append(_direction_row(root / "reports" / names[1], metric))
    candidate = _load(root / "reports" / "grounding_score_candidate_20260911.json")
    gs_rows = []
    for model, value in sorted((candidate.get("models") or {}).items()):
        score = float(value.get("score_median"))
        shuffle = float(value.get("identity_shuffle_mean_score"))
        shuffle_q95 = float(value.get("identity_shuffle_q95_mean_score"))
        coverage = float(value.get("coverage") or 0.0)
        gs_rows.append({
            "metric": "GS",
            "model": model,
            "coverage": coverage,
            "score_median": score,
            "identity_shuffle_mean": shuffle,
            "identity_shuffle_q95_mean": shuffle_q95,
            "coverage_gate": coverage >= 0.90,
            "identity_specificity_gate": score > shuffle_q95,
            "credible_for_grounding_scope": coverage >= 0.90 and score > shuffle_q95,
        })
    all_rows = rows + gs_rows
    return {
        "protocol": "iac-metric-credibility-audit-v1",
        "rows": all_rows,
        "summary": {
            "mas_rcs_yaw_models_passing": sum(r["credible_for_declared_yaw_scope"] for r in rows),
            "mas_rcs_yaw_models_total": len(rows),
            "gs_models_passing_identity_specificity": sum(r["credible_for_grounding_scope"] for r in gs_rows),
            "gs_models_total": len(gs_rows),
        },
        "claim_boundary": "This is a credibility audit for declared scopes, not architecture-universal validation or future-to-action causality.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

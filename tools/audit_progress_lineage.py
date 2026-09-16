#!/usr/bin/env python3
"""Audit same-history/same-seed/model-source contracts for progress pairs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def audit(rows: list[dict[str, Any]], expected_source: str | None = None, expected_model: str | None = None) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issues: list[dict[str, Any]] = []
    for row in rows:
        group = str(row.get("counterfactual_group_id") or row.get("source_key") or "")
        groups[group].append(row)
        if expected_source and row.get("future_images_source") != expected_source:
            issues.append({"sample_id": row.get("sample_id"), "issue": "future_images_source_mismatch", "actual": row.get("future_images_source")})
        if expected_model and row.get("wam_model_id") != expected_model and row.get("model_id") != expected_model:
            issues.append({"sample_id": row.get("sample_id"), "issue": "model_id_mismatch", "actual": row.get("wam_model_id") or row.get("model_id")})
    valid_groups = 0
    for group, members in groups.items():
        roles = {str(row.get("speed_role")): row for row in members}
        if len(roles) != len(members):
            issues.append({"group": group, "issue": "duplicate_speed_role"})
        required = {"stop", "slow", "normal", "fast"}
        if required.issubset(roles):
            fingerprints = {str(row.get("history_fingerprint")) for row in members}
            seeds = {str(row.get("nuisance_seed", row.get("seed"))) for row in members}
            sources = {str(row.get("source_sample_original") or row.get("source_sample")) for row in members}
            if len(fingerprints) != 1:
                issues.append({"group": group, "issue": "history_fingerprint_mismatch"})
            if len(seeds) != 1:
                issues.append({"group": group, "issue": "nuisance_seed_mismatch"})
            if len(sources) != 1:
                issues.append({"group": group, "issue": "source_sample_mismatch"})
            if len(fingerprints) == len(seeds) == len(sources) == 1:
                valid_groups += 1
        else:
            issues.append({"group": group, "issue": "missing_four_roles", "roles": sorted(roles)})
    return {
        "protocol": "iac-progress-lineage-audit-v1",
        "declared_rows": len(rows),
        "declared_groups": len(groups),
        "valid_four_role_groups": valid_groups,
        "issue_count": len(issues),
        "status": "pass" if not issues else "fail",
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source")
    parser.add_argument("--expected-model")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = audit(rows, args.expected_source, args.expected_model)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "declared_groups", "valid_four_role_groups", "issue_count")}, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "pass" else 2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify a privacy-preserving descriptor-only GS reference release."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tools.export_gs_reference_release import DESCRIPTORS


KEY_PATTERN = re.compile(r"^gs:[0-9a-f]{32}$")
ALLOWED_FIELDS = {"source_key", "interval_index", "input_available", "stratum", *DESCRIPTORS}


def verify(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("protocol") != "iac-gs-reference-descriptor-release-v1":
        raise ValueError("unsupported_reference_release_protocol")
    if not str(manifest.get("release_id") or "").strip():
        raise ValueError("release_id_required")
    seen: set[tuple[str, int]] = set()
    for row in rows:
        unknown = set(row) - ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"private_or_unknown_fields_present:{sorted(unknown)}")
        source = str(row.get("source_key") or "")
        if not KEY_PATTERN.fullmatch(source):
            raise ValueError("source_key_is_not_hmac_pseudonym")
        try:
            interval = int(row["interval_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("interval_index_required") from error
        key = (source, interval)
        if key in seen:
            raise ValueError(f"duplicate_source_interval:{key}")
        seen.add(key)
        for descriptor in DESCRIPTORS:
            if descriptor not in row:
                continue
            value = row[descriptor]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"descriptor_not_numeric:{descriptor}")
    source_count = len({source for source, _ in seen})
    interval_count = len(rows)
    if int(manifest.get("source_count", -1)) != source_count:
        raise ValueError("source_count_mismatch")
    if int(manifest.get("interval_count", -1)) != interval_count:
        raise ValueError("interval_count_mismatch")
    return {
        "protocol": "iac-gs-reference-descriptor-release-verification-v1",
        "status": "valid",
        "release_id": manifest["release_id"],
        "source_count": source_count,
        "interval_count": interval_count,
        "privacy_boundary": "descriptor_only_hmac_pseudonymized_no_images_or_ego_state",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.reference.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = verify(rows, manifest)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

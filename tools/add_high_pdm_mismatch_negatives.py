#!/usr/bin/env python3
"""Add high-PDM image/trajectory mismatch negatives to an IAC index.

The synthetic row keeps the target group's history/future images and ego state,
but replaces the candidate trajectory with a high-PDM trajectory from another
group. This creates the hard case we actually need: a reasonable trajectory
that is not supported by the target visual future.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Row = Dict[str, Any]


QUALITY_FIELDS = (
    "official_epdms_score",
    "epdms_score",
    "official_pdm_score",
    "pdms_score",
    "planning_score",
    "candidate_quality_score",
)


def _read_jsonl(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def _group_id(row: Row) -> str:
    value = row.get("group_id") or row.get("anchor_id")
    if value is not None:
        return str(value)
    return str(row.get("sample_id", "unknown")).rsplit("__", 1)[0]


def _source_type(row: Row) -> str:
    return str(row.get("source_type") or row.get("sample_type") or "unknown")


def _quality(row: Row, fields: Sequence[str] = QUALITY_FIELDS) -> float | None:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return max(0.0, min(1.0, out))
    return None


def _traj_xy(row: Row) -> List[Tuple[float, float]]:
    traj = row.get("candidate_traj") or []
    out: List[Tuple[float, float]] = []
    for point in traj:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            out.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return out


def _mean_l2(a: Sequence[Tuple[float, float]], b: Sequence[Tuple[float, float]]) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    return sum(math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]) for i in range(n)) / n


def _endpoint_l2(a: Sequence[Tuple[float, float]], b: Sequence[Tuple[float, float]]) -> float:
    if not a or not b:
        return 0.0
    return math.hypot(a[-1][0] - b[-1][0], a[-1][1] - b[-1][1])


def _motion_signature(row: Row, lateral_bin_m: float, progress_bin_m: float) -> str:
    traj = _traj_xy(row)
    if not traj:
        return "unknown"
    end_x, end_y = traj[-1]
    if abs(end_y) < max(0.5, lateral_bin_m * 0.5):
        turn = "straight"
    elif end_y > 0:
        turn = "left"
    else:
        turn = "right"
    progress_bin = int(round(end_x / max(progress_bin_m, 1e-3)))
    lateral_bin = int(round(end_y / max(lateral_bin_m, 1e-3)))
    return f"{turn}:p{progress_bin}:l{lateral_bin}"


def _stable_int(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _best_positive(groups: Dict[str, List[Row]], positive_source: str) -> Dict[str, Row]:
    positives: Dict[str, Row] = {}
    for gid, rows in groups.items():
        candidates = [
            row
            for row in rows
            if _source_type(row) == positive_source
            or float(row.get("consistency_label", 0.0)) > 0.5
        ]
        if not candidates:
            continue
        positives[gid] = max(candidates, key=lambda row: _quality(row) or 0.0)
    return positives


def _build_donor_pools(
    rows: Sequence[Row],
    donor_sources: set[str],
    min_pdm: float,
    lateral_bin_m: float,
    progress_bin_m: float,
) -> Tuple[Dict[str, List[int]], List[int]]:
    by_signature: Dict[str, List[int]] = defaultdict(list)
    global_pool: List[int] = []
    for idx, row in enumerate(rows):
        if _source_type(row) not in donor_sources:
            continue
        score = _quality(row)
        if score is None or score < min_pdm:
            continue
        if not _traj_xy(row):
            continue
        sig = _motion_signature(row, lateral_bin_m, progress_bin_m)
        by_signature[sig].append(idx)
        global_pool.append(idx)
    return by_signature, global_pool


def _choose_donor(
    target_gid: str,
    target_row: Row,
    rows: Sequence[Row],
    by_signature: Dict[str, List[int]],
    global_pool: Sequence[int],
    rng: random.Random,
    lateral_bin_m: float,
    progress_bin_m: float,
    min_mean_l2: float,
    min_endpoint_l2: float,
    max_attempts: int,
) -> Tuple[int | None, float, float, str]:
    target_traj = _traj_xy(target_row)
    sig = _motion_signature(target_row, lateral_bin_m, progress_bin_m)
    pool = list(by_signature.get(sig) or [])
    if not pool:
        pool = list(global_pool)
    if not pool:
        return None, 0.0, 0.0, sig
    start = _stable_int(target_gid) % len(pool)
    order = pool[start:] + pool[:start]
    if len(order) > max_attempts:
        head = order[: max_attempts // 2]
        tail = rng.sample(order, k=max(1, max_attempts - len(head)))
        order = head + tail
    best_idx: int | None = None
    best_mean = -1.0
    best_endpoint = -1.0
    for donor_idx in order[:max_attempts]:
        donor = rows[donor_idx]
        donor_gid = _group_id(donor)
        if donor_gid == target_gid:
            continue
        donor_traj = _traj_xy(donor)
        mean_l2 = _mean_l2(target_traj, donor_traj)
        endpoint_l2 = _endpoint_l2(target_traj, donor_traj)
        if mean_l2 > best_mean:
            best_idx = donor_idx
            best_mean = mean_l2
            best_endpoint = endpoint_l2
        if mean_l2 >= min_mean_l2 and endpoint_l2 >= min_endpoint_l2:
            return donor_idx, mean_l2, endpoint_l2, sig
    if best_idx is not None and best_mean >= min_mean_l2:
        return best_idx, best_mean, best_endpoint, sig
    return None, max(best_mean, 0.0), max(best_endpoint, 0.0), sig


def _make_mismatch_row(
    target: Row,
    donor: Row,
    source_type: str,
    serial: int,
    mean_l2: float,
    endpoint_l2: float,
    signature: str,
) -> Row:
    target_gid = _group_id(target)
    donor_gid = _group_id(donor)
    donor_quality = _quality(donor)
    if donor_quality is None:
        donor_quality = 1.0
    out = dict(target)
    out["group_id"] = target_gid
    out["sample_id"] = f"{target_gid}__{source_type}_{serial:05d}"
    out["source_type"] = source_type
    out["label_quality"] = "high_pdm_image_mismatch_negative"
    out["consistency_label"] = 0.0
    out["validity_label"] = float(donor.get("validity_label", 1.0))
    out["candidate_traj"] = donor["candidate_traj"]
    for key in (
        "speed_consistency_label",
        "steering_consistency_label",
        "progress_consistency_label",
        "temporal_coherence_label",
    ):
        if key in out:
            out[key] = 0.0
    out["candidate_quality_score"] = donor_quality
    out["official_pdm_score"] = donor_quality
    out["pdms_score"] = donor_quality
    out["planning_score"] = donor_quality
    if donor.get("official_epdms_score") is not None:
        out["official_epdms_score"] = donor.get("official_epdms_score")
    elif donor.get("epdms_score") is not None:
        out["official_epdms_score"] = donor.get("epdms_score")
    else:
        out["official_epdms_score"] = donor_quality
    out["mismatch_policy"] = "high_pdm_same_motion_different_group"
    out["mismatch_motion_signature"] = signature
    out["mismatch_donor_group_id"] = donor_gid
    out["mismatch_donor_sample_id"] = str(donor.get("sample_id", "unknown"))
    out["mismatch_donor_source_type"] = _source_type(donor)
    out["mismatch_donor_quality_score"] = donor_quality
    out["mismatch_mean_l2_to_target_gt"] = mean_l2
    out["mismatch_endpoint_l2_to_target_gt"] = endpoint_l2
    return out


def add_mismatch_negatives(
    rows: Sequence[Row],
    *,
    per_group: int,
    max_groups: int,
    min_pdm: float,
    donor_sources: set[str],
    positive_source: str,
    source_type: str,
    lateral_bin_m: float,
    progress_bin_m: float,
    min_mean_l2: float,
    min_endpoint_l2: float,
    max_attempts: int,
    seed: int,
) -> Tuple[List[Row], Dict[str, Any]]:
    rng = random.Random(seed)
    groups: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        groups[_group_id(row)].append(row)
    positives = _best_positive(groups, positive_source)
    by_signature, global_pool = _build_donor_pools(
        rows,
        donor_sources,
        min_pdm,
        lateral_bin_m,
        progress_bin_m,
    )
    out_rows = list(rows)
    added: List[Row] = []
    skipped = Counter()
    group_ids = list(groups)
    if max_groups > 0:
        group_ids = group_ids[:max_groups]
    for gid in group_ids:
        target = positives.get(gid)
        if target is None:
            skipped["missing_positive"] += 1
            continue
        for slot in range(max(1, per_group)):
            donor_idx, mean_l2, endpoint_l2, sig = _choose_donor(
                gid,
                target,
                rows,
                by_signature,
                global_pool,
                rng,
                lateral_bin_m,
                progress_bin_m,
                min_mean_l2,
                min_endpoint_l2,
                max_attempts,
            )
            if donor_idx is None:
                skipped["no_valid_donor"] += 1
                continue
            serial = len(added) + 1
            donor = rows[donor_idx]
            mismatch = _make_mismatch_row(
                target,
                donor,
                source_type,
                serial,
                mean_l2,
                endpoint_l2,
                sig,
            )
            if per_group > 1:
                mismatch["sample_id"] = f"{gid}__{source_type}_{serial:05d}_{slot}"
            added.append(mismatch)
    out_rows.extend(added)
    added_quality = [_quality(row) or 0.0 for row in added]
    summary = {
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "input_groups": len(groups),
        "groups_considered": len(group_ids),
        "added_rows": len(added),
        "per_group": per_group,
        "min_pdm": min_pdm,
        "donor_sources": sorted(donor_sources),
        "donor_pool": len(global_pool),
        "source_type": source_type,
        "skipped": dict(skipped),
        "mean_added_quality": mean(added_quality) if added_quality else None,
        "mean_mismatch_l2": (
            mean(float(row["mismatch_mean_l2_to_target_gt"]) for row in added)
            if added
            else None
        ),
        "source_counts": dict(Counter(_source_type(row) for row in out_rows)),
    }
    return out_rows, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default=None)
    p.add_argument("--per-group", type=int, default=1)
    p.add_argument("--max-groups", type=int, default=0)
    p.add_argument("--min-pdm", type=float, default=0.85)
    p.add_argument(
        "--donor-sources",
        default="gt_pos,perturb_speed,perturb_lateral,perturb_heading",
    )
    p.add_argument("--positive-source", default="gt_pos")
    p.add_argument("--source-type", default="high_pdm_image_mismatch")
    p.add_argument("--lateral-bin-m", type=float, default=1.5)
    p.add_argument("--progress-bin-m", type=float, default=5.0)
    p.add_argument("--min-mean-l2", type=float, default=1.0)
    p.add_argument("--min-endpoint-l2", type=float, default=2.0)
    p.add_argument("--max-attempts", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    donor_sources = {
        item.strip()
        for item in str(args.donor_sources).split(",")
        if item.strip()
    }
    rows = _read_jsonl(Path(args.input))
    out_rows, summary = add_mismatch_negatives(
        rows,
        per_group=int(args.per_group),
        max_groups=int(args.max_groups),
        min_pdm=float(args.min_pdm),
        donor_sources=donor_sources,
        positive_source=str(args.positive_source),
        source_type=str(args.source_type),
        lateral_bin_m=float(args.lateral_bin_m),
        progress_bin_m=float(args.progress_bin_m),
        min_mean_l2=float(args.min_mean_l2),
        min_endpoint_l2=float(args.min_endpoint_l2),
        max_attempts=int(args.max_attempts),
        seed=int(args.seed),
    )
    _write_jsonl(Path(args.output), out_rows)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

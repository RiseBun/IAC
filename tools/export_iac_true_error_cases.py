"""Export IAC-PathBench v3.2 true-error cases for visual inspection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont


TRUE_ERROR_CATEGORIES = {
    "unsupported_gt_error",
    "clear_negative_supported_error",
    "clear_negative_rejected_but_ranked",
    "ambiguous_or_model_error",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_image(path_value: str, image_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else image_root / path


def _load_frame(paths: Any, image_root: Path, index: int = -1, size: tuple[int, int] = (256, 144)) -> Image.Image:
    if not isinstance(paths, list) or not paths:
        return Image.new("RGB", size, (32, 32, 32))
    item = paths[index]
    if not isinstance(item, str):
        return Image.new("RGB", size, (32, 32, 32))
    path = _resolve_image(item, image_root)
    if not path.exists():
        return Image.new("RGB", size, (96, 32, 32))
    return Image.open(path).convert("RGB").resize(size)


def _traj_points(traj: Any, box: tuple[int, int, int, int]) -> List[tuple[int, int]]:
    x0, y0, x1, y1 = box
    if not isinstance(traj, list) or not traj:
        return []
    pts: List[tuple[float, float]] = []
    for item in traj:
        if isinstance(item, list) and len(item) >= 2:
            try:
                pts.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                pass
    if not pts:
        return []
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    max_x = max(max(xs), 1.0)
    max_y = max(max(abs(y) for y in ys), 2.0)
    out = []
    for x, y in pts:
        px = int((x0 + x1) / 2 + (y / max_y) * (x1 - x0) * 0.42)
        py = int(y1 - min(max(x / max_x, 0.0), 1.0) * (y1 - y0) * 0.86)
        out.append((px, py))
    return out


def _draw_traj(draw: ImageDraw.ImageDraw, traj: Any, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(90, 90, 90), width=1)
    center = int((x0 + x1) / 2)
    draw.line([(center, y1), (center, y0)], fill=(70, 70, 70), width=1)
    pts = _traj_points(traj, box)
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=3)
        for p in pts[:: max(1, len(pts) // 6)]:
            draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=color)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(235, 235, 235)) -> None:
    draw.text(xy, text, fill=fill)


def _make_case_sheet(
    record: Dict[str, Any],
    gt: Dict[str, Any],
    winner: Dict[str, Any],
    image_root: Path,
    out_path: Path,
) -> None:
    w, h = 880, 560
    canvas = Image.new("RGB", (w, h), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    _text(draw, (16, 12), record["visual_indistinguishability_category"], (255, 210, 120))
    _text(draw, (16, 34), str(record["group_id"])[:120])
    _text(
        draw,
        (16, 58),
        f"winner={record.get('current_winner_source')} gap={record.get('score_gap'):.6f} "
        f"gt_minADE={record.get('gt_minade'):.3f} win_minADE={record.get('current_winner_minade'):.3f}",
    )
    gt_img = _load_frame(gt.get("future_images"), image_root)
    win_img = _load_frame(winner.get("future_images"), image_root)
    hist_img = _load_frame(gt.get("history_images"), image_root)
    canvas.paste(hist_img, (16, 94))
    canvas.paste(gt_img, (312, 94))
    canvas.paste(win_img, (608, 94))
    _text(draw, (16, 242), "history last")
    _text(draw, (312, 242), "GT future last")
    _text(draw, (608, 242), "winner future last")

    _draw_traj(draw, gt.get("candidate_traj"), (40, 292, 400, 530), (80, 220, 120))
    _draw_traj(draw, winner.get("candidate_traj"), (480, 292, 840, 530), (240, 110, 110))
    _text(draw, (40, 270), "GT trajectory", (120, 240, 150))
    _text(draw, (480, 270), "winner trajectory", (255, 140, 130))
    _text(draw, (40, 536), f"GT score={float(gt.get('iac_consistency', 0.0)):.6f} supported={record.get('gt_supported')}")
    _text(draw, (480, 536), f"W score={float(winner.get('iac_consistency', 0.0)):.6f} supported={record.get('current_winner_supported')}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _row_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "sample_id",
        "source_type",
        "iac_consistency",
        "recovered_set_minade",
        "recovered_set_supported",
        "path_minus_sky_delta",
        "candidate_minus_wrong_exclusive_path_delta",
        "candidate_minus_wrong_path_delta",
    ]
    return {key: row.get(key) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-category", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(Path(args.scores))
    records = _load_jsonl(Path(args.groups))
    by_sample = {str(row.get("sample_id")): row for row in rows if row.get("sample_id") is not None}
    categories = set(args.include_category) if args.include_category else TRUE_ERROR_CATEGORIES
    selected = [
        record for record in records
        if record.get("visual_indistinguishability_category") in categories
    ]
    if args.max_cases:
        selected = selected[: int(args.max_cases)]

    out_dir = Path(args.output_dir)
    image_root = Path(args.image_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    for idx, record in enumerate(selected, start=1):
        gt = by_sample.get(str(record.get("gt_sample_id")))
        winner = by_sample.get(str(record.get("current_winner_sample_id")))
        if gt is None or winner is None:
            continue
        category = str(record["visual_indistinguishability_category"])
        case_id = f"{idx:03d}_{category}_{record['group_id']}"
        safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in case_id)[:180]
        case_dir = out_dir / category / safe_id
        case_dir.mkdir(parents=True, exist_ok=True)
        sheet_path = case_dir / "contact_sheet.jpg"
        _make_case_sheet(record, gt, winner, image_root, sheet_path)
        payload = {
            "record": record,
            "gt": _row_summary(gt),
            "winner": _row_summary(winner),
            "contact_sheet": str(sheet_path),
        }
        (case_dir / "case.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        flat = {
            "case_index": idx,
            "category": category,
            "group_id": record.get("group_id"),
            "winner_source": record.get("current_winner_source"),
            "score_gap": record.get("score_gap"),
            "gt_minade": record.get("gt_minade"),
            "winner_minade": record.get("current_winner_minade"),
            "gt_supported": record.get("gt_supported"),
            "winner_supported": record.get("current_winner_supported"),
            "gt_score": gt.get("iac_consistency"),
            "winner_score": winner.get("iac_consistency"),
            "gt_path_minus_sky_delta": gt.get("path_minus_sky_delta"),
            "winner_path_minus_sky_delta": winner.get("path_minus_sky_delta"),
            "gt_exact_delta": gt.get("candidate_minus_wrong_exclusive_path_delta"),
            "winner_exact_delta": winner.get("candidate_minus_wrong_exclusive_path_delta"),
            "contact_sheet": str(sheet_path),
        }
        csv_rows.append(flat)
        manifest.append(payload)

    with (out_dir / "true_error_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["case_index"])
        writer.writeheader()
        writer.writerows(csv_rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    counts: Dict[str, int] = defaultdict(int)
    for row in csv_rows:
        counts[str(row["category"])] += 1
    summary = {
        "num_exported": len(csv_rows),
        "counts": dict(sorted(counts.items())),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

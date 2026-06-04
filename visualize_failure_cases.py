#!/usr/bin/env python3
"""Create visual evidence boards for IAC failure cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import load_config  # noqa: E402


BG = (245, 247, 250)
PANEL = (255, 255, 255)
INK = (24, 30, 42)
MUTED = (97, 108, 123)
RED = (202, 60, 67)
GREEN = (38, 138, 83)
BLUE = (54, 102, 204)
AMBER = (210, 139, 36)
GRID = (219, 225, 232)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize IAC failure cases")
    p.add_argument("--failure-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--prefix", default="final_best_4096_rank512")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-classification", type=int, default=6)
    p.add_argument("--num-ranking", type=int, default=4)
    return p.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(30, True)
F_H2 = font(22, True)
F_BODY = font(17)
F_SMALL = font(14)
F_TINY = font(12)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wrap_text(text: str, max_chars: int) -> List[str]:
    text = str(text)
    lines = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        lines.append(text[:cut])
        text = text[cut:].strip()
    lines.append(text)
    return lines


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill=INK, fnt=F_BODY, max_chars: int | None = None) -> int:
    x, y = xy
    lines = wrap_text(text, max_chars) if max_chars else str(text).splitlines()
    for line in lines:
        draw.text((x, y), line, fill=fill, font=fnt)
        y += int(fnt.size * 1.35)
    return y


def rounded_rect(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], fill=PANEL, outline=(225, 231, 238)) -> None:
    draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=1)


def resolve_image(image_root: Path, rel: str) -> Path:
    p = Path(str(rel))
    return p if p.is_absolute() else image_root / p


def load_thumb(image_root: Path, rel: str, size: Tuple[int, int]) -> Image.Image:
    path = resolve_image(image_root, rel)
    try:
        img = Image.open(path).convert("RGB")
        return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)
    except Exception:
        img = Image.new("RGB", size, (235, 238, 242))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "missing image", fill=RED, font=F_SMALL)
        d.text((10, 32), str(path)[-80:], fill=MUTED, font=F_TINY)
        return img


def split_paths(paths: str) -> List[str]:
    return [p for p in str(paths).split("|") if p]


def paste_labeled_image(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image_root: Path,
    rel: str,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
) -> None:
    thumb = load_thumb(image_root, rel, (w, h))
    canvas.paste(thumb, (x, y))
    draw.rectangle((x, y, x + w, y + 26), fill=(0, 0, 0))
    draw.text((x + 8, y + 5), label, fill=(255, 255, 255), font=F_SMALL)


def draw_bar_chart(
    title: str,
    data: Dict[str, int],
    path: Path,
    color: Tuple[int, int, int] = BLUE,
) -> None:
    w, h = 1100, 620
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw.text((36, 28), title, fill=INK, font=F_TITLE)
    items = list(data.items())
    max_v = max([v for _, v in items] + [1])
    left, top = 260, 96
    chart_w, bar_h, gap = 780, 42, 22
    for i, (name, value) in enumerate(items):
        y = top + i * (bar_h + gap)
        draw.text((36, y + 8), name, fill=INK, font=F_BODY)
        draw.rectangle((left, y, left + chart_w, y + bar_h), fill=(232, 237, 243))
        bw = int(chart_w * value / max_v)
        draw.rectangle((left, y, left + bw, y + bar_h), fill=color)
        draw.text((left + bw + 12, y + 8), str(value), fill=INK, font=F_BODY)
    img.save(path)


def draw_threshold_chart(summary: Dict[str, Any], path: Path) -> None:
    points = summary["threshold_sweep_consistency"]["operating_points"]
    default = summary["threshold_sweep_consistency"]["default_0.5"]
    best = summary["threshold_sweep_consistency"]["best_f1"]
    rows = [("default 0.50", default), ("best F1", best)]
    rows.extend((k, v) for k, v in points.items() if v)
    w, h = 1120, 620
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw.text((36, 28), "Threshold tradeoff: recall vs false positives", fill=INK, font=F_TITLE)
    left, top, chart_w, chart_h = 80, 120, 720, 420
    draw.rectangle((left, top, left + chart_w, top + chart_h), fill=PANEL, outline=GRID)
    for i in range(6):
        x = left + int(chart_w * i / 5)
        y = top + int(chart_h * i / 5)
        draw.line((x, top, x, top + chart_h), fill=GRID)
        draw.line((left, y, left + chart_w, y), fill=GRID)
    draw.text((left + chart_w // 2 - 70, top + chart_h + 28), "Recall", fill=MUTED, font=F_BODY)
    draw.text((left - 60, top - 34), "Precision", fill=MUTED, font=F_BODY)

    palette = [BLUE, RED, GREEN, AMBER, (121, 82, 179), (0, 142, 170)]
    for i, (name, m) in enumerate(rows):
        x = left + int(chart_w * float(m["recall"]))
        y = top + chart_h - int(chart_h * float(m["precision"]))
        color = palette[i % len(palette)]
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
        draw.text((830, 125 + i * 66), name, fill=color, font=F_H2)
        draw.text(
            (830, 152 + i * 66),
            f"thr={m['threshold']:.2f} P={m['precision']:.3f} R={m['recall']:.3f} FP={m['fp']}",
            fill=INK,
            font=F_SMALL,
        )
    img.save(path)


def select_classification_examples(rows: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    fns = [r for r in rows if r.get("error_type") == "FN"]
    fps = [r for r in rows if r.get("error_type") == "FP"]
    fns.sort(key=lambda r: float(r.get("consistency_prob", 1.0)))
    fps.sort(key=lambda r: -float(r.get("consistency_prob", 0.0)))
    selected = fns[: max(1, n // 2)]
    wanted_fp_types = ["perturb_speed", "time_shift_future", "traj_swap"]
    for st in wanted_fp_types:
        match = [r for r in fps if r.get("source_type") == st]
        if match:
            selected.append(match[0])
    selected.extend(fps[: max(0, n - len(selected))])
    dedup = []
    seen = set()
    for row in selected:
        key = row.get("sample_id")
        if key not in seen:
            seen.add(key)
            dedup.append(row)
    return dedup[:n]


def draw_classification_board(rows: List[Dict[str, str]], image_root: Path, path: Path) -> None:
    cell_w, cell_h = 1860, 330
    margin = 34
    w = cell_w + margin * 2
    h = 92 + len(rows) * cell_h + margin
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw.text((margin, 28), "Classification failure examples", fill=INK, font=F_TITLE)

    y = 86
    for row in rows:
        rounded_rect(draw, (margin, y, margin + cell_w, y + cell_h - 18))
        error = row.get("error_type")
        color = RED if error == "FN" else AMBER
        draw.text((margin + 18, y + 16), f"{error}  {row.get('source_type')}  score={row.get('consistency_prob')} label={row.get('consistency_label')}", fill=color, font=F_H2)
        draw_text(draw, (margin + 18, y + 48), f"group: {row.get('group_id')}", fill=MUTED, fnt=F_SMALL, max_chars=120)
        draw_text(draw, (margin + 18, y + 90), f"traj displacement={row.get('traj_displacement')}  start={row.get('traj_start_xy')}  end={row.get('traj_end_xy')}", fill=INK, fnt=F_SMALL, max_chars=120)

        x0 = margin + 570
        thumbs = []
        for label, key in [("history -1", "history_images_tail"), ("history 0", "history_images_tail"), ("future -1", "future_images_tail"), ("future 0", "future_images_tail")]:
            paths = split_paths(row.get(key, ""))
            idx = 0 if "-1" in label else -1
            rel = paths[idx] if paths else ""
            thumbs.append((label, rel))
        for i, (label, rel) in enumerate(thumbs):
            paste_labeled_image(img, draw, image_root, rel, x0 + i * 300, y + 54, 280, 185, label)
        y += cell_h
    img.save(path)


def traj_canvas(sample: Dict[str, Any], size: Tuple[int, int], color: Tuple[int, int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, (250, 252, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w - 1, h - 1), outline=GRID)
    traj = sample.get("candidate_traj", [])
    try:
        arr = np.asarray(traj, dtype=float)
    except Exception:
        arr = np.zeros((0, 2))
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        draw.text((10, 10), "no traj", fill=RED, font=F_SMALL)
        return img
    xy = arr[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-3)
    pad = 22
    pts = []
    for x, y in xy:
        px = pad + (x - min_xy[0]) / span[0] * (w - 2 * pad)
        py = h - pad - (y - min_xy[1]) / span[1] * (h - 2 * pad)
        pts.append((float(px), float(py)))
    for gx in range(1, 4):
        draw.line((pad, pad + gx * (h - 2 * pad) / 4, w - pad, pad + gx * (h - 2 * pad) / 4), fill=GRID)
        draw.line((pad + gx * (w - 2 * pad) / 4, pad, pad + gx * (w - 2 * pad) / 4, h - pad), fill=GRID)
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=5)
    for i, p in enumerate(pts):
        r = 5 if i in (0, len(pts) - 1) else 3
        fill = GREEN if i == 0 else RED if i == len(pts) - 1 else color
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=fill)
    draw.text((8, 8), "trajectory", fill=MUTED, font=F_TINY)
    return img


def draw_ranking_board(failure: Dict[str, Any], samples: List[Dict[str, Any]], image_root: Path, path: Path) -> None:
    candidates = failure["candidates"]
    row_h = 220
    w = 1660
    h = 170 + len(candidates) * row_h + 40
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw.text((34, 26), "Ranking Top-1 failure", fill=INK, font=F_TITLE)
    draw_text(draw, (34, 66), f"group={failure['group_id']}", fill=MUTED, fnt=F_SMALL, max_chars=150)
    draw.text((34, 102), f"model chose {failure['top_source_type']} score={failure['top_score']} ; positive rank={failure['positive_rank']} score={failure['best_positive_score']}", fill=RED, font=F_H2)

    y = 150
    for cand in candidates:
        idx = int(cand["index"])
        sample = samples[idx]
        label = int(cand["label"])
        is_top = int(cand["rank"]) == 1
        is_pos = label == 1
        border = RED if is_top and not is_pos else GREEN if is_pos else GRID
        rounded_rect(draw, (34, y, w - 34, y + row_h - 18), fill=PANEL, outline=border)
        draw.text((54, y + 16), f"rank {cand['rank']}  {cand['source_type']}  label={label}  score={cand['score']}", fill=GREEN if is_pos else RED if is_top else INK, font=F_H2)
        draw_text(draw, (54, y + 52), f"sample={cand['sample_id']}", fill=MUTED, fnt=F_TINY, max_chars=82)
        paths_h = sample.get("history_images", [])
        paths_f = sample.get("future_images", [])
        hist = paths_h[-1] if paths_h else ""
        fut = paths_f[-1] if paths_f else ""
        paste_labeled_image(img, draw, image_root, hist, 620, y + 24, 250, 160, "history last")
        paste_labeled_image(img, draw, image_root, fut, 890, y + 24, 250, 160, "future last")
        timg = traj_canvas(sample, (250, 160), GREEN if is_pos else BLUE)
        img.paste(timg, (1160, y + 24))
        draw_text(draw, (1430, y + 40), f"{cand.get('perturb_type','')}\n{cand.get('perturb_level','')}\nmag={cand.get('perturb_magnitude','')}", fill=MUTED, fnt=F_SMALL, max_chars=18)
        y += row_h
    img.save(path)


def main() -> None:
    args = parse_args()
    failure_dir = Path(args.failure_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    image_root = Path(cfg["image_root"])
    index_path = Path(cfg["val_index"] if args.split == "val" else cfg["train_index"])
    samples = load_jsonl(index_path)

    cls_rows = read_csv(failure_dir / f"{args.prefix}_classification_failures.csv")
    ranking_failures = read_json(failure_dir / f"{args.prefix}_ranking_top1_failures.json")
    summary = read_json(failure_dir / f"{args.prefix}_failure_summary.json")

    fp_by_type = summary["classification"]["fp_by_source_type"]
    ranking_by_type = summary["ranking"]["ranking_failure_top_source_type"]
    draw_bar_chart("False positives by source type", fp_by_type, out_dir / "fp_by_source_type.png", RED)
    draw_bar_chart("Ranking Top-1 failures by chosen source type", ranking_by_type, out_dir / "ranking_failures_by_type.png", AMBER)
    draw_threshold_chart(summary, out_dir / "threshold_tradeoff.png")

    examples = select_classification_examples(cls_rows, args.num_classification)
    draw_classification_board(examples, image_root, out_dir / "classification_failure_examples.png")

    for i, failure in enumerate(ranking_failures[: args.num_ranking], start=1):
        draw_ranking_board(failure, samples, image_root, out_dir / f"ranking_failure_{i:02d}.png")

    manifest = {
        "image_root": str(image_root),
        "index_path": str(index_path),
        "outputs": sorted(str(p) for p in out_dir.glob("*.png")),
    }
    with (out_dir / "visualization_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Train a visual-conditioned hard-mismatch gate for IAC ranking.

This scorer is intentionally not a replacement ranker. It predicts a
conservative non-mismatch signal:

  gt / high-quality same-scene candidates -> bounded high
  medium-quality near same-scene candidates -> neutral band
  image/time/high-PDM visual-time mismatches -> bounded low

The margin loss keeps logits useful as a calibrated gate instead of letting BCE
drive them to saturated 0/1 decisions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SAME_SCENE_SOURCES = "perturb_speed,perturb_lateral,perturb_heading"
DEFAULT_HARD_SOURCES = "image_swap,time_shift_future,high_pdm_image_mismatch"


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _safe_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _source(row: Dict[str, Any], wam_key: str) -> str:
    for key in ("source_type", "action_type", wam_key, "wam_name", "sample_type", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _is_gt_positive(row: Dict[str, Any], wam_key: str) -> bool:
    if row.get("consistency_label") is not None:
        return float(row["consistency_label"]) > 0.5
    if row.get("label") is not None:
        return float(row["label"]) > 0.5
    return _source(row, wam_key) == "gt_pos"


def _quality(row: Dict[str, Any]) -> float:
    for key in (
        "candidate_quality_score",
        "official_epdms_score",
        "epdms_score",
        "official_pdm_score",
        "pdms_score",
        "planning_score",
    ):
        if row.get(key) is not None:
            return _safe_float(row, key, math.nan)
    return math.nan


def _group_id(row: Dict[str, Any], group_key: str) -> str | None:
    value = row.get(group_key) or row.get("anchor_id") or row.get("group_id")
    if value is not None:
        return str(value)
    sample_id = row.get("sample_id")
    if sample_id is None:
        return None
    sample_id = str(sample_id)
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def _groups(rows: Sequence[Dict[str, Any]], group_key: str) -> List[List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        gid = _group_id(row, group_key)
        if gid is not None:
            grouped[gid].append(idx)
    return [items for items in grouped.values() if len(items) >= 2]


def _traj_xy(row: Dict[str, Any]) -> List[tuple[float, float]]:
    pts: List[tuple[float, float]] = []
    for item in row.get("candidate_traj") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                pts.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                pass
    return pts


def _traj_features(row: Dict[str, Any]) -> List[float]:
    pts = _traj_xy(row)
    if len(pts) < 2:
        return [0.0] * 8
    steps = [
        math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        for i in range(1, len(pts))
    ]
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    return [
        pts[-1][0],
        pts[-1][1],
        sum(steps),
        math.hypot(dx, dy),
        math.atan2(dy, dx),
        sum(y for _, y in pts) / len(pts),
        max(abs(y) for _, y in pts),
        max(steps) if steps else 0.0,
    ]


def _scalar_features(row: Dict[str, Any]) -> List[float]:
    cp = _safe_float(row, "iac_consistency", 0.5)
    recovered = _safe_float(row, "recovered_set_agreement", 0.5)
    return [
        _logit(cp),
        _logit(recovered),
        abs(_logit(cp) - _logit(recovered)),
        _safe_float(row, "recovered_set_minade"),
        _safe_float(row, "recovered_set_topmode_ade"),
        _safe_float(row, "recovered_set_best_mode_fde"),
        _safe_float(row, "recovered_set_heading_error"),
        _safe_float(row, "recovered_set_progress_error"),
        _safe_float(row, "recovered_set_path_iou"),
        _safe_float(row, "recovered_set_supported"),
        _safe_float(row, "path_minus_sky_delta"),
        _safe_float(row, "candidate_minus_wrong_path_delta"),
        _safe_float(row, "candidate_minus_wrong_exclusive_path_delta"),
        *_traj_features(row),
    ]


def _feature_cache(path: Path) -> Dict[str, torch.Tensor]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    sample_ids = [str(x) for x in cache.get("sample_id", [])]
    x = cache["x"].float()
    return {sid: x[idx] for idx, sid in enumerate(sample_ids)}


def _load_dataset(rows_path: Path, visual_cache_path: Path) -> tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
    rows = _load_rows(rows_path)
    visual_by_sample = _feature_cache(visual_cache_path)
    kept: List[Dict[str, Any]] = []
    visual: List[torch.Tensor] = []
    scalar: List[List[float]] = []
    for row in rows:
        feat = visual_by_sample.get(str(row.get("sample_id")))
        if feat is None:
            continue
        kept.append(row)
        visual.append(feat)
        scalar.append(_scalar_features(row))
    if not kept:
        raise ValueError(f"no rows matched visual cache: {rows_path}")
    return kept, torch.stack(visual, dim=0), torch.tensor(scalar, dtype=torch.float32)


def _target_kind(
    row: Dict[str, Any],
    *,
    wam_key: str,
    supported_sources: set[str],
    unknown_sources: set[str],
    hard_sources: set[str],
    min_supported_quality: float,
) -> str:
    source = _source(row, wam_key)
    if _is_gt_positive(row, wam_key):
        return "supported"
    quality = _quality(row)
    if source in hard_sources:
        return "hard"
    if source in supported_sources and math.isfinite(quality) and quality >= min_supported_quality:
        return "supported"
    if source in unknown_sources:
        return "unknown"
    return "unknown"


class MismatchGate(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        scalar_dim: int,
        visual_hidden: int,
        hidden_dim: int,
        dropout: float,
        interaction_kind: str,
    ) -> None:
        super().__init__()
        self.interaction_kind = interaction_kind
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, visual_hidden),
            nn.LayerNorm(visual_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        if interaction_kind == "concat":
            head_dim = visual_hidden + scalar_dim
            self.scalar_proj = None
        elif interaction_kind == "bilinear":
            self.scalar_proj = nn.Sequential(
                nn.Linear(scalar_dim, visual_hidden),
                nn.LayerNorm(visual_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            head_dim = visual_hidden * 4 + scalar_dim
        else:
            raise ValueError(f"unknown interaction kind: {interaction_kind}")
        self.head = nn.Sequential(
            nn.Linear(head_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, visual: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        z = self.visual_proj(visual)
        if self.interaction_kind == "concat":
            fused = torch.cat([z, scalar], dim=-1)
        else:
            assert self.scalar_proj is not None
            s = self.scalar_proj(scalar)
            fused = torch.cat([z, s, z * s, torch.abs(z - s), scalar], dim=-1)
        return self.head(fused).squeeze(-1)


def _standardize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    return mean, std


def _normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, clip: float) -> torch.Tensor:
    out = (x - mean) / std
    if clip > 0.0:
        out = out.clamp(min=-float(clip), max=float(clip))
    return out


def _train(rows: Sequence[Dict[str, Any]], visual: torch.Tensor, scalar: torch.Tensor, args: argparse.Namespace) -> Dict[str, Any]:
    supported_raw = args.supported_sources
    unknown_raw = args.unknown_sources
    if args.protect_sources:
        if supported_raw == DEFAULT_SAME_SCENE_SOURCES:
            supported_raw = args.protect_sources
        if unknown_raw == DEFAULT_SAME_SCENE_SOURCES:
            unknown_raw = args.protect_sources
    supported_sources = {item.strip() for item in supported_raw.split(",") if item.strip()}
    unknown_sources = {item.strip() for item in unknown_raw.split(",") if item.strip()}
    hard_sources = {item.strip() for item in args.hard_sources.split(",") if item.strip()}
    kinds = [
        _target_kind(
            row,
            wam_key=args.wam_key,
            supported_sources=supported_sources,
            unknown_sources=unknown_sources,
            hard_sources=hard_sources,
            min_supported_quality=float(args.min_supported_quality),
        )
        for row in rows
    ]
    train_mask = torch.tensor([kind in {"supported", "hard"} for kind in kinds], dtype=torch.bool)
    unknown_mask = torch.tensor([kind == "unknown" for kind in kinds], dtype=torch.bool)
    labels = torch.tensor([1.0 if kind == "supported" else 0.0 for kind in kinds], dtype=torch.float32)
    if int(train_mask.sum().item()) == 0:
        raise ValueError("no labeled rows for mismatch gate")

    visual_mean, visual_std = _standardize(visual)
    scalar_mean, scalar_std = _standardize(scalar)
    visual_n = _normalize(visual, visual_mean, visual_std, float(args.standardize_clip))
    scalar_n = _normalize(scalar, scalar_mean, scalar_std, float(args.standardize_clip))
    groups = _groups(rows, args.group_key)
    model = MismatchGate(
        visual.shape[1],
        scalar.shape[1],
        int(args.visual_hidden_dim),
        int(args.hidden_dim),
        float(args.dropout),
        str(args.interaction_kind),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state = None
    best_loss = math.inf
    history: List[Dict[str, Any]] = []

    for step in range(1, int(args.steps) + 1):
        logits = model(visual_n, scalar_n)
        labeled_logits = logits[train_mask]
        labeled_targets = labels[train_mask]
        pos_frac = float(labeled_targets.mean().item())
        pos_weight = torch.tensor(
            [(1.0 - pos_frac) / max(pos_frac, 1e-4)],
            dtype=torch.float32,
        )
        if args.loss_kind == "bce":
            loss = F.binary_cross_entropy_with_logits(labeled_logits, labeled_targets, pos_weight=pos_weight)
        elif args.loss_kind == "margin":
            pos_logits = logits[(train_mask) & (labels > 0.5)]
            hard_logits = logits[(train_mask) & (labels < 0.5)]
            parts: List[torch.Tensor] = []
            if bool(pos_logits.numel()):
                parts.append(F.relu(float(args.supported_margin) - pos_logits).mean())
            if bool(hard_logits.numel()):
                parts.append(F.relu(hard_logits + float(args.hard_margin)).mean())
            if not parts:
                raise ValueError("margin loss has no positive or hard rows")
            loss = torch.stack(parts).mean()
        else:
            raise ValueError(f"unknown loss kind: {args.loss_kind}")
        if bool(unknown_mask.any()) and float(args.unknown_weight) > 0.0:
            unknown_logits = logits[unknown_mask]
            unknown_loss = F.relu(unknown_logits.abs() - float(args.unknown_margin)).mean()
            loss = loss + float(args.unknown_weight) * unknown_loss
        pair_losses: List[torch.Tensor] = []
        for idxs in groups:
            local = torch.tensor(idxs, dtype=torch.long)
            local_labels = labels.index_select(0, local)
            local_mask = train_mask.index_select(0, local)
            pos = local[(local_mask) & (local_labels > 0.5)]
            hard = local[(local_mask) & (local_labels < 0.5)]
            if len(pos) and len(hard):
                pair_losses.append(
                    F.softplus(
                        float(args.pairwise_margin)
                        - logits.index_select(0, pos)[:, None]
                        + logits.index_select(0, hard)[None, :]
                    ).mean()
                )
        if pair_losses:
            loss = loss + float(args.pairwise_weight) * torch.stack(pair_losses).mean()
        if float(args.logit_l2_weight) > 0.0:
            loss = loss + float(args.logit_l2_weight) * logits.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        value = float(loss.detach().item())
        if value < best_loss:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            pred = torch.sigmoid(logits.detach())
            hard_mean = float(pred[(train_mask) & (labels < 0.5)].mean().item()) if bool(((train_mask) & (labels < 0.5)).any()) else None
            supported_mean = float(pred[(train_mask) & (labels > 0.5)].mean().item()) if bool(((train_mask) & (labels > 0.5)).any()) else None
            unknown_mean = float(pred[unknown_mask].mean().item()) if bool(unknown_mask.any()) else None
            unknown_logit_abs = float(logits.detach()[unknown_mask].abs().mean().item()) if bool(unknown_mask.any()) else None
            record = {
                "step": step,
                "loss": value,
                "best_loss": best_loss,
                "num_labeled": int(train_mask.sum().item()),
                "num_unknown": int(unknown_mask.sum().item()),
                "supported_mean": supported_mean,
                "unknown_mean": unknown_mean,
                "unknown_logit_abs": unknown_logit_abs,
                "hard_mean": hard_mean,
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    assert best_state is not None
    model.load_state_dict(best_state)
    return {
        "model": model,
        "visual_mean": visual_mean,
        "visual_std": visual_std,
        "scalar_mean": scalar_mean,
        "scalar_std": scalar_std,
        "history": history,
        "metadata": {
            "kind": "visual_mismatch_gate_scorer",
            "target_policy": "supported_gt_or_high_quality_same_scene__unknown_near__hard_visual_time_mismatch",
            "supported_sources": sorted(supported_sources),
            "unknown_sources": sorted(unknown_sources),
            "hard_sources": sorted(hard_sources),
            "target_counts": {
                "supported": sum(1 for kind in kinds if kind == "supported"),
                "hard": sum(1 for kind in kinds if kind == "hard"),
                "unknown": sum(1 for kind in kinds if kind == "unknown"),
            },
            "args": vars(args),
            "interaction_kind": str(args.interaction_kind),
            "visual_dim": int(visual.shape[1]),
            "scalar_dim": int(scalar.shape[1]),
        },
    }


@torch.no_grad()
def _score_rows(bundle: Dict[str, Any], rows: Sequence[Dict[str, Any]], visual: torch.Tensor, scalar: torch.Tensor) -> List[Dict[str, Any]]:
    model: MismatchGate = bundle["model"]
    model.eval()
    clip = float(bundle["metadata"].get("args", {}).get("standardize_clip", 5.0))
    visual_n = _normalize(visual, bundle["visual_mean"], bundle["visual_std"], clip)
    scalar_n = _normalize(scalar, bundle["scalar_mean"], bundle["scalar_std"], clip)
    logits = model(visual_n, scalar_n)
    out: List[Dict[str, Any]] = []
    for row, logit in zip(rows, logits):
        score = float(_sigmoid(float(logit.item())))
        item = dict(row)
        item["visual_non_mismatch_logit"] = float(logit.item())
        item["visual_non_mismatch"] = score
        item["iac_consistency"] = score
        item["score_fusion_label"] = "visual_mismatch_gate_non_mismatch"
        out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", required=True)
    parser.add_argument("--train-visual-cache", required=True)
    parser.add_argument("--eval", action="append", default=[], metavar="NAME=ROWS,CACHE,OUT")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--supported-sources", default=DEFAULT_SAME_SCENE_SOURCES)
    parser.add_argument("--unknown-sources", default=DEFAULT_SAME_SCENE_SOURCES)
    parser.add_argument("--protect-sources", default=None, help="Deprecated alias for supported/unknown same-scene sources.")
    parser.add_argument("--hard-sources", default=DEFAULT_HARD_SOURCES)
    parser.add_argument("--min-supported-quality", type=float, default=0.90)
    parser.add_argument("--unknown-weight", type=float, default=0.10)
    parser.add_argument("--unknown-margin", type=float, default=1.0)
    parser.add_argument("--loss-kind", choices=["margin", "bce"], default="margin")
    parser.add_argument("--supported-margin", type=float, default=1.0)
    parser.add_argument("--hard-margin", type=float, default=1.0)
    parser.add_argument("--logit-l2-weight", type=float, default=1e-3)
    parser.add_argument("--standardize-clip", type=float, default=5.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.50)
    parser.add_argument("--pairwise-margin", type=float, default=1.0)
    parser.add_argument("--interaction-kind", choices=["concat", "bilinear"], default="concat")
    parser.add_argument("--visual-hidden-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows, train_visual, train_scalar = _load_dataset(
        Path(args.train_rows),
        Path(args.train_visual_cache),
    )
    bundle = _train(train_rows, train_visual, train_scalar, args)
    torch.save(
        {
            "state_dict": bundle["model"].state_dict(),
            "visual_mean": bundle["visual_mean"],
            "visual_std": bundle["visual_std"],
            "scalar_mean": bundle["scalar_mean"],
            "scalar_std": bundle["scalar_std"],
            "metadata": bundle["metadata"],
            "history": bundle["history"],
        },
        out_dir / "visual_mismatch_gate_scorer.pt",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(bundle["metadata"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for spec in args.eval:
        name, sep, rest = spec.partition("=")
        if not sep:
            raise ValueError(f"--eval must be NAME=ROWS,CACHE,OUT, got {spec!r}")
        rows_raw, cache_raw, out_raw = rest.split(",", 2)
        rows, visual, scalar = _load_dataset(Path(rows_raw), Path(cache_raw))
        scored = _score_rows(bundle, rows, visual, scalar)
        _write_jsonl(Path(out_raw), scored)
        print(json.dumps({"eval": name, "rows": len(scored), "output": out_raw}), flush=True)


if __name__ == "__main__":
    main()

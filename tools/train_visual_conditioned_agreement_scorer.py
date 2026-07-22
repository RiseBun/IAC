"""Train a visual-conditioned agreement scorer for IAC candidate ranking.

This is a post-hoc scorer over frozen features. It learns a group/listwise
ranking from future visual evidence, candidate trajectory features, and
recovered-set geometry. Unlike coordinate-margin exclusion, it does not force
hard mismatches to be geometrically far from the recovered set; it learns when
similar geometry is still not supported by the future image.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


NEAR_SOURCES = {"perturb_speed", "perturb_lateral", "perturb_heading"}
HARD_SOURCES = {
    "image_swap",
    "time_shift",
    "time_shift_future",
    "time_shift_past",
    "high_pdm_image_mismatch",
}


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
    return [idxs for idxs in grouped.values() if len(idxs) >= 2]


def _traj_xy(row: Dict[str, Any]) -> List[tuple[float, float]]:
    pts: List[tuple[float, float]] = []
    for item in row.get("candidate_traj") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                pts.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                pass
    return pts


def _traj_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float | None:
    aa = _traj_xy(a)
    bb = _traj_xy(b)
    n = min(len(aa), len(bb))
    if n == 0:
        return None
    return sum(math.hypot(aa[i][0] - bb[i][0], aa[i][1] - bb[i][1]) for i in range(n)) / n


def _traj_features(row: Dict[str, Any]) -> List[float]:
    pts = _traj_xy(row)
    if len(pts) < 2:
        return [0.0] * 8
    steps = [
        math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        for i in range(1, len(pts))
    ]
    length = sum(steps)
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    heading = math.atan2(dy, dx)
    max_abs_y = max(abs(y) for _, y in pts)
    mean_y = sum(y for _, y in pts) / len(pts)
    return [
        pts[-1][0],
        pts[-1][1],
        length,
        math.hypot(dx, dy),
        heading,
        mean_y,
        max_abs_y,
        max(steps) if steps else 0.0,
    ]


def _scalar_features(row: Dict[str, Any]) -> List[float]:
    cp = _safe_float(row, "iac_consistency", 0.5)
    recovered = _safe_float(row, "recovered_set_agreement", 0.5)
    scalars = [
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
    ]
    return scalars + _traj_features(row)


def _feature_cache(path: Path) -> Dict[str, torch.Tensor]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    sample_ids = [str(x) for x in cache.get("sample_id", [])]
    x = cache["x"].float()
    return {sid: x[idx] for idx, sid in enumerate(sample_ids)}


def _load_dataset(
    rows_path: Path,
    visual_cache_path: Path,
) -> tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
    rows = _load_rows(rows_path)
    visual_by_sample = _feature_cache(visual_cache_path)
    visual: List[torch.Tensor] = []
    scalar: List[List[float]] = []
    kept_rows: List[Dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        feat = visual_by_sample.get(sample_id)
        if feat is None:
            continue
        kept_rows.append(row)
        visual.append(feat)
        scalar.append(_scalar_features(row))
    if not kept_rows:
        raise ValueError(f"no rows with matching visual cache: {rows_path}")
    return kept_rows, torch.stack(visual, dim=0), torch.tensor(scalar, dtype=torch.float32)


def _soft_target_weights(
    rows: Sequence[Dict[str, Any]],
    idxs: Sequence[int],
    *,
    wam_key: str,
    near_soft_weight: float,
    distance_tau: float,
) -> torch.Tensor | None:
    positives = [idx for idx in idxs if _is_gt_positive(rows[idx], wam_key)]
    if not positives:
        return None
    gt_idx = positives[0]
    weights = torch.zeros(len(idxs), dtype=torch.float32)
    for local_idx, row_idx in enumerate(idxs):
        row = rows[row_idx]
        source = _source(row, wam_key)
        if row_idx == gt_idx:
            weights[local_idx] = 1.0
        elif source in NEAR_SOURCES:
            distance = _traj_distance(row, rows[gt_idx])
            soft = 0.25 if distance is None else math.exp(-distance / max(distance_tau, 1e-6))
            weights[local_idx] = float(near_soft_weight) * max(0.15, min(1.0, soft))
    total = float(weights.sum().item())
    if total <= 0.0:
        return None
    return weights / total


class VisualAgreementScorer(nn.Module):
    def __init__(self, visual_dim: int, scalar_dim: int, visual_hidden: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, visual_hidden),
            nn.LayerNorm(visual_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(visual_hidden + scalar_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, visual: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        z = self.visual_proj(visual)
        return self.head(torch.cat([z, scalar], dim=-1)).squeeze(-1)


def _standardize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    return mean, std


def _train(
    rows: Sequence[Dict[str, Any]],
    visual: torch.Tensor,
    scalar: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    visual_mean, visual_std = _standardize(visual)
    scalar_mean, scalar_std = _standardize(scalar)
    visual_n = (visual - visual_mean) / visual_std
    scalar_n = (scalar - scalar_mean) / scalar_std
    groups = _groups(rows, args.group_key)
    model = VisualAgreementScorer(
        visual.shape[1],
        scalar.shape[1],
        int(args.visual_hidden_dim),
        int(args.hidden_dim),
        float(args.dropout),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history: List[Dict[str, Any]] = []
    hard_sources = {item.strip() for item in args.hard_sources.split(",") if item.strip()}
    best_state = None
    best_loss = math.inf

    for step in range(1, int(args.steps) + 1):
        losses: List[torch.Tensor] = []
        for idxs in groups:
            idx = torch.tensor(idxs, dtype=torch.long)
            logits = model(visual_n.index_select(0, idx), scalar_n.index_select(0, idx))
            targets = _soft_target_weights(
                rows,
                idxs,
                wam_key=args.wam_key,
                near_soft_weight=float(args.near_soft_weight),
                distance_tau=float(args.distance_tau),
            )
            if targets is None:
                continue
            losses.append(-(targets * F.log_softmax(logits, dim=0)).sum())
            pos_mask = targets > 0.0
            hard_mask = torch.tensor(
                [_source(rows[row_idx], args.wam_key) in hard_sources for row_idx in idxs],
                dtype=torch.bool,
            )
            if bool(pos_mask.any()) and bool(hard_mask.any()):
                pos = logits[pos_mask]
                hard = logits[hard_mask]
                losses.append(
                    float(args.pairwise_weight)
                    * F.softplus(float(args.pairwise_margin) - pos[:, None] + hard[None, :]).mean()
                )
        if not losses:
            raise ValueError("no trainable groups")
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        value = float(loss.detach().item())
        if value < best_loss:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            record = {"step": step, "loss": value, "best_loss": best_loss}
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
            "kind": "visual_conditioned_agreement_scorer",
            "near_sources": sorted(NEAR_SOURCES),
            "hard_sources": sorted(hard_sources),
            "args": vars(args),
            "visual_dim": int(visual.shape[1]),
            "scalar_dim": int(scalar.shape[1]),
        },
    }


@torch.no_grad()
def _score_rows(bundle: Dict[str, Any], rows: Sequence[Dict[str, Any]], visual: torch.Tensor, scalar: torch.Tensor) -> List[Dict[str, Any]]:
    model: VisualAgreementScorer = bundle["model"]
    model.eval()
    visual_n = (visual - bundle["visual_mean"]) / bundle["visual_std"]
    scalar_n = (scalar - bundle["scalar_mean"]) / bundle["scalar_std"]
    logits = model(visual_n, scalar_n)
    out: List[Dict[str, Any]] = []
    for row, logit in zip(rows, logits):
        score = float(_sigmoid(float(logit.item())))
        item = dict(row)
        item["visual_conditioned_agreement_logit"] = float(logit.item())
        item["visual_conditioned_agreement"] = score
        item["iac_consistency"] = score
        item["score_fusion_label"] = "visual_conditioned_agreement"
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
    parser.add_argument("--hard-sources", default=",".join(sorted(HARD_SOURCES)))
    parser.add_argument("--near-soft-weight", type=float, default=0.55)
    parser.add_argument("--distance-tau", type=float, default=1.5)
    parser.add_argument("--pairwise-weight", type=float, default=0.50)
    parser.add_argument("--pairwise-margin", type=float, default=1.0)
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
        out_dir / "visual_conditioned_agreement_scorer.pt",
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

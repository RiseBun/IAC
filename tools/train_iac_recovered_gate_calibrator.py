"""Train a lightweight learned gate for consistency + path + recovered-set scores.

This is a post-hoc diagnostic calibrator. It keeps the existing IAC model fixed
and learns when to trust path evidence and recovered-set agreement at candidate
ranking time.
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


def _source(row: Dict[str, Any], wam_key: str) -> str:
    for key in ("source_type", "action_type", wam_key, "wam_name", "sample_type", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _is_positive(row: Dict[str, Any], wam_key: str) -> bool:
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


def _row_key(row: Dict[str, Any], group_key: str, wam_key: str) -> tuple[str | None, str | None, str]:
    return (_group_id(row, group_key), str(row.get("sample_id")), _source(row, wam_key))


def _numeric_iac_fields(rows: Iterable[Dict[str, Any]]) -> set[str]:
    fields: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith("iac_consistency") and isinstance(value, (int, float)):
                fields.add(key)
    return fields


def _merge_triplet(
    main_path: Path,
    path_path: Path,
    recovered_path: Path,
    *,
    group_key: str,
    wam_key: str,
) -> List[Dict[str, Any]]:
    main_rows = _load_rows(main_path)
    path_rows = _load_rows(path_path)
    recovered_rows = _load_rows(recovered_path)
    if not (len(main_rows) == len(path_rows) == len(recovered_rows)):
        raise ValueError(
            f"row count mismatch: main={len(main_rows)} path={len(path_rows)} recovered={len(recovered_rows)}"
        )
    merged: List[Dict[str, Any]] = []
    for idx, (main, path, recovered) in enumerate(zip(main_rows, path_rows, recovered_rows)):
        key = _row_key(main, group_key, wam_key)
        if _row_key(path, group_key, wam_key) != key or _row_key(recovered, group_key, wam_key) != key:
            raise ValueError(f"row {idx} key mismatch: {key}")
        row = dict(main)
        row["path_head_iac_consistency"] = float(path["iac_consistency"])
        row["recovered_gate_iac_consistency"] = float(recovered["iac_consistency"])
        for k, v in path.items():
            if k.startswith("iac_consistency") and isinstance(v, (int, float)):
                row[f"path_head_{k}"] = float(v)
        for k, v in recovered.items():
            if k.startswith("iac_consistency") and isinstance(v, (int, float)):
                row[f"recovered_gate_{k}"] = float(v)
        for k, v in recovered.items():
            if k.startswith("recovered_set_"):
                row[k] = v
        merged.append(row)
    return merged


def _groups(rows: Sequence[Dict[str, Any]], group_key: str) -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gid = _group_id(row, group_key)
        if gid is not None:
            grouped[gid].append(row)
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


def _traj_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float | None:
    aa = _traj_xy(a)
    bb = _traj_xy(b)
    n = min(len(aa), len(bb))
    if n == 0:
        return None
    return sum(math.hypot(aa[i][0] - bb[i][0], aa[i][1] - bb[i][1]) for i in range(n)) / n


def _soft_targets(
    group: Sequence[Dict[str, Any]],
    *,
    wam_key: str,
    near_sources: set[str],
    near_soft_weight: float,
    min_near_soft_weight: float,
    distance_tau: float,
) -> List[float]:
    positives = [row for row in group if _is_positive(row, wam_key)]
    if not positives:
        return [0.0 for _ in group]
    gt = positives[0]
    weights: List[float] = []
    for row in group:
        if row is gt:
            weights.append(1.0)
            continue
        if _source(row, wam_key) not in near_sources:
            weights.append(0.0)
            continue
        distance = _traj_distance(row, gt)
        soft = min_near_soft_weight if distance is None else max(
            min_near_soft_weight,
            math.exp(-distance / max(distance_tau, 1e-6)),
        )
        weights.append(float(near_soft_weight) * min(1.0, soft))
    total = sum(weights)
    return [value / total for value in weights] if total > 0.0 else [0.0 for _ in group]


def _evidence_delta(row: Dict[str, Any]) -> float:
    for key in ("candidate_minus_wrong_exclusive_path_delta", "candidate_minus_wrong_path_delta"):
        if row.get(key) is not None:
            return float(row[key])
    return 0.0


def _raw_features(row: Dict[str, Any]) -> List[float]:
    main = _logit(float(row["iac_consistency"]))
    path = _logit(float(row.get("path_head_iac_consistency", row["iac_consistency"])))
    recovered = _logit(float(row.get("recovered_gate_iac_consistency", row["iac_consistency"])))
    return [
        main,
        path,
        recovered,
        abs(main - path),
        abs(main - recovered),
        float(row.get("path_minus_sky_delta") or 0.0),
        _evidence_delta(row),
        float(row.get("recovered_set_minade") or 0.0),
        float(row.get("recovered_set_topmode_ade") or 0.0),
        float(row.get("recovered_set_path_iou") or 0.0),
        float(row.get("recovered_set_supported") or 0.0),
    ]


class RecoveredGateCalibrator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, path_alpha: float, recovered_alpha: float) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.residual = nn.Linear(feature_dim, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        self.path_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(path_alpha), dtype=torch.float32))
        self.recovered_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(recovered_alpha), dtype=torch.float32))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        x_norm: torch.Tensor,
        main_logit: torch.Tensor,
        path_logit: torch.Tensor,
        recovered_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = torch.sigmoid(self.gate(x_norm))
        path_scale = F.softplus(self.path_scale_raw)
        recovered_scale = F.softplus(self.recovered_scale_raw)
        residual = 0.25 * torch.tanh(self.residual(x_norm)).squeeze(-1)
        score = (
            main_logit
            + gates[:, 0] * path_scale * path_logit
            + gates[:, 1] * recovered_scale * recovered_logit
            + residual
            + self.bias
        )
        return score, gates


def _inv_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.exp(value) - 1.0)


def _feature_stats(rows: Sequence[Dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([_raw_features(row) for row in rows], dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    return mean, std


def _prepared_groups(
    row_groups: Sequence[Sequence[Dict[str, Any]]],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    wam_key: str,
    near_sources: set[str],
    hard_sources: set[str],
    near_soft_weight: float,
    min_near_soft_weight: float,
    distance_tau: float,
) -> List[Dict[str, Any]]:
    out = []
    for group in row_groups:
        targets = _soft_targets(
            group,
            wam_key=wam_key,
            near_sources=near_sources,
            near_soft_weight=near_soft_weight,
            min_near_soft_weight=min_near_soft_weight,
            distance_tau=distance_tau,
        )
        if sum(targets) <= 0.0:
            continue
        gt_idx = next((i for i, row in enumerate(group) if _is_positive(row, wam_key)), None)
        if gt_idx is None:
            continue
        raw = torch.tensor([_raw_features(row) for row in group], dtype=torch.float32)
        sources = [_source(row, wam_key) for row in group]
        out.append(
            {
                "x_norm": (raw - mean) / std,
                "main": raw[:, 0],
                "path": raw[:, 1],
                "recovered": raw[:, 2],
                "target": torch.tensor(targets, dtype=torch.float32),
                "gt_idx": gt_idx,
                "hard_idxs": [i for i, src in enumerate(sources) if i != gt_idx and src in hard_sources],
                "near_idxs": [i for i, src in enumerate(sources) if i != gt_idx and src in near_sources],
            }
        )
    return out


def _train(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    near_sources: set[str],
    hard_sources: set[str],
) -> tuple[RecoveredGateCalibrator, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    mean, std = _feature_stats(rows)
    prepared = _prepared_groups(
        _groups(rows, args.group_key),
        mean=mean,
        std=std,
        wam_key=args.wam_key,
        near_sources=near_sources,
        hard_sources=hard_sources,
        near_soft_weight=args.near_soft_weight,
        min_near_soft_weight=args.min_near_soft_weight,
        distance_tau=args.distance_tau,
    )
    if not prepared:
        raise ValueError("no trainable groups found")
    model = RecoveredGateCalibrator(
        len(_raw_features(rows[0])),
        int(args.hidden_dim),
        float(args.init_path_alpha),
        float(args.init_recovered_alpha),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: List[Dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        losses = []
        gate_values = []
        for item in prepared:
            logits, gates = model(item["x_norm"], item["main"], item["path"], item["recovered"])
            log_probs = F.log_softmax(logits, dim=0)
            loss = -(item["target"] * log_probs).sum()
            gt = logits[int(item["gt_idx"])]
            pair_losses = []
            if item["hard_idxs"]:
                hard = logits[torch.tensor(item["hard_idxs"], dtype=torch.long)]
                pair_losses.append(F.softplus(args.hard_margin - gt + hard).mean())
            if item["near_idxs"] and args.near_margin > 0.0:
                near = logits[torch.tensor(item["near_idxs"], dtype=torch.long)]
                pair_losses.append(F.softplus(args.near_margin - gt + near).mean())
            if pair_losses:
                loss = loss + float(args.pairwise_weight) * torch.stack(pair_losses).mean()
            losses.append(loss)
            gate_values.append(gates.detach())
        total = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            gates = torch.cat(gate_values, dim=0)
            record = {
                "step": step,
                "loss": float(total.detach().item()),
                "path_gate_mean": float(gates[:, 0].mean().item()),
                "recovered_gate_mean": float(gates[:, 1].mean().item()),
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False))
    return model, mean, std, history


def _common_fields(*row_sets: Sequence[Dict[str, Any]]) -> List[str]:
    fields = None
    for rows in row_sets:
        current = _numeric_iac_fields(rows)
        fields = current if fields is None else fields & current
    return sorted(fields or {"iac_consistency"})


@torch.no_grad()
def _apply(
    rows: Sequence[Dict[str, Any]],
    model: RecoveredGateCalibrator,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    fields: Sequence[str],
) -> List[Dict[str, Any]]:
    model.eval()
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        raw = torch.tensor([_raw_features(row)], dtype=torch.float32)
        x_norm = (raw - mean) / std
        main = raw[:, 0]
        path = raw[:, 1]
        recovered = raw[:, 2]
        _, gates = model(x_norm, main, path, recovered)
        out = dict(row)
        out["iac_consistency_before_recovered_gate"] = float(row["iac_consistency"])
        out["recovered_gate_path_weight"] = float(gates[0, 0].item())
        out["recovered_gate_recovered_weight"] = float(gates[0, 1].item())
        for field in fields:
            main_value = _logit(float(row[field]))
            path_value = _logit(float(row.get(f"path_head_{field}", row.get("path_head_iac_consistency", row[field]))))
            recovered_value = _logit(float(row.get(f"recovered_gate_{field}", row.get("recovered_gate_iac_consistency", row[field]))))
            score, _ = model(
                x_norm,
                torch.tensor([main_value], dtype=torch.float32),
                torch.tensor([path_value], dtype=torch.float32),
                torch.tensor([recovered_value], dtype=torch.float32),
            )
            out[field] = _sigmoid(float(score.item()))
        out["score_fusion_label"] = "learned_recovered_gate"
        _recompute_delta_fields(out)
        out_rows.append(out)
    return out_rows


def _recompute_delta_fields(row: Dict[str, Any]) -> None:
    if "iac_consistency" not in row:
        return
    score = float(row["iac_consistency"])
    if "iac_consistency_path_masked" in row:
        row["path_mask_delta"] = score - float(row["iac_consistency_path_masked"])
    if "iac_consistency_sky_masked" in row:
        row["sky_mask_delta"] = score - float(row["iac_consistency_sky_masked"])
    if "path_mask_delta" in row and "sky_mask_delta" in row:
        row["path_minus_sky_delta"] = float(row["path_mask_delta"]) - float(row["sky_mask_delta"])
    if "iac_consistency_wrong_path_masked" in row:
        row["wrong_path_delta"] = score - float(row["iac_consistency_wrong_path_masked"])
    if "path_mask_delta" in row and "wrong_path_delta" in row:
        row["candidate_minus_wrong_path_delta"] = float(row["path_mask_delta"]) - float(row["wrong_path_delta"])
    if "iac_consistency_candidate_exclusive_path_masked" in row:
        row["candidate_exclusive_path_delta"] = score - float(row["iac_consistency_candidate_exclusive_path_masked"])
    if "iac_consistency_wrong_exclusive_path_masked" in row:
        row["wrong_exclusive_path_delta"] = score - float(row["iac_consistency_wrong_exclusive_path_masked"])
    if "candidate_exclusive_path_delta" in row and "wrong_exclusive_path_delta" in row:
        row["candidate_minus_wrong_exclusive_path_delta"] = (
            float(row["candidate_exclusive_path_delta"]) - float(row["wrong_exclusive_path_delta"])
        )


def _parse_eval(spec: str) -> tuple[str, Path, Path, Path]:
    if "=" not in spec:
        raise SystemExit(f"bad --eval spec: {spec}")
    label, raw_paths = spec.split("=", 1)
    paths = [Path(item) for item in raw_paths.split(",")]
    if len(paths) != 3:
        raise SystemExit(f"bad --eval spec: {spec}")
    return label, paths[0], paths[1], paths[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-main", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--train-recovered", required=True)
    parser.add_argument("--eval", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--near-sources", default="perturb_speed,perturb_lateral,perturb_heading")
    parser.add_argument(
        "--hard-sources",
        default="image_swap,time_shift_future,traj_swap,reverse_traj,high_pdm_image_mismatch",
    )
    parser.add_argument("--near-soft-weight", type=float, default=0.20)
    parser.add_argument("--min-near-soft-weight", type=float, default=0.02)
    parser.add_argument("--distance-tau", type=float, default=2.0)
    parser.add_argument("--hard-margin", type=float, default=1.0)
    parser.add_argument("--near-margin", type=float, default=0.05)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--init-path-alpha", type=float, default=0.2)
    parser.add_argument("--init-recovered-alpha", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    near_sources = {item.strip() for item in args.near_sources.split(",") if item.strip()}
    hard_sources = {item.strip() for item in args.hard_sources.split(",") if item.strip()}
    train_rows = _merge_triplet(
        Path(args.train_main),
        Path(args.train_path),
        Path(args.train_recovered),
        group_key=args.group_key,
        wam_key=args.wam_key,
    )
    model, mean, std, history = _train(train_rows, args, near_sources, hard_sources)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "iac_recovered_gate_calibrator",
        "train_main": args.train_main,
        "train_path": args.train_path,
        "train_recovered": args.train_recovered,
        "near_sources": sorted(near_sources),
        "hard_sources": sorted(hard_sources),
        "mean": [float(v) for v in mean.tolist()],
        "std": [float(v) for v in std.tolist()],
        "history": history,
    }
    torch.save(
        {
            "metadata": metadata,
            "state_dict": model.state_dict(),
        },
        out_dir / "recovered_gate_calibrator.pt",
    )
    (out_dir / "recovered_gate_calibrator.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for spec in args.eval:
        label, main_path, path_path, recovered_path = _parse_eval(spec)
        rows = _merge_triplet(main_path, path_path, recovered_path, group_key=args.group_key, wam_key=args.wam_key)
        fields = _common_fields(_load_rows(main_path), _load_rows(path_path), _load_rows(recovered_path))
        scored = _apply(rows, model, mean, std, fields=fields)
        _write_jsonl(out_dir / f"{label}_scores.jsonl", scored)
        gate_mean = {
            "path_gate_mean": sum(float(r["recovered_gate_path_weight"]) for r in scored) / max(len(scored), 1),
            "recovered_gate_mean": sum(float(r["recovered_gate_recovered_weight"]) for r in scored) / max(len(scored), 1),
            "num_rows": len(scored),
        }
        (out_dir / f"{label}_gate_summary.json").write_text(
            json.dumps(gate_mean, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(label, json.dumps(gate_mean, ensure_ascii=False))


if __name__ == "__main__":
    main()

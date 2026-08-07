"""对反事实轨迹做模拟评分，量化"每种反事实骗过主线的概率"。

评分模型（简化版 v3+gate 主线）：
    v3_score = f_geometry(traj)  + f_dynamics(traj)  + noise
    gate_logit = g_visual_traj(gt_xy, traj)          + noise
    fused = v3 - beta * max(0, group_max_gate - gate_logit)

关键指标（每种反事实类型）：
    - pass_rate:      被判为 acceptable 的样本比例（越高 → 主线越差）
    - top1_over_gt:   分数超过 GT 的比例（最直接的失败）
    - avg_margin:     GT - CF 的平均差距（越小 → CF 越难识别）

预期：
    - cf_semantic:      pass_rate 中等（弯道能识破，直路困难）
    - cf_physical_*:    pass_rate 高（v3 特征几乎没有速度剖面信息）
    - cf_hybrid:        pass_rate 最高（前半段完全合法）
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from counterfactual_trajectories import (
    Trajectory, path_length, turning_signed, turning_magnitude,
    compute_headings, cumulative_arc_length,
)


# ---------------------------------------------------------------------------
# 简化版 v3+gate 评分模型

def geometry_features(xy: np.ndarray) -> np.ndarray:
    """v3 用到的几何特征子集：终点、路径长度、直度、平均步长、转角、y 偏移。"""
    if xy.shape[0] < 2:
        return np.zeros(8)
    diffs = np.diff(xy, axis=0)
    steps = np.linalg.norm(diffs, axis=1)
    endpoint = xy[-1]
    path_len = float(steps.sum())
    direct = float(np.linalg.norm(endpoint))
    return np.array([
        endpoint[0] / 40.0,
        endpoint[1] / 10.0,
        abs(endpoint[1]) / 10.0,
        path_len / 40.0,
        direct / max(path_len, 1e-4),
        float(steps.mean()) / 5.0,
        float(steps.max()) / 5.0,
        turning_signed(xy) / math.pi,
    ], dtype=np.float64)


def v3_score(gt_xy: np.ndarray, cand_xy: np.ndarray,
             rng: np.random.Generator, noise_std: float = 0.03) -> float:
    """v3 打分：候选与 GT 的几何差异越小，分越高。

    这是**模拟**版本：真实 v3 用 31 维手工特征 + MLP，这里用 8 维核心几何特征代替。
    """
    f_gt = geometry_features(gt_xy)
    f_cand = geometry_features(cand_xy)
    dist = float(np.linalg.norm(f_gt - f_cand))
    # 距离 → 分数（sigmoid 变换）
    score = 1.0 / (1.0 + math.exp(4.0 * dist - 1.5))
    return float(np.clip(score + rng.normal(0, noise_std), 0.0, 1.0))


def gate_logit(gt_xy: np.ndarray, cand_xy: np.ndarray,
               rng: np.random.Generator, noise_std: float = 0.2) -> float:
    """gate 打分：视觉×轨迹交叉注意力的模拟。

    这里假设：gate 能通过视频看到 GT 的路径走向，因此 cand 与 GT 空间路径越接近，
    gate_logit 越高。**注意 gate 看不到速度剖面**，所以对 physical_* 无能为力。
    """
    # 只看空间路径（不看时间参数化）→ 从两条轨迹上按弧长重采样对齐
    from counterfactual_trajectories import resample_by_arc
    n = 16
    gt_re = resample_by_arc(gt_xy, n)
    cand_re = resample_by_arc(cand_xy, n)
    d = np.mean(np.linalg.norm(gt_re - cand_re, axis=1))
    logit = 2.0 - 2.0 * d  # d=0 → +2.0，d=1 → 0，d=2 → -2.0
    return float(logit + rng.normal(0, noise_std))


def score_row(gt_xy: np.ndarray, cand_xy: np.ndarray,
              rng: np.random.Generator) -> dict[str, float]:
    v3 = v3_score(gt_xy, cand_xy, rng)
    gate = gate_logit(gt_xy, cand_xy, rng)
    return {"v3_score": v3, "gate_logit": gate}


def fuse_group(group_scores: list[dict[str, float]],
               beta: float = 0.15) -> list[dict[str, float]]:
    """组内保守融合：fused = v3 - beta * max(0, group_max_gate - gate_logit)。"""
    group_max_gate = max(s["gate_logit"] for s in group_scores)
    out = []
    for s in group_scores:
        penalty = max(0.0, group_max_gate - s["gate_logit"])
        fused = s["v3_score"] - beta * penalty
        out.append({**s, "fused_score": float(np.clip(fused, 0.0, 1.0))})
    return out


# ---------------------------------------------------------------------------
# 主评估流程

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(rows: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    # 按 group_id 聚合
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)

    # 每组：先算所有候选的 v3+gate，再融合
    scored_by_source: dict[str, list[dict[str, float]]] = defaultdict(list)
    gt_beat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "cf_over_gt": 0})
    margin_stats: dict[str, list[float]] = defaultdict(list)

    for group_id, items in groups.items():
        gt_row = next((r for r in items if r["source_type"] == "gt_pos"), None)
        if gt_row is None:
            continue
        gt_xy = np.array(gt_row["candidate_traj"])[:, :2]

        # 每个候选打分
        temp = []
        for r in items:
            cand_xy = np.array(r["candidate_traj"])[:, :2]
            sc = score_row(gt_xy, cand_xy, rng)
            temp.append({**sc, "source_type": r["source_type"]})

        # 组内融合
        fused = fuse_group(temp)
        for s in fused:
            scored_by_source[s["source_type"]].append(s)

        # 每种 CF 比对 GT
        gt_score = next(s["fused_score"] for s in fused if s["source_type"] == "gt_pos")
        for s in fused:
            src = s["source_type"]
            if src == "gt_pos":
                continue
            gt_beat_stats[src]["total"] += 1
            if s["fused_score"] > gt_score:
                gt_beat_stats[src]["cf_over_gt"] += 1
            margin_stats[src].append(gt_score - s["fused_score"])

    # 汇总
    summary: dict[str, Any] = {}
    for src, stats in gt_beat_stats.items():
        margins = margin_stats[src]
        summary[src] = {
            "n_samples": stats["total"],
            "cf_over_gt_rate": stats["cf_over_gt"] / max(stats["total"], 1),
            "avg_margin": float(np.mean(margins)) if margins else 0.0,
            "min_margin": float(np.min(margins)) if margins else 0.0,
            "p25_margin": float(np.percentile(margins, 25)) if margins else 0.0,
            "median_margin": float(np.percentile(margins, 50)) if margins else 0.0,
        }

    # 每源的平均分
    per_source_avg = {}
    for src, lst in scored_by_source.items():
        per_source_avg[src] = {
            "avg_v3": float(np.mean([s["v3_score"] for s in lst])),
            "avg_gate_logit": float(np.mean([s["gate_logit"] for s in lst])),
            "avg_fused": float(np.mean([s["fused_score"] for s in lst])),
        }

    return {"per_source_scores": per_source_avg, "counterfactual_stats": summary}


def print_report(result: dict[str, Any]) -> None:
    print("=" * 76)
    print("各样本类型的平均分（越接近 gt_pos 的分，主线越难识别）")
    print("=" * 76)
    scores = result["per_source_scores"]
    gt = scores.get("gt_pos", {})
    print(f"\n{'source_type':<24} {'v3':>8} {'gate':>8} {'fused':>8}  vs GT_fused")
    print("-" * 76)
    for src in ["gt_pos", "cf_semantic",
                "cf_physical_accel", "cf_physical_decel",
                "cf_hybrid"]:
        if src not in scores:
            continue
        s = scores[src]
        diff = s["avg_fused"] - gt.get("avg_fused", 0.0)
        marker = "  <- GT" if src == "gt_pos" else f"  Δ={diff:+.3f}"
        print(f"{src:<24} {s['avg_v3']:>8.3f} {s['avg_gate_logit']:>8.3f} "
              f"{s['avg_fused']:>8.3f}{marker}")

    print("\n" + "=" * 76)
    print("反事实样本骗过主线的能力（cf_over_gt_rate 越高 → 主线越差）")
    print("=" * 76)
    stats = result["counterfactual_stats"]
    print(f"\n{'source_type':<24} {'n':>4} {'cf>GT率':>9} {'avg margin':>11} "
          f"{'p25 margin':>11} {'min margin':>11}")
    print("-" * 76)
    for src in ["cf_semantic",
                "cf_physical_accel", "cf_physical_decel",
                "cf_hybrid"]:
        if src not in stats:
            continue
        st = stats[src]
        rate = st["cf_over_gt_rate"] * 100
        print(f"{src:<24} {st['n_samples']:>4d} {rate:>8.1f}% "
              f"{st['avg_margin']:>+11.3f} {st['p25_margin']:>+11.3f} "
              f"{st['min_margin']:>+11.3f}")

    # 诊断
    print("\n" + "=" * 76)
    print("诊断")
    print("=" * 76)
    fail_rates = {src: stats[src]["cf_over_gt_rate"] for src in stats}
    worst = max(fail_rates, key=fail_rates.get) if fail_rates else None
    if worst:
        print(f"\n最难识别的反事实类型: {worst} ({fail_rates[worst]*100:.1f}% 骗过主线)")

    high_risk = [s for s, r in fail_rates.items() if r > 0.30]
    med_risk = [s for s, r in fail_rates.items() if 0.15 < r <= 0.30]
    if high_risk:
        print(f"❌ 严重失效: {', '.join(high_risk)}")
        print("   → 主线几乎无法区分这些反事实。急需引入到训练集。")
    if med_risk:
        print(f"⚠️  中等失效: {', '.join(med_risk)}")
        print("   → 主线部分识别，加入训练集能显著提升。")

    print("\n结论：")
    print("- 若 cf_physical_* 的 cf>GT 率 > 30% → gate 无法感知速度剖面（预期结果）")
    print("- 若 cf_hybrid 的 cf>GT 率最高      → 前后段拼接是最强反事实")
    print("- 若 cf_semantic 的 cf>GT 率 < 15%  → 主线在几何差异上表现尚可")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_jsonl", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("work/counterfactual_report.json"))
    args = ap.parse_args()

    rows = load_jsonl(args.input_jsonl)
    print(f"加载 {len(rows)} 条样本")
    result = evaluate(rows, seed=args.seed)
    print_report(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n完整报告已保存: {args.output}")


if __name__ == "__main__":
    main()

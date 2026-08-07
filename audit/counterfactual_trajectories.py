"""反事实轨迹生成器（DiffusionDrive-inspired anchored sampling）。

三种反事实类型：
    semantic  - 换动作模式的 anchor（直行→左转），几何合理但意图不对
    physical  - 保持空间路径，改速度剖面，视觉细微但物理反常
    hybrid    - GT 前半段 + 另一场景后半段，样条平滑接缝

统一采样公式（DiffusionDrive-style）：
    tau_cf = sqrt(alpha) * anchor_wrong + sqrt(1 - alpha) * noise
    alpha ∈ [0.70, 0.95]，保留 anchor 结构，加少量扰动
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 数据结构

@dataclass
class Trajectory:
    """轨迹：T 个航点，每个 (x, y, heading)。"""
    points: np.ndarray  # shape (T, 3)

    @classmethod
    def from_list(cls, raw: Sequence[Sequence[float]]) -> "Trajectory":
        arr = np.array([[p[0], p[1], p[2] if len(p) >= 3 else 0.0] for p in raw],
                       dtype=np.float64)
        return cls(points=arr)

    def to_list(self) -> list[list[float]]:
        return self.points.tolist()

    @property
    def xy(self) -> np.ndarray:
        return self.points[:, :2]

    @property
    def T(self) -> int:
        return self.points.shape[0]


# ---------------------------------------------------------------------------
# 轨迹几何工具

def cumulative_arc_length(xy: np.ndarray) -> np.ndarray:
    """累积弧长 [T]，起点为 0。"""
    diffs = np.diff(xy, axis=0)
    steps = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample_by_arc(xy: np.ndarray, num_points: int) -> np.ndarray:
    """按弧长等距重采样到 num_points 个点。"""
    if xy.shape[0] < 2:
        return xy
    arc = cumulative_arc_length(xy)
    total = arc[-1]
    if total < 1e-6:
        return np.repeat(xy[:1], num_points, axis=0)
    targets = np.linspace(0.0, total, num_points)
    x = np.interp(targets, arc, xy[:, 0])
    y = np.interp(targets, arc, xy[:, 1])
    return np.column_stack([x, y])


def compute_headings(xy: np.ndarray) -> np.ndarray:
    """从 xy 计算航向 [T]，最后一个航向复用倒数第二个的。"""
    diffs = np.diff(xy, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    if headings.size == 0:
        return np.zeros(xy.shape[0])
    return np.concatenate([headings, [headings[-1]]])


def path_length(xy: np.ndarray) -> float:
    return float(cumulative_arc_length(xy)[-1])


def turning_signed(xy: np.ndarray) -> float:
    """总航向变化 rad（有符号）：左转正、右转负、直路 ~ 0。"""
    headings = compute_headings(xy)
    if headings.size < 2:
        return 0.0
    deltas = np.diff(headings)
    deltas = np.arctan2(np.sin(deltas), np.cos(deltas))
    return float(np.sum(deltas))


def turning_magnitude(xy: np.ndarray) -> float:
    """总航向变化的绝对量 rad，直路 ~ 0，锐弯大。"""
    return float(abs(turning_signed(xy)))


# ---------------------------------------------------------------------------
# Anchor 聚类（K-Means 简化实现）

def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-Means++ 初始化：按距离概率挑选初始 center。"""
    n = X.shape[0]
    if k >= n:
        return X[rng.permutation(n)[:k]].copy()
    centers = np.zeros((k, X.shape[1]), dtype=X.dtype)
    first = rng.integers(0, n)
    centers[0] = X[first]
    for i in range(1, k):
        d2 = np.min(
            np.linalg.norm(X[:, None, :] - centers[None, :i, :], axis=2) ** 2,
            axis=1,
        )
        probs = d2 / (d2.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers[i] = X[idx]
    return centers


def kmeans_trajectories(trajs: list[np.ndarray], k: int,
                        max_iter: int = 50, seed: int = 42) -> tuple[np.ndarray, list[int]]:
    """对轨迹做 K-Means++ 聚类。

    输入：trajs 每条 xy (T, 2)，T 相同
    返回：centers (k, T, 2) 和 每条轨迹的 label
    """
    rng = np.random.default_rng(seed)
    n = len(trajs)
    if n == 0 or k <= 0:
        return np.zeros((0,)), []
    T = trajs[0].shape[0]

    X = np.stack([tj.flatten() for tj in trajs], axis=0)  # (n, T*2)
    centers = _kmeans_pp_init(X, min(k, n), rng)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = d.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(centers.shape[0]):
            mask = labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
            else:
                # 空簇：从最大簇里挑一条最远的点接管
                counts = np.bincount(labels, minlength=centers.shape[0])
                donor = int(np.argmax(counts))
                donor_pts = X[labels == donor]
                far = donor_pts[
                    np.argmax(np.linalg.norm(donor_pts - centers[donor], axis=1))
                ]
                centers[j] = far

    centers = centers.reshape(-1, T, 2)
    return centers, labels.tolist()


# ---------------------------------------------------------------------------
# 三种反事实生成器

def _sample_wrong_anchor(gt_xy: np.ndarray, anchors: np.ndarray,
                         rng: np.random.Generator,
                         semantic_gap_deg: float = 25.0) -> np.ndarray:
    """从 anchors 中选一个与 GT 语义显著不同的（有符号航向变化差别足够大）。

    用 turning_signed 是关键：左转 +45° 和右转 -45° 相差 90°，是不同语义。
    """
    gt_turn = turning_signed(gt_xy)
    gap_rad = math.radians(semantic_gap_deg)
    candidates = []
    for j in range(anchors.shape[0]):
        a_turn = turning_signed(anchors[j])
        if abs(a_turn - gt_turn) >= gap_rad:
            candidates.append(j)
    if not candidates:
        turns = np.array([turning_signed(anchors[j]) for j in range(anchors.shape[0])])
        candidates = [int(np.argmax(np.abs(turns - gt_turn)))]
    j = int(rng.choice(candidates))
    return anchors[j].copy()


def semantic_counterfactual(gt: Trajectory, anchors: np.ndarray,
                            alpha: float = 0.85,
                            rng: np.random.Generator | None = None) -> Trajectory:
    """语义反事实：换动作模式的 anchor + 少量高斯扰动。

    tau_cf = sqrt(alpha) * anchor_wrong + sqrt(1 - alpha) * noise
    """
    if rng is None:
        rng = np.random.default_rng()
    anchor = _sample_wrong_anchor(gt.xy, anchors, rng)  # (T, 2)
    noise = rng.normal(0.0, 1.0, size=anchor.shape)
    xy_cf = math.sqrt(alpha) * anchor + math.sqrt(1.0 - alpha) * noise
    headings = compute_headings(xy_cf)
    return Trajectory(points=np.column_stack([xy_cf, headings]))


def physical_counterfactual(gt: Trajectory, mode: str = "accelerate",
                            rng: np.random.Generator | None = None) -> Trajectory:
    """物理反事实：路径不变，改速度剖面。

    - accelerate: 前慢后快（t^0.5）
    - decelerate: 前快后慢（t^2）
    - reverse:    速度序列反转（先快后慢再快 → 先快后慢反过来）
    """
    if rng is None:
        rng = np.random.default_rng()
    T = gt.T
    xy = gt.xy
    arc = cumulative_arc_length(xy)
    total = arc[-1]
    if total < 1e-6:
        return Trajectory(points=gt.points.copy())

    # 原始时间参数 [0, 1]
    t = np.linspace(0.0, 1.0, T)

    if mode == "accelerate":
        # 前慢后快 → 时间参数被拉伸到 t^0.5
        t_new = t ** 0.5
    elif mode == "decelerate":
        t_new = t ** 2.0
    elif mode == "reverse":
        # 采样点距离分布反转
        dist_original = arc / total  # 每个时刻的累积路径比例
        dist_reversed = 1.0 - dist_original[::-1]
        t_new = dist_reversed
    else:
        raise ValueError(f"unknown physical mode: {mode}")

    # 按 t_new 在原路径上重新采样弧长位置
    target_arc = t_new * total
    x_new = np.interp(target_arc, arc, xy[:, 0])
    y_new = np.interp(target_arc, arc, xy[:, 1])
    xy_new = np.column_stack([x_new, y_new])
    headings = compute_headings(xy_new)
    return Trajectory(points=np.column_stack([xy_new, headings]))


def hybrid_counterfactual(gt: Trajectory, partner: Trajectory,
                          split_frac: float = 0.5,
                          smooth_window: int = 2,
                          rng: np.random.Generator | None = None) -> Trajectory:
    """混合反事实：GT 前半段 + partner 后半段，样条平滑接缝。

    split_frac 是接缝位置（0.5 表示中点）。
    """
    T = gt.T
    if partner.T != T:
        # 重采样 partner 到 T 点
        partner_xy = resample_by_arc(partner.xy, T)
    else:
        partner_xy = partner.xy.copy()

    k = max(1, min(T - 1, int(round(split_frac * T))))
    xy = np.zeros_like(gt.xy)
    xy[:k] = gt.xy[:k]
    # 后半段：把 partner 的后半段平移到接缝点
    partner_tail = partner_xy[k:] - partner_xy[k] + gt.xy[k - 1] if k < T else partner_xy[k:]
    # 上面表达式如果 k==T 会取空片段，保持结构
    if k < T:
        partner_tail = partner_xy[k:] - partner_xy[k - 1] + gt.xy[k - 1]
        xy[k:] = partner_tail

    # 平滑接缝：对接缝前后 smooth_window 个点做滑动平均
    if smooth_window > 0 and 0 < k < T:
        lo = max(0, k - smooth_window)
        hi = min(T, k + smooth_window)
        for i in range(lo, hi):
            l = max(0, i - smooth_window)
            r = min(T, i + smooth_window + 1)
            xy[i] = xy[l:r].mean(axis=0)

    headings = compute_headings(xy)
    return Trajectory(points=np.column_stack([xy, headings]))


# ---------------------------------------------------------------------------
# 生成 pipeline

def build_anchors_from_pool(gt_pool: list[np.ndarray], k: int = 8) -> np.ndarray:
    """从 GT 轨迹池聚类得到 anchor（DiffusionDrive-style anchored distribution）。"""
    if not gt_pool:
        raise ValueError("empty gt_pool")
    centers, _ = kmeans_trajectories(gt_pool, k=k)
    return centers


def generate_counterfactuals(gt: Trajectory, anchors: np.ndarray,
                             partner: Trajectory | None,
                             rng: np.random.Generator) -> dict[str, Trajectory]:
    """一次性生成三类反事实。"""
    out: dict[str, Trajectory] = {}
    out["cf_semantic"] = semantic_counterfactual(gt, anchors, alpha=0.85, rng=rng)
    out["cf_physical_accel"] = physical_counterfactual(gt, mode="accelerate", rng=rng)
    out["cf_physical_decel"] = physical_counterfactual(gt, mode="decelerate", rng=rng)
    if partner is not None:
        out["cf_hybrid"] = hybrid_counterfactual(gt, partner, split_frac=0.5, rng=rng)
    return out


# ---------------------------------------------------------------------------
# 简易模拟数据 + CLI

def _synthetic_gt_pool(num_traj: int, T: int, rng: np.random.Generator,
                       distribution: str = "discrete") -> list[np.ndarray]:
    """合成 GT 轨迹池。

    - discrete: 3 种离散模式（直/左/右 45°），场景差异极端，反事实容易被识别
    - continuous: 转角 ~ N(0, 15°)，接近真实驾驶分布，反事实难识别
    """
    pool = []
    step = 2.0  # 每步 2m
    for i in range(num_traj):
        if distribution == "discrete":
            mode = i % 3
            if mode == 0:
                turn_deg = 0.0
            elif mode == 1:
                turn_deg = 45.0
            else:
                turn_deg = -45.0
        elif distribution == "continuous":
            # 真实分布：多数直行，少数轻微转弯
            turn_deg = float(rng.normal(0, 15.0))
            turn_deg = max(-60.0, min(60.0, turn_deg))
        else:
            raise ValueError(f"unknown distribution: {distribution}")

        theta = np.linspace(0, math.radians(turn_deg), T)
        xy = np.column_stack([
            step * np.arange(T) * np.cos(theta),
            step * np.arange(T) * np.sin(theta),
        ])
        if abs(turn_deg) < 1.0:
            xy[:, 1] += rng.normal(0, 0.1, size=T)
        pool.append(xy)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-gt", type=int, default=60)
    ap.add_argument("--horizon-steps", type=int, default=8)
    ap.add_argument("--num-anchors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--distribution", choices=["discrete", "continuous"],
                    default="continuous",
                    help="合成 GT 池的分布类型：discrete 极端，continuous 真实")
    ap.add_argument("--output", type=Path, default=Path("work/counterfactuals.jsonl"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    gt_pool_xy = _synthetic_gt_pool(args.num_gt, args.horizon_steps, rng,
                                    distribution=args.distribution)
    anchors = build_anchors_from_pool(gt_pool_xy, k=args.num_anchors)

    print(f"从 {args.num_gt} 条 GT 聚类得到 {anchors.shape[0]} 个 anchor")
    for j in range(anchors.shape[0]):
        turn_deg = math.degrees(turning_signed(anchors[j]))
        length = path_length(anchors[j])
        label = "直行" if abs(turn_deg) < 10 else ("左转" if turn_deg > 0 else "右转")
        print(f"  anchor[{j}]: path={length:.1f}m, turning={turn_deg:+.1f}° ({label})")

    rows: list[dict[str, Any]] = []
    for i, gt_xy in enumerate(gt_pool_xy):
        gt = Trajectory(points=np.column_stack([gt_xy, compute_headings(gt_xy)]))
        # partner: 从池里随机挑一条不同的
        j = (i + rng.integers(1, len(gt_pool_xy))) % len(gt_pool_xy)
        partner_xy = gt_pool_xy[j]
        partner = Trajectory(points=np.column_stack([partner_xy, compute_headings(partner_xy)]))

        cfs = generate_counterfactuals(gt, anchors, partner, rng)

        group_id = f"scene_{i:04d}"
        # GT
        rows.append({
            "group_id": group_id,
            "sample_id": f"{group_id}__gt_pos",
            "source_type": "gt_pos",
            "candidate_traj": gt.to_list(),
        })
        # 反事实
        for src, tj in cfs.items():
            rows.append({
                "group_id": group_id,
                "sample_id": f"{group_id}__{src}",
                "source_type": src,
                "candidate_traj": tj.to_list(),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n生成 {len(rows)} 条样本（{args.num_gt} 场景 × 每场景 5 类）")
    print(f"保存至: {args.output}")


if __name__ == "__main__":
    main()

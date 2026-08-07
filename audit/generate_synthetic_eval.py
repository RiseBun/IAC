"""生成模拟评估数据用于对照组分析和消融实验。

由于没有真实的评估结果，我们生成模拟数据来演示分析流程。
这个数据模拟了主线在不同类型样本上的典型行为：
- 对 acceptable 样本给出高分（0.7-0.95）
- 对 simple hard 样本给出低分（0.05-0.3）—— 容易识别
- 对 plausible hard 样本给出模糊分（0.3-0.7）—— 难识别，可能被误判

这个分布反映了"快捷方式学习"的典型症状。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SOURCE_TYPES = [
    "gt_pos",              # acceptable
    "perturb_speed",       # acceptable
    "perturb_lateral",     # acceptable
    "perturb_heading",     # acceptable
    "image_swap",          # simple hard - 场景明显不一致
    "traj_swap",           # simple hard - 轨迹明显错误
    "time_shift_future",   # simple hard - 时间戳错位
    "reverse_traj",        # plausible hard - 反转轨迹（较难）
    "high_pdm_image_mismatch",  # plausible hard - PDM 认定的边缘案例
]


def generate_scores(source_type: str, rng: random.Random) -> dict[str, float]:
    """根据样本类型生成典型分数。

    模拟当前主线的行为：simple hard 容易识别，plausible hard 困难。
    """
    if source_type == "gt_pos":
        # GT 得分：主要高但有波动（模型不完美）
        v3 = rng.uniform(0.65, 0.98)
        gate = rng.uniform(0.8, 3.0)
    elif source_type in ["perturb_speed", "perturb_lateral", "perturb_heading"]:
        # 扰动样本：分数范围宽，有时甚至低于 plausible_hard
        v3 = rng.uniform(0.55, 0.90)
        gate = rng.uniform(0.2, 2.0)
    elif source_type in ["image_swap", "traj_swap", "time_shift_future"]:
        # 简单 hard - 模型很容易识别
        v3 = rng.uniform(0.05, 0.25)
        gate = rng.uniform(-3.0, -1.0)
    elif source_type == "reverse_traj":
        # ⚠️ 反转轨迹：50% 情况下模型无法识别（分数与 acceptable 重叠）
        if rng.random() < 0.50:
            v3 = rng.uniform(0.60, 0.90)  # 严重误判：分数与 GT 重叠
            gate = rng.uniform(0.3, 2.0)
        else:
            v3 = rng.uniform(0.30, 0.60)
            gate = rng.uniform(-0.5, 0.5)
    elif source_type == "high_pdm_image_mismatch":
        # ⚠️ 高 PDM 图像不匹配：60% 情况下超过某些 acceptable
        if rng.random() < 0.60:
            v3 = rng.uniform(0.65, 0.92)  # 严重误判
            gate = rng.uniform(0.5, 2.2)
        else:
            v3 = rng.uniform(0.35, 0.65)
            gate = rng.uniform(-0.2, 0.7)
    else:
        v3 = 0.5
        gate = 0.0

    # 融合分数（应用主线公式，简化版）
    fused = v3 - 0.15 * max(0, 2.0 - gate)  # 简化的融合

    return {
        "v3_score": round(v3, 4),
        "gate_logit": round(gate, 4),
        "fused_score": round(max(0, min(1, fused)), 4)
    }


def generate_group(scene_id: str, rng: random.Random) -> list[dict]:
    """为一个场景生成完整对照组（每个 source_type 一个样本）。"""
    rows = []
    for source_type in SOURCE_TYPES:
        scores = generate_scores(source_type, rng)
        row = {
            "group_id": f"scene_{scene_id}",
            "sample_id": f"scene_{scene_id}__{source_type}",
            "scene_id": f"scene_{scene_id}",
            "source_type": source_type,
            "candidate_traj": [[i * 0.5, 0.0, 0.0] for i in range(8)],  # 简化轨迹
            "history_images": [f"scene_{scene_id}/hist_{i}.jpg" for i in range(4)],
            "future_images": [f"scene_{scene_id}/fut_{i}.jpg" for i in range(8)],
            "horizon_seconds": 4.0,
            **scores
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-groups", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("synthetic_eval_scored.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    all_rows = []
    for group_idx in range(args.num_groups):
        rows = generate_group(str(group_idx), rng)
        all_rows.extend(rows)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"已生成 {len(all_rows)} 个样本 ({args.num_groups} 组 × {len(SOURCE_TYPES)} 类型)")
    print(f"保存到: {args.output}")


if __name__ == "__main__":
    main()

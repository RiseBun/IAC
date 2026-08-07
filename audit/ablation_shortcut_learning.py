"""消融实验：测试模型是否依赖简单负样本（image_swap, traj_swap, time_shift_future）。

实验设计：
1. 原始测试集：包含所有 source_type
2. 消融测试集 A：移除 simple_hard（image_swap, traj_swap, time_shift_future）
3. 消融测试集 B：只保留 acceptable 和 plausible_hard
4. 对比各测试集上的指标变化

如果指标几乎不掉 → 模型学到了快捷方式
如果大幅下降 → 模型确实理解一致性
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ACCEPTABLE_SOURCES = {"gt_pos", "perturb_speed", "perturb_lateral", "perturb_heading"}
SIMPLE_HARD_SOURCES = {"image_swap", "traj_swap", "time_shift_future"}
PLAUSIBLE_HARD_SOURCES = {"reverse_traj", "high_pdm_image_mismatch"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_verdict(rows: list[dict[str, Any]], score_key: str = "fused_score",
                     match_margin: float = 0.20, mismatch_margin: float = -0.50) -> dict[str, Any]:
    """按组进行 verdict 评估。

    模拟 score_iac_confidence.py 的判决逻辑。
    """
    # 按 group_id 分组
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)

    verdict_counts = defaultdict(int)
    top1_stats = defaultdict(int)  # 记录 top1 是哪种类型
    margin_stats = []  # 记录 margin 分布

    total_groups = 0
    hard_beats_gt_count = 0  # 记录 hard 分数超过 GT 的情况

    for group_id, group_rows in groups.items():
        # 只考虑有 acceptable 样本的组
        acceptable_rows = [r for r in group_rows if r["source_type"] in ACCEPTABLE_SOURCES]
        hard_rows = [r for r in group_rows
                     if r["source_type"] in (SIMPLE_HARD_SOURCES | PLAUSIBLE_HARD_SOURCES)]

        if not acceptable_rows or not hard_rows:
            continue

        total_groups += 1

        # 找 best_accept 和 best_bad
        best_accept = max(acceptable_rows, key=lambda r: r[score_key])
        best_bad = max(hard_rows, key=lambda r: r[score_key])

        # 记录 hard 样本超过任何 acceptable 的情况
        gt_row = next((r for r in acceptable_rows if r["source_type"] == "gt_pos"), None)
        if gt_row and best_bad[score_key] > gt_row[score_key]:
            hard_beats_gt_count += 1

        # top1 分析
        all_sorted = sorted(group_rows, key=lambda r: r[score_key], reverse=True)
        top1_source = all_sorted[0]["source_type"]
        if top1_source in ACCEPTABLE_SOURCES:
            top1_stats["acceptable_top1"] += 1
        elif top1_source in SIMPLE_HARD_SOURCES:
            top1_stats["simple_hard_top1"] += 1
        elif top1_source in PLAUSIBLE_HARD_SOURCES:
            top1_stats["plausible_hard_top1"] += 1

        # 计算 margin
        margin = best_accept[score_key] - best_bad[score_key]
        margin_stats.append(margin)

        # 判决
        if margin >= match_margin:
            verdict = "match"
        elif margin <= mismatch_margin:
            verdict = "mismatch"
        else:
            verdict = "ambiguous"

        verdict_counts[verdict] += 1

    # margin 统计
    avg_margin = sum(margin_stats) / len(margin_stats) if margin_stats else 0
    min_margin = min(margin_stats) if margin_stats else 0

    return {
        "total_groups": total_groups,
        "verdict_counts": dict(verdict_counts),
        "verdict_percentage": {
            k: f"{v/total_groups*100:.1f}%" for k, v in verdict_counts.items()
        } if total_groups > 0 else {},
        "top1_stats": dict(top1_stats),
        "top1_percentage": {
            k: f"{v/total_groups*100:.1f}%" for k, v in top1_stats.items()
        } if total_groups > 0 else {},
        "acceptable_top1_rate": top1_stats["acceptable_top1"] / total_groups if total_groups > 0 else 0,
        "hard_mismatch_top1_rate": (
            (top1_stats["simple_hard_top1"] + top1_stats["plausible_hard_top1"]) / total_groups
            if total_groups > 0 else 0
        ),
        "avg_margin": avg_margin,
        "min_margin": min_margin,
        "hard_beats_gt_rate": hard_beats_gt_count / total_groups if total_groups > 0 else 0,
        "ambiguous_rate": verdict_counts.get("ambiguous", 0) / total_groups if total_groups > 0 else 0
    }


def filter_dataset(rows: list[dict], remove_sources: set[str]) -> list[dict]:
    """从数据集中移除指定 source_type 的样本。"""
    return [r for r in rows if r["source_type"] not in remove_sources]


def run_ablation(rows: list[dict]) -> dict[str, Any]:
    """运行完整消融实验。"""

    print("=" * 70)
    print("消融实验：测试模型是否依赖简单负样本")
    print("=" * 70)

    def print_results(name: str, r: dict) -> None:
        print(f"  总组数: {r['total_groups']}")
        print(f"  acceptable_top1: {r['acceptable_top1_rate']*100:.1f}%")
        print(f"  hard_mismatch_top1: {r['hard_mismatch_top1_rate']*100:.1f}%")
        print(f"  Hard 超过 GT 比例: {r['hard_beats_gt_rate']*100:.1f}%")
        print(f"  平均 margin: {r['avg_margin']:.3f}")
        print(f"  最小 margin: {r['min_margin']:.3f}")
        print(f"  Ambiguous 比例: {r['ambiguous_rate']*100:.1f}%")
        print(f"  Verdict 分布: {r['verdict_percentage']}")

    # 实验 1：只保留 simple_hard（模拟"漂亮"评估）
    print("\n[实验 1] 只保留 acceptable + simple_hard（漂亮但不真实）")
    rows_only_simple = filter_dataset(rows, PLAUSIBLE_HARD_SOURCES)
    results_only_simple = evaluate_verdict(rows_only_simple)
    print_results("only_simple", results_only_simple)

    # 实验 2：原始测试集
    print("\n[实验 2] 原始测试集（所有 source_type）")
    results_original = evaluate_verdict(rows)
    print_results("original", results_original)

    # 实验 3：只保留 plausible_hard（真实评估）
    print("\n[实验 3] 只保留 acceptable + plausible_hard（真正的挑战）")
    rows_only_plausible = filter_dataset(rows, SIMPLE_HARD_SOURCES)
    results_only_plausible = evaluate_verdict(rows_only_plausible)
    print_results("only_plausible", results_only_plausible)

    # 对比分析
    print("\n" + "=" * 70)
    print("对比分析：simple_hard vs plausible_hard 挑战难度")
    print("=" * 70)

    simple_top1 = results_only_simple['acceptable_top1_rate']
    plausible_top1 = results_only_plausible['acceptable_top1_rate']
    simple_hard_beats = results_only_simple['hard_beats_gt_rate']
    plausible_hard_beats = results_only_plausible['hard_beats_gt_rate']

    top1_drop = (simple_top1 - plausible_top1) * 100
    hard_beats_gap = (plausible_hard_beats - simple_hard_beats) * 100

    print(f"\n只有 simple_hard 时:")
    print(f"  acceptable_top1: {simple_top1*100:.1f}%")
    print(f"  hard 超过 GT 比例: {simple_hard_beats*100:.1f}%")

    print(f"\n只有 plausible_hard 时:")
    print(f"  acceptable_top1: {plausible_top1*100:.1f}%")
    print(f"  hard 超过 GT 比例: {plausible_hard_beats*100:.1f}%")

    print(f"\nacceptable_top1 差距: {top1_drop:+.1f}% (simple → plausible)")
    print(f"Hard 超越 GT 差距: {hard_beats_gap:+.1f}% (plausible - simple)")

    # 诊断：核心看 simple 和 plausible 之间的性能差距
    if top1_drop > 20 or plausible_hard_beats > 0.30:
        diagnosis = (
            "❌ 严重快捷方式学习！\n"
            f"   - 只考虑 simple_hard 时: {simple_top1*100:.1f}% top1\n"
            f"   - 只考虑 plausible_hard 时: {plausible_top1*100:.1f}% top1\n"
            f"   - 差距 {top1_drop:.1f}% 说明模型主要靠识别简单负样本刷分\n"
            f"   - {plausible_hard_beats*100:.1f}% 的 plausible_hard 甚至超过 GT\n"
            "   建议：\n"
            "   1. 用 diffusion model 生成更多 plausible hard negatives\n"
            "   2. 减少 image_swap/traj_swap 训练比重\n"
            "   3. 引入反事实视频生成（world model 条件化）"
        )
        risk = "critical"
    elif top1_drop > 10 or plausible_hard_beats > 0.15:
        diagnosis = (
            "⚠️  中等快捷方式学习：\n"
            f"   simple → plausible 时 top1 下降 {top1_drop:.1f}%\n"
            f"   {plausible_hard_beats*100:.1f}% 的 plausible_hard 超过 GT\n"
            "   建议：加入更多 plausible hard negatives"
        )
        risk = "medium"
    else:
        diagnosis = (
            "✅ 未检测到明显的快捷方式学习。\n"
            "   模型在 plausible_hard 上保持一致性能。"
        )
        risk = "low"

    print(f"\n{diagnosis}")

    margin_change = 0  # 这里 margin_change 已经没意义
    ambiguous_change = 0
    hard_beats_gt_change = hard_beats_gap

    return {
        "experiments": {
            "only_simple_hard": results_only_simple,
            "original": results_original,
            "only_plausible_hard": results_only_plausible
        },
        "top1_drop_simple_to_plausible": top1_drop,
        "hard_beats_gap": hard_beats_gap,
        "diagnosis": diagnosis,
        "risk_level": risk
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="输入 JSONL 文件（包含评分）")
    parser.add_argument("--output", type=Path, default=Path("ablation_report.json"))
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)

    results = run_ablation(rows)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n完整报告已保存: {args.output}")


if __name__ == "__main__":
    main()

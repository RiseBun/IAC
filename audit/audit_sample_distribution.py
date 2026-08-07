"""审计对照组样本分布和错误模式分析。

分析：
1. 每种 source_type 的样本数量分布
2. v3 和 gate 在各类样本上的错误率
3. 识别是否存在"只能正确分类简单负样本"的快捷方式学习
4. 生成对照组质量报告
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# Source type 分类
ACCEPTABLE_SOURCES = {"gt_pos", "perturb_speed", "perturb_lateral", "perturb_heading"}
SIMPLE_HARD_SOURCES = {"image_swap", "traj_swap", "time_shift_future"}
PLAUSIBLE_HARD_SOURCES = {"reverse_traj", "high_pdm_image_mismatch"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_source(source: str) -> str:
    """将 source_type 分类为：acceptable / simple_hard / plausible_hard / unknown."""
    if source in ACCEPTABLE_SOURCES:
        return "acceptable"
    elif source in SIMPLE_HARD_SOURCES:
        return "simple_hard"
    elif source in PLAUSIBLE_HARD_SOURCES:
        return "plausible_hard"
    else:
        return "unknown"


def analyze_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """分析样本分布。"""
    source_counts = defaultdict(int)
    source_class_counts = defaultdict(int)

    for row in rows:
        source = row.get("source_type", "unknown")
        source_counts[source] += 1
        source_class = classify_source(source)
        source_class_counts[source_class] += 1

    total = len(rows)
    return {
        "total_samples": total,
        "by_source": dict(source_counts),
        "by_class": dict(source_class_counts),
        "class_percentage": {
            cls: f"{count/total*100:.1f}%"
            for cls, count in source_class_counts.items()
        }
    }


def analyze_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """分析 v3 和 gate 在不同样本类型上的错误模式。

    假设输入已经包含 v3_score, gate_logit, fused_score 等字段。
    """
    # 按 source_class 统计错误
    errors_by_class = {
        "acceptable": {"v3_errors": [], "gate_errors": [], "fused_errors": []},
        "simple_hard": {"v3_errors": [], "gate_errors": [], "fused_errors": []},
        "plausible_hard": {"v3_errors": [], "gate_errors": [], "fused_errors": []},
    }

    counts_by_class = defaultdict(int)

    for row in rows:
        source = row.get("source_type", "unknown")
        source_class = classify_source(source)

        if source_class == "unknown":
            continue

        counts_by_class[source_class] += 1

        # 根据 source_class 判断期望标签
        is_acceptable = source_class == "acceptable"

        # 检查各模型是否犯错（需要有评分字段）
        v3_score = row.get("v3_score")
        gate_logit = row.get("gate_logit")
        fused_score = row.get("fused_score")

        if v3_score is not None:
            # v3: acceptable 应该 >0.5, hard 应该 <0.5
            v3_pred = v3_score > 0.5
            if v3_pred != is_acceptable:
                errors_by_class[source_class]["v3_errors"].append({
                    "sample_id": row.get("sample_id", "unknown"),
                    "source": source,
                    "v3_score": v3_score,
                    "expected": "acceptable" if is_acceptable else "mismatch"
                })

        if gate_logit is not None:
            gate_pred = gate_logit > 0
            if gate_pred != is_acceptable:
                errors_by_class[source_class]["gate_errors"].append({
                    "sample_id": row.get("sample_id", "unknown"),
                    "source": source,
                    "gate_logit": gate_logit,
                    "expected": "acceptable" if is_acceptable else "mismatch"
                })

        if fused_score is not None:
            fused_pred = fused_score > 0.5
            if fused_pred != is_acceptable:
                errors_by_class[source_class]["fused_errors"].append({
                    "sample_id": row.get("sample_id", "unknown"),
                    "source": source,
                    "fused_score": fused_score,
                    "expected": "acceptable" if is_acceptable else "mismatch"
                })

    # 计算错误率
    error_rates = {}
    for source_class in ["acceptable", "simple_hard", "plausible_hard"]:
        total = counts_by_class[source_class]
        if total == 0:
            continue
        error_rates[source_class] = {
            "v3_error_rate": len(errors_by_class[source_class]["v3_errors"]) / total,
            "gate_error_rate": len(errors_by_class[source_class]["gate_errors"]) / total,
            "fused_error_rate": len(errors_by_class[source_class]["fused_errors"]) / total,
            "total_samples": total
        }

    return {
        "error_rates": error_rates,
        "error_examples": errors_by_class
    }


def diagnose_shortcut_learning(error_analysis: dict[str, Any]) -> dict[str, Any]:
    """诊断是否存在快捷方式学习。

    症状：
    - simple_hard 错误率很低（模型轻松识别）
    - plausible_hard 错误率很高（模型无法识别）
    → 说明模型学到的是简单特征（场景不匹配、时间错位）而非真正的一致性
    """
    error_rates = error_analysis.get("error_rates", {})

    simple_hard_error = error_rates.get("simple_hard", {}).get("v3_error_rate", 0)
    plausible_hard_error = error_rates.get("plausible_hard", {}).get("v3_error_rate", 0)

    # 如果 simple_hard 错误率 < 5% 且 plausible_hard 错误率 > 20%
    # 说明可能存在快捷方式学习
    shortcut_risk = "low"
    if simple_hard_error < 0.05 and plausible_hard_error > 0.20:
        shortcut_risk = "high"
    elif simple_hard_error < 0.10 and plausible_hard_error > 0.15:
        shortcut_risk = "medium"

    return {
        "shortcut_learning_risk": shortcut_risk,
        "simple_hard_error_rate": simple_hard_error,
        "plausible_hard_error_rate": plausible_hard_error,
        "diagnosis": (
            "模型可能学到了快捷方式（简单的场景/时间不匹配检测），"
            "而非真正的动作-视觉一致性理解。"
            if shortcut_risk == "high"
            else "未检测到明显的快捷方式学习。"
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="输入 JSONL 文件（包含评分结果）")
    parser.add_argument("--output", type=Path, default=Path("sample_distribution_report.json"))
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)

    print(f"加载 {len(rows)} 个样本")

    # 1. 样本分布分析
    distribution = analyze_distribution(rows)
    print("\n=== 样本分布 ===")
    print(f"总样本数: {distribution['total_samples']}")
    print("\n按 source_type:")
    for source, count in sorted(distribution['by_source'].items()):
        print(f"  {source}: {count}")
    print("\n按分类:")
    for cls, count in distribution['by_class'].items():
        pct = distribution['class_percentage'][cls]
        print(f"  {cls}: {count} ({pct})")

    # 2. 错误模式分析
    error_analysis = analyze_errors(rows)
    print("\n=== 错误率分析 ===")
    for source_class, rates in error_analysis['error_rates'].items():
        print(f"\n{source_class} (n={rates['total_samples']}):")
        print(f"  v3 错误率: {rates['v3_error_rate']*100:.1f}%")
        print(f"  gate 错误率: {rates['gate_error_rate']*100:.1f}%")
        print(f"  fused 错误率: {rates['fused_error_rate']*100:.1f}%")

    # 3. 快捷方式学习诊断
    diagnosis = diagnose_shortcut_learning(error_analysis)
    print("\n=== 快捷方式学习诊断 ===")
    print(f"风险等级: {diagnosis['shortcut_learning_risk']}")
    print(f"诊断: {diagnosis['diagnosis']}")

    # 保存报告
    report = {
        "distribution": distribution,
        "error_analysis": error_analysis,
        "shortcut_diagnosis": diagnosis
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存到: {args.output}")


if __name__ == "__main__":
    main()

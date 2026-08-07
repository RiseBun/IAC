"""可视化 clean gate 的交叉注意力权重。

分析目的：
1. 检查 gate 的注意力是否呈现时序对齐（对角线模式）
2. 还是全局关注（每个轨迹点关注所有视觉 token）
3. 或者退化到只关注少数几个视觉 token

对角线模式 → 学到时空对齐
均匀模式 → 未学到有效对齐（可能是快捷方式）
稀疏模式 → 依赖少数关键帧（可能过拟合）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from train_visual_mismatch_gate_scorer import MismatchGate


def load_gate_model(model_path: Path) -> tuple[MismatchGate, dict]:
    """加载 clean gate 模型。"""
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    metadata = bundle.get("metadata", {})
    train_args = metadata.get("args", {})

    model = MismatchGate(
        int(metadata["visual_dim"]),
        int(metadata["scalar_dim"]),
        int(train_args.get("visual_hidden_dim", 32)),
        int(train_args.get("hidden_dim", 64)),
        float(train_args.get("dropout", 0.0)),
        str(metadata.get("interaction_kind", "traj_cross_attention")),
        int((metadata.get("traj_shape") or [8, 5])[-1]),
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, metadata


def extract_attention_weights(
    model: MismatchGate,
    visual: torch.Tensor,   # [B, 16, 1024]
    traj: torch.Tensor,     # [B, 8, 5]
    scalar: torch.Tensor,   # [B, 1]
) -> torch.Tensor:
    """从 gate 中提取交叉注意力权重。

    返回: [B, 8, 16] - 每个样本 8 个轨迹点对 16 个视觉 token 的注意力
    """
    with torch.no_grad():
        # 投影
        z = model.visual_proj(visual)  # [B, 16, hidden]
        query = model.traj_proj(traj)  # [B, 8, hidden]

        # 手动实现 attention 以获取权重
        # 使用 MultiheadAttention 的 need_weights=True
        attended, attn_weights = model.cross_attn(
            query, z, z,
            need_weights=True,
            average_attn_weights=True  # 4 heads 平均
        )
        # attn_weights: [B, 8, 16]

    return attn_weights


def analyze_attention_pattern(attn: np.ndarray) -> dict[str, Any]:
    """分析注意力矩阵的模式。

    输入: attn [8, 16] - 8 个轨迹点对 16 个视觉 token 的权重
    """
    # 每行归一化（应该已经归一化，但保险起见）
    attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-8)

    # 1. 对角线得分：轨迹点 t 应该关注视觉 token 2t（因为 16/8=2）
    diagonal_mass = 0.0
    for t in range(8):
        # 轨迹 t 对应视觉 [2t, 2t+1]
        target_tokens = [2 * t, 2 * t + 1]
        diagonal_mass += attn[t, target_tokens].sum()
    diagonal_score = diagonal_mass / 8  # 平均每行对角线区域的质量

    # 2. 均匀性得分：熵越大越均匀
    entropy_per_row = -np.sum(attn * np.log(attn + 1e-8), axis=1)
    max_entropy = np.log(16)  # 均匀分布的熵
    normalized_entropy = entropy_per_row.mean() / max_entropy

    # 3. 稀疏度：top-1 的注意力权重
    top1_weight = attn.max(axis=1).mean()

    # 4. 全局关注度：所有行的方差（如果模式一致 → 全局固定关注）
    row_variance = attn.var(axis=0).mean()

    # 诊断模式
    if diagonal_score > 0.35 and normalized_entropy < 0.85:
        pattern = "temporal_aligned"
        diagnosis = "✅ 学到了时空对齐（对角线模式）"
    elif normalized_entropy > 0.95:
        pattern = "uniform"
        diagnosis = "❌ 注意力过于均匀，未学到有效对齐"
    elif top1_weight > 0.5:
        pattern = "sparse"
        diagnosis = "⚠️  注意力过于稀疏，可能只关注少数关键帧"
    else:
        pattern = "mixed"
        diagnosis = "🤔 混合模式，未展现明确的时空对齐"

    return {
        "diagonal_score": float(diagonal_score),
        "normalized_entropy": float(normalized_entropy),
        "top1_weight": float(top1_weight),
        "row_variance": float(row_variance),
        "pattern": pattern,
        "diagnosis": diagnosis
    }


def render_ascii_heatmap(attn: np.ndarray, title: str = "") -> str:
    """渲染 ASCII 热力图。"""
    lines = [f"\n{title}"]
    lines.append("       " + " ".join(f"v{i:02d}" for i in range(16)))
    lines.append("       " + " ".join("---" for _ in range(16)))

    for t in range(8):
        row_str = f"t{t}({t*0.5:.1f}s)"
        for v in range(16):
            weight = attn[t, v]
            if weight > 0.15:
                char = "██"
            elif weight > 0.10:
                char = "▓▓"
            elif weight > 0.05:
                char = "▒▒"
            elif weight > 0.02:
                char = "░░"
            else:
                char = "  "
            row_str += f" {char}"
        # 标记对角线目标位置
        expected_v = 2 * t
        row_str += f"  → expected: v{expected_v:02d}-v{expected_v+1:02d}"
        lines.append(row_str)

    lines.append("\n图例: ██>0.15  ▓▓>0.10  ▒▒>0.05  ░░>0.02")
    return "\n".join(lines)


def simulate_attention_patterns() -> dict[str, np.ndarray]:
    """生成三种典型注意力模式用于演示。"""
    rng = np.random.default_rng(42)

    # 1. 完美时空对齐（对角线）
    aligned = np.zeros((8, 16))
    for t in range(8):
        for v in range(16):
            expected_v = 2 * t
            dist = abs(v - expected_v)
            aligned[t, v] = np.exp(-dist / 2)
    aligned = aligned / aligned.sum(axis=1, keepdims=True)

    # 2. 均匀分布（学不到东西）
    uniform = np.ones((8, 16)) / 16 + rng.normal(0, 0.005, (8, 16))
    uniform = np.abs(uniform)
    uniform = uniform / uniform.sum(axis=1, keepdims=True)

    # 3. 稀疏关注（快捷方式）
    sparse = np.ones((8, 16)) * 0.01
    for t in range(8):
        # 只关注 v0 和 v15
        sparse[t, 0] = 0.45
        sparse[t, 15] = 0.45
    sparse = sparse / sparse.sum(axis=1, keepdims=True)

    # 4. 弱对齐（部分学到）
    weak_align = np.ones((8, 16)) / 16 * 0.5
    for t in range(8):
        expected_v = 2 * t
        weak_align[t, expected_v] += 0.15
        weak_align[t, min(15, expected_v + 1)] += 0.15
    weak_align = weak_align / weak_align.sum(axis=1, keepdims=True)

    return {
        "aligned": aligned,
        "uniform": uniform,
        "sparse": sparse,
        "weak_align": weak_align
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/clean_vjepa_traj_gate.pt"))
    parser.add_argument("--rows", type=Path, help="评分 JSONL")
    parser.add_argument("--visual-cache", type=Path, help="V-JEPA 特征缓存 .pt")
    parser.add_argument("--num-samples", type=int, default=3, help="每种类型可视化几个样本")
    parser.add_argument("--output", type=Path, default=Path("attention_visualization.json"))
    parser.add_argument("--demo-only", action="store_true",
                       help="只运行演示模式（不加载真实模型）")
    args = parser.parse_args()

    print("=" * 80)
    print("Clean Gate 交叉注意力可视化")
    print("=" * 80)

    if args.demo_only or not args.rows:
        print("\n[演示模式] 展示典型注意力模式\n")

        patterns = simulate_attention_patterns()

        results = {}
        for name, attn in patterns.items():
            print(render_ascii_heatmap(attn, title=f"\n【模式: {name}】"))
            analysis = analyze_attention_pattern(attn)
            print(f"\n分析结果:")
            print(f"  对角线得分: {analysis['diagonal_score']:.3f}")
            print(f"  归一化熵: {analysis['normalized_entropy']:.3f}")
            print(f"  Top1 权重: {analysis['top1_weight']:.3f}")
            print(f"  行方差: {analysis['row_variance']:.5f}")
            print(f"  诊断: {analysis['diagnosis']}")
            print("-" * 80)
            results[name] = analysis

        # 保存结果
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n分析结果已保存: {args.output}")

        # 总结
        print("\n" + "=" * 80)
        print("总结：如何解读注意力模式")
        print("=" * 80)
        print("""
1. 对角线模式（aligned）:
   - diagonal_score > 0.35, entropy < 0.85
   - ✅ 模型学到时空对齐：轨迹第 t 秒关注视频第 t 秒
   - 这是我们期望的理想模式

2. 均匀模式（uniform）:
   - entropy > 0.95
   - ❌ 模型未学到有效对齐，注意力权重接近均匀分布
   - 说明交叉注意力实际上是"空"的，可能靠 residual/scalar 分支决策

3. 稀疏模式（sparse）:
   - top1_weight > 0.5
   - ⚠️  模型只关注少数几个视觉 token（如第一帧/最后一帧）
   - 说明学到了快捷方式：不需要完整时序，只需边界帧

4. 弱对齐（weak_align）:
   - diagonal_score 0.20-0.35
   - 🤔 有对齐倾向但不明显，说明学习不充分

对当前 IAC gate 的建议：
- 如果实际注意力是 uniform → 交叉注意力设计失效
- 如果实际注意力是 sparse → 需要在训练时加入时序约束
- 如果实际注意力是 aligned → 主线设计合理
""")

    else:
        # 真实模型可视化
        print(f"\n加载模型: {args.model}")
        model, metadata = load_gate_model(args.model)
        print(f"模型元数据: interaction_kind={metadata.get('interaction_kind')}")
        # 实际数据加载和推理需要完整数据集，此处省略
        print("\n⚠️  真实数据可视化需要评估数据集和 V-JEPA 缓存")
        print("请使用 --demo-only 运行演示模式")


if __name__ == "__main__":
    main()

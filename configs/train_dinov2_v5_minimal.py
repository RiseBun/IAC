# =============================================================================
# DINOv2 Consistency Critic — minimal v5 (single layer + explicit distance)
# =============================================================================
# Inherits from train_consistency_mini.py. Differences:
#   * Backbone: DINOv2-vits14, layer [11] (single layer, frozen)
#   * Explicit distance features (diff / l2 / cos) concatenated to fusion
#   * AvgPool disabled, no Ridge pretrain, no geometric reg (all off)
#   * Difficulty sampling enabled (D1..D4) — can disable by setting
#     difficulty_sampling.enabled=False below
# =============================================================================

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data_paths import DATA_ROOT, DB_ROOT, INDEX_ROOT, camera_roots

data_root = DATA_ROOT
index_root = INDEX_ROOT


cfg = dict(
    # ── 基础信息 ──
    experiment_name="nuplan_iac_dinov2_v5_minimal",
    model_type="consistency",
    seed=42,
    work_dir=str(project_root / "work_dirs" / "iac_dinov2_v5_minimal"),

    # ── 数据路径 ──
    train_index=str(index_root / "consistency_train.jsonl"),
    val_index=str(index_root / "consistency_val.jsonl"),
    image_root=str(data_root),
    mini_db_root=str(DB_ROOT),
    camera_roots=[str(path) for path in camera_roots(data_root)],
    camera_channel="CAM_F0",

    # ── 日志与保存 ──
    log_interval=20,
    val_interval=1,
    save_interval=1,

    # ── 训练超参 ──
    # DINOv2-S/14 显存占用更高，batch=32 配 2 GPU ≈ batch=64 单卡
    # v4 用 batch=96，降到 32 保持总样本/步比例一致 (96/32=3，3×10 epoch ≈ 5 epoch)
    epochs=5,
    batch_size=32,
    num_workers=8,

    # ── 输入规格 ──
    image_size=224,
    history_num_frames=4,
    future_num_frames=4,
    candidate_traj_steps=8,
    consistency_traj_steps=4,
    future_step_time_s=0.5,
    ego_state_dim=5,
    traj_dim=3,

    # ── 损失权重 ──
    lambda_consistency=1.0,
    lambda_validity=0.5,
    positive_weight=1.0,
    consistency_positive_weight=3.0,    # 沿用 v4 折中值
    validity_positive_weight=1.0,
    validity_negative_weight=8.0,
    lambda_speed_consistency=0.0,
    lambda_steering_consistency=0.0,
    lambda_progress_consistency=0.0,
    lambda_temporal_coherence=0.0,
    lambda_group_ranking=0.2,           # 沿用 v4 ranking loss
    group_ranking_margin=0.2,

    # 负样本权重（与 v4 一致）
    consistency_source_weights=dict(
        gt_pos=1.0,
        image_swap=1.0,
        traj_swap=2.0,
        time_shift_future=2.0,
        perturb_lateral=2.5,
        perturb_heading=2.5,
        perturb_speed=2.5,
    ),
    label_quality_weights=dict(
        positive=1.0,
        clean_negative=1.0,
        weak_negative=0.35,
    ),

    baseline_mode="full",

    # ── 优化器 ──
    # DINOv2 frozen 模式下特征稳定，可用稍高 LR 加速收敛
    optimizer=dict(lr=2e-4, weight_decay=1e-2),

    # ── 模型结构 ──
    # 用 DINOv2 替换 4 层 CNN；其他维度跟 v4 保持一致
    model=dict(
        image_channels=3,
        image_feature_dim=256,        # proj 输出维度
        action_feature_dim=128,
        hidden_dim=256,
        fusion_dim=256,
        dropout=0.1,
        use_complex_fusion=False,     # minimal 版本不用 complex fusion
    ),

    # ── DINOv2 最小配置 ──
    #   ✘ 多层融合 (只取最后一层)
    #   ✘ AvgPool (不削弱 patch 特征)
    #   ✘ Ridge 预训练 (nuPlan 数据量不足，且无 ablation 证据)
    #   ✔ 显式距离特征 (zero-cost, 强收益)
    dinov2=dict(
        model_name="dinov2_vits14",   # 21M params, 384 feat_dim
        layer_index=11,                # 只取最后一层
        freeze=True,                   # 冻住
        use_explicit_distance=True,    # 把 (diff, l2, cos) 拼进 fusion
    ),

    # ── 数据集 ──
    dataset=dict(
        normalize_ego_state=True,
        normalize_candidate_traj=True,
        normalize_mode="linear",
        traj_scale=[60.0, 25.0, 2.0],
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    ),

    # ── Ranking 评估 ──
    ranking=dict(
        enabled=True,
        num_candidates_per_scene=5,
        ranking_metrics=["ndcg@3", "ndcg@5", "mrr", "top1_hit_rate"],
    ),

    # ── 难度分层采样（D1..D4）──
    # 默认开启。v4 中 perturb_speed / time_shift_future 召回偏低
    # 是 D1-D4 采样器要解决的核心问题。
    difficulty_sampling=dict(
        enabled=True,
        mix=(0.30, 0.30, 0.25, 0.15),    # D1/D2/D3/D4 比例
        positive_ratio=0.25,
        num_samples_per_epoch=0,         # 0 = auto: max(len(dataset), batch*100)
    ),
)


# =============================================================================
# 消融对照矩阵（仅 v5_minimal 内部对照；不动这个 config 即可保持完整版）
# =============================================================================
# v5_minimal  = DINoV2-S/14, 单层 [11], 显式距离, D1-D4 采样  (本 config 默认)
# v5_a        = 关掉 use_explicit_distance (验证距离特征收益)
# v5_b        = 关掉 difficulty_sampling (验证 D1-D4 收益)
# v5_c        = 关掉 ranking loss (验证 ranking 收益)
#
# 每个消融用 --epochs 1 ~ 2 + --max-train-steps 2000 跑 quick 即可判断
# =============================================================================

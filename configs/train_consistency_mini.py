import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data_paths import DATA_ROOT, DB_ROOT, INDEX_ROOT, camera_roots

data_root = DATA_ROOT
index_root = INDEX_ROOT


cfg = dict(
    experiment_name="nuplan_iac_full",
    model_type="consistency",
    seed=42,
    work_dir=str(project_root / "work_dirs" / "iac_full"),

    train_index=str(index_root / "consistency_train.jsonl"),
    val_index=str(index_root / "consistency_val.jsonl"),
    image_root=str(data_root),
    mini_db_root=str(DB_ROOT),
    camera_roots=[str(path) for path in camera_roots(data_root)],
    camera_channel="CAM_F0",

    log_interval=20,
    val_interval=1,
    save_interval=1,

    epochs=30,
    batch_size=8,
    num_workers=4,

    image_size=224,
    history_num_frames=4,
    future_num_frames=4,
    candidate_traj_steps=8,
    consistency_traj_steps=4,
    future_step_time_s=0.5,
    ego_state_dim=5,
    traj_dim=3,

    lambda_consistency=1.0,
    lambda_validity=0.5,
    positive_weight=1.0,
    consistency_positive_weight=3.0,
    validity_positive_weight=1.0,
    validity_negative_weight=8.0,
    default_consistency_weight=1.0,
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

    lambda_speed_consistency=0.0,
    lambda_steering_consistency=0.0,
    lambda_progress_consistency=0.0,
    lambda_temporal_coherence=0.0,
    lambda_group_ranking=0.2,
    group_ranking_margin=0.2,
    baseline_mode="full",

    optimizer=dict(lr=1e-4, weight_decay=1e-2),

    model=dict(
        image_channels=3,
        image_feature_dim=256,
        action_feature_dim=128,
        hidden_dim=256,
        fusion_dim=256,
        dropout=0.1,
        use_action_visual_interaction=True,
        temporal_encoder="gru",
    ),

    dataset=dict(
        normalize_ego_state=True,
        normalize_candidate_traj=True,
        normalize_mode="linear",
        traj_scale=[60.0, 25.0, 2.0],
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    ),

    ranking=dict(
        enabled=True,
        group_batches=True,
        loss_weight=0.2,
        margin=0.2,
        num_candidates_per_scene=5,
        ranking_metrics=["ndcg@3", "ndcg@5", "mrr", "top1_hit_rate"],
    ),

    # Kept for ablations. When group ranking is enabled, group_batches take
    # precedence because ranking loss needs in-group positive/negative pairs.
    difficulty_sampling=dict(
        enabled=False,
        mix=(0.30, 0.30, 0.25, 0.15),
        positive_ratio=0.25,
        num_samples_per_epoch=0,
    ),
)

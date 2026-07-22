"""Supported-set IAC v3 probe: train video-action as a side head.

The goal is to test whether video_action_match_logit can independently rank
image-trajectory consistency before allowing it to affect the final scorer.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_supported_set_listwise_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_supported_set_v3_videoaction_probe_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_supported_set_v3_videoaction_probe_vnext"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_video_action_cross_attention"] = True
cfg["dinov2"]["video_action_add_to_shared"] = False
cfg["dinov2"]["video_action_num_heads"] = 4
cfg["dinov2"]["video_action_mix"] = 0.0
cfg["dinov2"]["use_future_latent_prediction"] = False
cfg["dinov2"]["future_latent_mix"] = 0.0
cfg["dinov2"]["find_unused_parameters"] = True

# Probe only: make the video-action head compete as a side scorer.
cfg["lambda_video_action_match"] = 0.08
cfg["lambda_video_action_rank"] = 0.16
cfg["video_action_rank_margin"] = 0.14
cfg["video_action_rank_min_target_gap"] = 0.08

cfg["lambda_future_latent_prediction"] = 0.0
cfg["lambda_future_latent_match"] = 0.0

# Do not update the main scorer in this probe. It remains the supported-set
# baseline while the video-action head learns its own ranking surface.
cfg["lambda_consistency"] = 0.0
cfg["lambda_image_trajectory_consistency_head"] = 0.0
cfg["lambda_group_ranking"] = 0.0
cfg["lambda_group_hard_negative"] = 0.0
cfg["lambda_trajectory_reasonableness"] = 0.0
cfg["lambda_path_evidence_consistency"] = 0.0
cfg["lambda_motion_rule_match"] = 0.0
cfg["lambda_motion_rule_rank"] = 0.0

cfg["trainable_parameter_prefixes"] = [
    "traj_token_encoder",
    "video_to_traj_attn",
    "traj_to_video_attn",
    "video_action_fusion",
    "video_action_match_head",
]

cfg["checkpoint_metric"] = "val_loss"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 1.5e-5

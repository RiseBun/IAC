"""Supported-set IAC v3: trajectory-aware video-action consistency.

This version strengthens the architecture around the actual benchmark
question: does this trajectory correspond to this history/future video?

PDMS/EPDMS still supervises trajectory reasonableness and supported positives,
but it is not a raw input to the final consistency scorer.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_supported_set_listwise_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_supported_set_v3_videoaction_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_supported_set_v3_videoaction_vnext"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_video_action_cross_attention"] = True
cfg["dinov2"]["video_action_num_heads"] = 4
cfg["dinov2"]["video_action_mix"] = 0.15
cfg["dinov2"]["use_future_latent_prediction"] = True
cfg["dinov2"]["future_latent_mix"] = 0.12
cfg["dinov2"]["find_unused_parameters"] = True

# New v3 supervision:
# - video_action_* learns direct image-trajectory correspondence.
# - future_latent_* forces history+trajectory to explain the observed future.
cfg["lambda_video_action_match"] = 0.14
cfg["lambda_video_action_rank"] = 0.10
cfg["video_action_rank_margin"] = 0.14
cfg["video_action_rank_min_target_gap"] = 0.08
cfg["lambda_future_latent_prediction"] = 0.12
cfg["lambda_future_latent_match"] = 0.12
cfg["future_latent_positive_threshold"] = 0.70

# Keep the supported-set objective as the main supervision. Slightly reduce
# older rule pressure so v3 heads can learn the correspondence signal.
cfg["lambda_consistency"] = 0.46
cfg["lambda_image_trajectory_consistency_head"] = 0.28
cfg["lambda_group_ranking"] = 0.34
cfg["lambda_group_hard_negative"] = 0.16
cfg["lambda_trajectory_reasonableness"] = 0.24
cfg["lambda_path_evidence_consistency"] = 0.14
cfg["lambda_motion_rule_match"] = 0.06
cfg["lambda_motion_rule_rank"] = 0.06

cfg["trainable_parameter_prefixes"] = list(cfg.get("trainable_parameter_prefixes", []))
for prefix in [
    "consistency_traj_encoder",
    "ego_encoder",
    "shared_fusion",
    "consistency_head",
    "speed_consistency_head",
    "steering_consistency_head",
    "progress_consistency_head",
    "temporal_coherence_head",
    "traj_token_encoder",
    "video_to_traj_attn",
    "traj_to_video_attn",
    "video_action_fusion",
    "video_action_match_head",
    "future_latent_predictor",
    "future_latent_match_head",
]:
    if prefix not in cfg["trainable_parameter_prefixes"]:
        cfg["trainable_parameter_prefixes"].append(prefix)

cfg["checkpoint_metric"] = "val_iac_consistency"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 1.2e-5

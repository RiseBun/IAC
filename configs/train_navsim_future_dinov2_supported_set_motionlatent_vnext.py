"""Supported-set IAC with explicit visual/trajectory motion latent alignment."""

from __future__ import annotations

from configs.train_navsim_future_dinov2_supported_set_listwise_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_supported_set_motionlatent_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_supported_set_motionlatent_vnext"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_motion_latent_alignment"] = True
cfg["dinov2"]["motion_latent_dim"] = 64

cfg["lambda_motion_latent_match"] = 0.12
cfg["lambda_motion_latent_align"] = 0.10
cfg["motion_latent_positive_threshold"] = 0.70
cfg["motion_latent_negative_threshold"] = 0.25
cfg["motion_latent_negative_margin"] = 0.15

cfg["lambda_motion_rule_match"] = 0.08
cfg["lambda_motion_rule_rank"] = 0.08

cfg["trainable_parameter_prefixes"] = list(cfg.get("trainable_parameter_prefixes", []))
for prefix in [
    "visual_motion_latent_head",
    "traj_motion_latent_head",
    "motion_latent_match_head",
]:
    if prefix not in cfg["trainable_parameter_prefixes"]:
        cfg["trainable_parameter_prefixes"].append(prefix)

cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 1.4e-5

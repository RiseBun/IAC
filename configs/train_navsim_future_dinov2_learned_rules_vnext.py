"""Learnable motion-rule consistency for IAC.

This continuation turns explicit human-style checks into trainable signals:
visual motion attributes are predicted from history/future images, trajectory
attributes are computed from the candidate path, and a small learned residual
judges whether the two agree.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_history_transition_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_learned_rules_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_learned_rules_vnext"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_learned_motion_rules"] = True
cfg["dinov2"]["motion_rule_attr_dim"] = 12
cfg["dinov2"]["motion_rule_mix"] = 0.08

cfg["lambda_motion_rule_attribute"] = 0.18
cfg["lambda_motion_rule_match"] = 0.14
cfg["motion_rule_attribute_min_target"] = 0.55
cfg["motion_rule_match_use_soft_targets"] = True

cfg["trainable_parameter_prefixes"] = list(
    cfg.get("trainable_parameter_prefixes", [])
)
for prefix in ("motion_rule_visual_head", "motion_rule_match_head"):
    if prefix not in cfg["trainable_parameter_prefixes"]:
        cfg["trainable_parameter_prefixes"].append(prefix)

cfg["checkpoint_metric"] = "val_iac_precision"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 3.0e-5

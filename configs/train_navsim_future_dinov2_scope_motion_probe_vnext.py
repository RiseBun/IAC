"""Candidate-blind DINO motion probe.

This config isolates the image-side motion evidence question:

  DINO frame features -> temporal motion head -> trajectory comparator

The base IAC scorer is resumed from an existing checkpoint and frozen.  Only the
scope motion head and uncertainty-aware comparator are trained, so improvements
or failures can be attributed to the visual motion evidence branch rather than a
general re-tuning of the consistency model.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_scope_motion_head import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_scope_motion_probe_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_scope_motion_probe_vnext"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["motion_rule_mix"] = 0.0

cfg["lambda_consistency"] = 0.0
cfg["lambda_validity"] = 0.0
cfg["lambda_rank"] = 0.0
cfg["lambda_progress_alignment"] = 0.0
cfg["lambda_path_grounding"] = 0.0
cfg["lambda_path_evidence_consistency"] = 0.0
cfg["lambda_trajectory_specific_grounding"] = 0.0
cfg["lambda_reasonableness"] = 0.0

cfg["lambda_motion_rule_attribute"] = 0.35
cfg["motion_rule_attribute_weight_mode"] = "threshold"
cfg["motion_rule_attribute_min_target"] = 0.999
cfg["lambda_motion_rule_match"] = 0.20
cfg["lambda_motion_rule_rank"] = 0.30
cfg["motion_rule_match_use_soft_targets"] = True
cfg["motion_rule_rank_margin"] = 0.20
cfg["motion_rule_rank_min_target_gap"] = 0.08

cfg["trainable_parameter_prefixes"] = [
    "scope_motion_head",
    "scope_motion_comparator",
]

cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 5.0e-5

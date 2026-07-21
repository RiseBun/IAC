"""Structured learnable-rule continuation for IAC.

This is the stronger version of the learned-rules idea. It keeps the
history-conditioned path evidence, but upgrades motion rules from one global
attribute vector to global + temporal-segment attributes and adds a listwise
ranking objective on the rule-match head.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_learned_rules_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_structured_rules_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_structured_rules_vnext"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_learned_motion_rules"] = True
cfg["dinov2"]["motion_rule_segment_count"] = 3
cfg["dinov2"]["motion_rule_attr_dim"] = 36
cfg["dinov2"]["motion_rule_mix"] = 0.14
cfg["dinov2"]["path_evidence_use_transition_context"] = True

cfg["lambda_motion_rule_attribute"] = 0.22
cfg["lambda_motion_rule_match"] = 0.12
cfg["lambda_motion_rule_rank"] = 0.12
cfg["motion_rule_attribute_weight_mode"] = "soft_target"
cfg["motion_rule_match_use_soft_targets"] = True
cfg["motion_rule_rank_margin"] = 0.16
cfg["motion_rule_rank_min_target_gap"] = 0.08

# Dormant unless these fields are present in the index. This lets PDMS/EPDMS
# become a weak candidate-quality target without changing the trainer again.
cfg["candidate_quality_score_fields"] = [
    "epdms_score",
    "pdms_score",
    "planning_score",
    "candidate_quality_score",
]
cfg["candidate_quality_score_weight"] = 0.25
cfg["candidate_quality_allowed_sources"] = [
    "gt_pos",
    "perturb_speed",
    "perturb_lateral",
    "perturb_heading",
]

cfg["trainable_parameter_prefixes"] = [
    "consistency_traj_encoder",
    "ego_encoder",
    "shared_fusion",
    "consistency_head",
    "path_conditioned_traj_proj",
    "path_conditioned_temporal_head",
    "path_conditioned_fusion",
    "path_residual_head",
    "path_evidence_head",
    "path_evidence_gate_head",
    "progress_alignment_head",
    "motion_rule_visual_head",
    "motion_rule_match_head",
]

cfg["lambda_consistency"] = 0.55
cfg["lambda_group_ranking"] = 0.30
cfg["lambda_group_hard_negative"] = 0.10
cfg["lambda_path_evidence_consistency"] = 0.30
cfg["lambda_progress_alignment"] = 0.08
cfg["lambda_history_counterfactual"] = 0.12

cfg["checkpoint_metric"] = "val_iac_precision"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 2.5e-5

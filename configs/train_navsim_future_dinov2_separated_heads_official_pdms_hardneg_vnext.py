"""Separated-head IAC training with official PDMS auxiliary supervision.

This version keeps the task boundary explicit:

- image_trajectory_consistency learns visual correspondence.
- trajectory_reasonableness learns official PDM/feasibility.
- final consistency uses learned internal fusion, never raw PDMS input.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_multisolution_official_pdms_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_separated_heads_official_pdms_hardneg_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_separated_heads_official_pdms_hardneg_vnext"

cfg["train_index"] = "indices_navsim_future/consistency_train_official_pdms_highpdm_mismatch.jsonl"
cfg["val_index"] = "indices_navsim_future/consistency_val_official_pdms_highpdm_mismatch.jsonl"

cfg["candidate_quality_score_fields"] = [
    "official_epdms_score",
    "epdms_score",
    "official_pdm_score",
    "pdms_score",
    "planning_score",
    "candidate_quality_score",
]

# Do not rewrite final consistency targets with PDMS. PDMS is only an
# auxiliary reasonableness target below.
cfg["candidate_quality_score_weight"] = 0.0
cfg["candidate_quality_target_mode"] = "blend"
cfg["consistency_soft_target_mode"] = "none"
cfg["consistency_source_soft_targets"] = {}
cfg["auxiliary_consistency_target_mode"] = "hard"
cfg["consistency_positive_mask_mode"] = "hard"

correspondence_sources = [
    "gt_pos",
    "image_swap",
    "time_shift_future",
    "traj_swap",
    "reverse_traj",
    "high_pdm_image_mismatch",
]
cfg["consistency_supervision_sources"] = correspondence_sources
cfg["auxiliary_consistency_supervision_sources"] = correspondence_sources
cfg["consistency_ignored_sources"] = [
    "perturb_speed",
    "perturb_lateral",
    "perturb_heading",
]
cfg["auxiliary_consistency_ignored_sources"] = list(cfg["consistency_ignored_sources"])

cfg["trajectory_reasonableness_allowed_sources"] = [
    "gt_pos",
    "perturb_speed",
    "perturb_lateral",
    "perturb_heading",
    "high_pdm_image_mismatch",
]
cfg["trajectory_reasonableness_source_weights"] = {
    "gt_pos": 1.0,
    "perturb_speed": 0.9,
    "perturb_lateral": 0.9,
    "perturb_heading": 0.9,
    "high_pdm_image_mismatch": 1.2,
}

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_trajectory_reasonableness_head"] = True
cfg["dinov2"]["use_learned_consistency_fusion"] = True
cfg["dinov2"]["consistency_fusion_reasonableness_mix"] = 0.20
cfg["dinov2"]["consistency_fusion_use_repr"] = True
cfg["dinov2"]["find_unused_parameters"] = True

cfg["lambda_consistency"] = 0.62
cfg["lambda_image_trajectory_consistency_head"] = 0.35
cfg["lambda_trajectory_reasonableness"] = 0.28
cfg["lambda_group_ranking"] = 0.22
cfg["lambda_group_hard_negative"] = 0.18
cfg["lambda_path_evidence_consistency"] = 0.18
cfg["lambda_motion_rule_match"] = 0.08
cfg["lambda_motion_rule_rank"] = 0.08
cfg["motion_rule_match_use_soft_targets"] = False
cfg["motion_rule_attribute_weight_mode"] = "threshold"

cfg["group_ranking_target_mode"] = "hard"
cfg["group_hard_negative_target_mode"] = "hard"
cfg["group_hard_negative_sources"] = [
    "image_swap",
    "time_shift_future",
    "traj_swap",
    "reverse_traj",
    "high_pdm_image_mismatch",
]

cfg["consistency_source_weights"] = dict(cfg.get("consistency_source_weights", {}))
cfg["consistency_source_weights"].update(
    {
        "gt_pos": 1.0,
        "perturb_speed": 0.0,
        "perturb_lateral": 0.0,
        "perturb_heading": 0.0,
        "image_swap": 1.0,
        "time_shift_future": 1.1,
        "traj_swap": 1.1,
        "reverse_traj": 0.9,
        "high_pdm_image_mismatch": 1.6,
    }
)
cfg["consistency_source_margins"] = dict(cfg.get("consistency_source_margins", {}))
cfg["consistency_source_margins"].update(
    {
        "high_pdm_image_mismatch": 0.18,
    }
)

cfg["trainable_parameter_prefixes"] = list(cfg.get("trainable_parameter_prefixes", []))
for prefix in [
    "validity_traj_encoder",
    "validity_fusion",
    "trajectory_reasonableness_head",
    "consistency_fusion_gate_head",
]:
    if prefix not in cfg["trainable_parameter_prefixes"]:
        cfg["trainable_parameter_prefixes"].append(prefix)

cfg["checkpoint_metric"] = "val_iac_precision"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 1.8e-5

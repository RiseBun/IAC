"""Supported-set/listwise IAC training.

This config changes the supervision, not the architecture:

- GT is a positive, but not the only positive.
- Same-scene perturbations with high official PDMS/EPDMS become soft positives.
- Medium-quality same-scene perturbations are treated as unknown, not negatives.
- Cross-image/time/trajectory mismatches remain hard negatives.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_separated_heads_official_pdms_hardneg_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_supported_set_listwise_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_supported_set_listwise_vnext"

# Keep the same high-PDMS mismatch data; the change is how same-scene
# perturbations are supervised.
cfg["train_index"] = "indices_navsim_future/consistency_train_official_pdms_highpdm_mismatch.jsonl"
cfg["val_index"] = "indices_navsim_future/consistency_val_official_pdms_highpdm_mismatch.jsonl"

same_scene_sources = [
    "perturb_speed",
    "perturb_lateral",
    "perturb_heading",
]
hard_mismatch_sources = [
    "image_swap",
    "time_shift_future",
    "traj_swap",
    "reverse_traj",
    "high_pdm_image_mismatch",
]

cfg["supported_set_target_mode"] = "pdms_same_scene"
cfg["supported_set_positive_sources"] = same_scene_sources
cfg["supported_set_unknown_sources"] = same_scene_sources
cfg["supported_set_hard_negative_sources"] = hard_mismatch_sources
cfg["supported_set_positive_quality_threshold"] = 0.76
cfg["supported_set_unknown_quality_threshold"] = 0.45
cfg["supported_set_positive_target"] = 0.86
cfg["supported_set_unknown_target"] = 0.50
cfg["supported_set_use_quality_as_target"] = True

# Do not require closeness to GT for a soft-positive same-scene perturbation.
# Requiring high GT IoU would reintroduce the single-GT bias we are removing.
cfg["supported_set_require_geometry_for_positive"] = False
cfg["supported_set_positive_path_iou_threshold"] = 0.35
cfg["supported_set_positive_distance_threshold"] = 2.5

cfg["consistency_supervision_sources"] = [
    "gt_pos",
    *same_scene_sources,
    *hard_mismatch_sources,
]
cfg["auxiliary_consistency_supervision_sources"] = list(
    cfg["consistency_supervision_sources"]
)
cfg["consistency_ignored_sources"] = []
cfg["auxiliary_consistency_ignored_sources"] = []

cfg["auxiliary_consistency_target_mode"] = "soft"
cfg["consistency_positive_mask_mode"] = "soft"
cfg["soft_positive_target_threshold"] = 0.70

cfg["group_ranking_target_mode"] = "soft"
cfg["group_ranking_min_target_gap"] = 0.25
cfg["group_hard_negative_target_mode"] = "soft"
cfg["group_hard_negative_max_target"] = 0.25
cfg["group_hard_negative_sources"] = hard_mismatch_sources
cfg["progress_alignment_target_mode"] = "soft"
cfg["motion_rule_match_use_soft_targets"] = True

cfg["consistency_source_weights"] = dict(cfg.get("consistency_source_weights", {}))
cfg["consistency_source_weights"].update(
    {
        "gt_pos": 1.0,
        "perturb_speed": 0.85,
        "perturb_lateral": 0.85,
        "perturb_heading": 0.85,
        "image_swap": 1.0,
        "time_shift_future": 1.15,
        "traj_swap": 1.15,
        "reverse_traj": 0.9,
        "high_pdm_image_mismatch": 1.45,
    }
)

# Reduce direct BCE pressure and let group/listwise + auxiliary heads carry the
# multi-solution structure.
cfg["lambda_consistency"] = 0.48
cfg["lambda_image_trajectory_consistency_head"] = 0.28
cfg["lambda_group_ranking"] = 0.34
cfg["lambda_group_hard_negative"] = 0.16
cfg["lambda_trajectory_reasonableness"] = 0.24
cfg["lambda_motion_rule_match"] = 0.10
cfg["lambda_motion_rule_rank"] = 0.10

# These hard-label auxiliary losses conflict with supported-set supervision.
cfg["lambda_speed_consistency"] = 0.0
cfg["lambda_steering_consistency"] = 0.0
cfg["lambda_progress_consistency"] = 0.0
cfg["lambda_temporal_coherence"] = 0.0

cfg["checkpoint_metric"] = "val_iac_precision"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 1.5e-5

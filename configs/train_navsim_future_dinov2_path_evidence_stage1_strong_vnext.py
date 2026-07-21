"""Stage 1: make the path-evidence certificate materially stronger.

This stage freezes the main critic surface and trains the candidate-path
certificate harder, with segmented path pooling and stronger exact-path
grounding pressure.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_path_evidence_head import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_path_evidence_stage1_strong_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_path_evidence_stage1_strong_vnext"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_path_conditioned_evidence"] = True
cfg["dinov2"]["use_path_evidence_head"] = True
cfg["dinov2"]["use_path_residual_score"] = True
cfg["dinov2"]["mix_path_evidence_into_consistency"] = False
cfg["dinov2"]["path_conditioned_segment_count"] = 3
cfg["dinov2"]["path_conditioned_forward_m"] = 40.0
cfg["dinov2"]["path_conditioned_lateral_m"] = 10.0
cfg["dinov2"]["path_conditioned_width"] = 0.10

cfg["trainable_parameter_prefixes"] = [
    "path_conditioned_traj_proj",
    "path_conditioned_temporal_head",
    "path_conditioned_fusion",
    "path_residual_head",
    "path_evidence_head",
    "progress_alignment_head",
]

cfg["lambda_consistency"] = 0.0
cfg["lambda_validity"] = 0.0
cfg["lambda_speed_consistency"] = 0.0
cfg["lambda_steering_consistency"] = 0.0
cfg["lambda_progress_consistency"] = 0.0
cfg["lambda_temporal_coherence"] = 0.0
cfg["lambda_group_ranking"] = 0.0
cfg["lambda_group_hard_negative"] = 0.0
cfg["lambda_future_consistency_evidence"] = 0.0

cfg["lambda_path_evidence_consistency"] = 0.70
cfg["path_grounding_score_key"] = "path_evidence_logit"
cfg["trajectory_specific_grounding_score_key"] = "path_evidence_logit"
cfg["lambda_path_grounding"] = 0.15
cfg["path_grounding_margin"] = 0.02
cfg["lambda_path_sky_contrast"] = 0.10
cfg["path_sky_contrast_margin"] = 0.03
cfg["lambda_trajectory_specific_grounding"] = 0.45
cfg["trajectory_specific_grounding_margin"] = 0.06
cfg["trajectory_specific_wrong_selection"] = "mask_iou"
cfg["trajectory_specific_grounding_exclusive"] = True

cfg["lambda_progress_alignment"] = 0.05
cfg["progress_alignment_mode"] = "final_displacement"
cfg["progress_alignment_scale"] = 40.0
cfg["progress_alignment_hard_margin"] = 0.05
cfg["progress_alignment_near_margin"] = 0.005
cfg["progress_alignment_near_weight"] = 0.05
cfg["progress_alignment_hard_sources"] = [
    "image_swap",
    "time_shift_future",
    "traj_swap",
    "reverse_traj",
]
cfg["progress_alignment_near_sources"] = [
    "perturb_speed",
    "perturb_lateral",
    "perturb_heading",
]

cfg["consistency_soft_target_mode"] = "path_iou"
cfg["consistency_soft_target_near_sources"] = [
    "perturb_speed",
    "perturb_lateral",
    "perturb_heading",
]
cfg["consistency_soft_target_hard_negative_sources"] = [
    "image_swap",
    "time_shift_future",
    "traj_swap",
    "reverse_traj",
]
cfg["soft_target_gamma"] = 1.5
cfg["soft_target_min"] = 0.05
cfg["soft_target_distance_tau"] = 2.0
cfg["consistency_source_soft_targets"] = {}

cfg["consistency_source_weights"] = dict(cfg.get("consistency_source_weights", {}))
cfg["consistency_source_weights"].update(
    dict(
        image_swap=0.8,
        time_shift_future=2.0,
        traj_swap=1.5,
        perturb_speed=0.7,
        perturb_lateral=0.7,
        perturb_heading=0.7,
        reverse_traj=1.5,
    )
)

cfg["consistency_source_margins"] = dict(cfg.get("consistency_source_margins", {}))
cfg["consistency_source_margins"].update(
    dict(
        image_swap=0.18,
        time_shift_future=0.22,
        traj_swap=0.16,
        perturb_speed=0.05,
        perturb_lateral=0.05,
        perturb_heading=0.05,
        reverse_traj=0.16,
    )
)

cfg["checkpoint_metric"] = "val_c_score_gap"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 5.0e-5

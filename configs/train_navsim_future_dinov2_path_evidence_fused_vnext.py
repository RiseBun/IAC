"""Second-stage fused judge continuation for IAC.

This stage assumes the path-evidence branch has already learned a usable
candidate-path certificate. It then lets that certificate contribute a small,
bounded residual to the main consistency logit so the raw judge becomes more
path-grounded without turning into a pure shortcut.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_path_evidence_segmented_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_path_evidence_fused_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_path_evidence_fused_vnext"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["mix_path_evidence_into_consistency"] = True
cfg["dinov2"]["path_evidence_mix"] = 0.12
cfg["dinov2"]["path_residual_mix"] = 0.18
cfg["dinov2"]["use_path_evidence_gate"] = True

# Stage 2 should nudge the decision surface, not overwrite it.
cfg["trainable_parameter_prefixes"] = [
    "path_conditioned_traj_proj",
    "path_conditioned_temporal_head",
    "path_conditioned_fusion",
    "path_residual_head",
    "path_evidence_head",
    "progress_alignment_head",
    "consistency_head",
]

# Keep the evidence branch strong, but put slightly more weight on the raw
# judge once the certificate has stabilized.
cfg["lambda_consistency"] = 0.60
cfg["lambda_validity"] = 0.05
cfg["lambda_speed_consistency"] = 0.0
cfg["lambda_steering_consistency"] = 0.0
cfg["lambda_progress_consistency"] = 0.0
cfg["lambda_temporal_coherence"] = 0.0
cfg["lambda_group_ranking"] = 0.28
cfg["lambda_group_hard_negative"] = 0.10
cfg["lambda_path_evidence_consistency"] = 0.35
cfg["lambda_path_grounding"] = 0.10
cfg["path_grounding_score_key"] = "path_evidence_logit"
cfg["trajectory_specific_grounding_score_key"] = "path_evidence_logit"
cfg["lambda_path_sky_contrast"] = 0.05
cfg["path_sky_contrast_margin"] = 0.025
cfg["lambda_trajectory_specific_grounding"] = 0.24
cfg["trajectory_specific_grounding_margin"] = 0.045
cfg["trajectory_specific_wrong_selection"] = "mask_iou"
cfg["trajectory_specific_grounding_exclusive"] = True
cfg["lambda_progress_alignment"] = 0.08
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

cfg["group_hard_negative_margin"] = 0.02
cfg["group_hard_negative_target"] = 0.24

cfg["ranking"] = dict(cfg.get("ranking", {}))
cfg["ranking"].update(
    dict(
        enabled=True,
        group_batches=True,
        max_negatives_per_group=6,
        hard_negative_sources=[
            "time_shift_future",
            "traj_swap",
            "image_swap",
            "reverse_traj",
            "perturb_speed",
            "perturb_heading",
            "perturb_lateral",
        ],
    )
)

cfg["difficulty_sampling"] = dict(cfg.get("difficulty_sampling", {}))
cfg["difficulty_sampling"].update(
    dict(
        enabled=True,
        mix=(0.05, 0.10, 0.30, 0.55),
        positive_ratio=0.25,
    )
)

cfg["checkpoint_metric"] = "val_iac_precision"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 4.0e-5

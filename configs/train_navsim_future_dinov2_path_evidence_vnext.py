"""IAC vNext continuation.

This config keeps the path-evidence certificate as the scientific anchor,
adds the visual progress auxiliary head, and restores ambiguity-aware ranking
pressure so the decision surface can improve without collapsing into a pure
trajectory-geometry shortcut.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_path_evidence_head import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_path_evidence_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_path_evidence_vnext"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_progress_alignment_head"] = True
cfg["dinov2"]["mix_path_evidence_into_consistency"] = False

# Keep the certificate branch trainable, but do not let the vNext run
# overwrite the image encoder backbone. The decision head and the certificate
# head each get their own pressure.
cfg["trainable_parameter_prefixes"] = [
    "consistency_traj_encoder",
    "validity_traj_encoder",
    "ego_encoder",
    "shared_fusion",
    "validity_fusion",
    "consistency_head",
    "speed_consistency_head",
    "steering_consistency_head",
    "progress_consistency_head",
    "temporal_coherence_head",
    "validity_head",
    "path_conditioned_traj_proj",
    "path_conditioned_fusion",
    "path_residual_head",
    "path_evidence_head",
    "progress_alignment_head",
]

# Decision surface: ambiguity-aware ranking plus hard-negative suppression.
cfg["lambda_consistency"] = 0.65
cfg["lambda_validity"] = 0.05
cfg["lambda_speed_consistency"] = 0.0
cfg["lambda_steering_consistency"] = 0.0
cfg["lambda_progress_consistency"] = 0.0
cfg["lambda_temporal_coherence"] = 0.0
cfg["lambda_group_ranking"] = 0.35
cfg["lambda_group_hard_negative"] = 0.08

# Certificate surface: keep exact-path evidence and path grounding explicit.
cfg["lambda_future_consistency_evidence"] = 0.0
cfg["lambda_path_evidence_consistency"] = 0.20
cfg["path_grounding_score_key"] = "path_evidence_logit"
cfg["trajectory_specific_grounding_score_key"] = "path_evidence_logit"
cfg["lambda_path_grounding"] = 0.06
cfg["path_grounding_margin"] = 0.025
cfg["lambda_path_sky_contrast"] = 0.06
cfg["path_sky_contrast_margin"] = 0.025
cfg["lambda_trajectory_specific_grounding"] = 0.18
cfg["trajectory_specific_grounding_margin"] = 0.045
cfg["trajectory_specific_wrong_selection"] = "mask_iou"
cfg["trajectory_specific_grounding_exclusive"] = True

# Auxiliary visual progress prior.
cfg["lambda_progress_alignment"] = 0.15
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

# Ambiguity-aware soft targets and hard negatives.
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
        time_shift_future=2.2,
        traj_swap=1.6,
        perturb_speed=0.7,
        perturb_lateral=0.7,
        perturb_heading=0.7,
        reverse_traj=1.6,
    )
)

cfg["consistency_source_margins"] = dict(cfg.get("consistency_source_margins", {}))
cfg["consistency_source_margins"].update(
    dict(
        image_swap=0.20,
        time_shift_future=0.24,
        traj_swap=0.18,
        perturb_speed=0.05,
        perturb_lateral=0.05,
        perturb_heading=0.05,
        reverse_traj=0.18,
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

cfg["checkpoint_metric"] = "val_iac_consistency"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 5.0e-5

"""Ambiguity-aware continuation for improving judge top1.

Failure analysis showed that most misses are not obvious image mismatches.
They are near-neighbor trajectory perturbations:

- perturb_speed
- perturb_lateral
- perturb_heading

These candidates often remain visually plausible in the same future image.
This config stops treating them as equally hard binary negatives. It uses soft
targets and small ranking margins for visually ambiguous perturbations, while
keeping stronger pressure on more observable mismatches.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_path_evidence_head import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_ambiguity_aware"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_ambiguity_aware"

# This run adapts the global decision surface lightly, while preserving the
# path evidence branch learned in the previous continuation.
cfg.pop("trainable_parameter_prefixes", None)

cfg["lambda_consistency"] = 0.65
cfg["lambda_validity"] = 0.05
cfg["lambda_speed_consistency"] = 0.0
cfg["lambda_steering_consistency"] = 0.0
cfg["lambda_progress_consistency"] = 0.0
cfg["lambda_temporal_coherence"] = 0.0
cfg["lambda_group_ranking"] = 0.35
cfg["lambda_group_hard_negative"] = 0.08
cfg["lambda_path_evidence_consistency"] = 0.20
cfg["lambda_path_grounding"] = 0.06
cfg["lambda_trajectory_specific_grounding"] = 0.18

# Treat near-neighbor geometric perturbations as plausible but imperfect, not
# as fully wrong visual evidence. The target is continuous: candidate/GT
# projected-path IoU^gamma, with a floor for weakly overlapping near-neighbors.
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

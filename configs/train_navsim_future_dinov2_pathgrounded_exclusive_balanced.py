"""Balanced exclusive path grounding continuation.

The exclusive/equal-area objective gives the cleanest trajectory-specific
causal signal, but by itself can hurt TNR. This config keeps that causal signal
while restoring stronger hard-negative pressure for geometry-like negatives.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_pathgrounded_exclusive import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_pathgrounded_exclusive_balanced"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_pathgrounded_exclusive_balanced"

# Keep trajectory-specific evidence pressure, but avoid overwhelming the
# ranking/hard-negative objectives that protect TNR.
cfg["lambda_trajectory_specific_grounding"] = 0.14
cfg["trajectory_specific_grounding_margin"] = 0.035
cfg["lambda_path_grounding"] = 0.10

# Re-strengthen negative suppression. The main failure after exclusive training
# is false positives, especially geometry-like perturbations.
cfg["lambda_group_ranking"] = 0.45
cfg["lambda_group_hard_negative"] = 0.24
cfg["group_hard_negative_margin"] = 0.08
cfg["group_hard_negative_target"] = 0.22

cfg["consistency_source_weights"] = dict(cfg.get("consistency_source_weights", {}))
cfg["consistency_source_weights"].update(
    dict(
        image_swap=0.6,
        traj_swap=2.8,
        time_shift_future=3.2,
        perturb_heading=3.8,
        perturb_lateral=3.8,
        perturb_speed=5.0,
        reverse_traj=2.8,
    )
)

cfg["consistency_source_margins"] = dict(cfg.get("consistency_source_margins", {}))
cfg["consistency_source_margins"].update(
    dict(
        traj_swap=0.30,
        time_shift_future=0.34,
        perturb_heading=0.40,
        perturb_lateral=0.40,
        perturb_speed=0.45,
        reverse_traj=0.30,
    )
)

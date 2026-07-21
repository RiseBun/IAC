"""Path evidence plus visual progress alignment continuation.

This run keeps the already-positive path evidence certificate frozen and trains
only a visual progress head. The goal is not to relearn path evidence, but to
add a small image-driven progress prior that can be fused at evaluation time.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_path_evidence_head import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_path_evidence_progress_alignment"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_path_evidence_progress_alignment"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_progress_alignment_head"] = True

# Freeze the already-trained path evidence branch. Only the new visual progress
# head should learn in this continuation.
cfg["trainable_parameter_prefixes"] = [
    "progress_alignment_head",
]

# Keep the path evidence certificate intact; train only the progress prior.
cfg["lambda_consistency"] = 0.0
cfg["lambda_validity"] = 0.0
cfg["lambda_speed_consistency"] = 0.0
cfg["lambda_steering_consistency"] = 0.0
cfg["lambda_progress_consistency"] = 0.0
cfg["lambda_temporal_coherence"] = 0.0
cfg["lambda_group_ranking"] = 0.0
cfg["lambda_group_hard_negative"] = 0.0
cfg["lambda_future_consistency_evidence"] = 0.0
cfg["lambda_path_evidence_consistency"] = 0.0
cfg["lambda_path_grounding"] = 0.0
cfg["lambda_trajectory_specific_grounding"] = 0.0

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

cfg["checkpoint_metric"] = "val_loss"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 5.0e-5

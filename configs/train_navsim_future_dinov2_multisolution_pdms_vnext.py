"""Multi-solution image-trajectory consistency training.

This config removes the GT-top1 assumption from the objective. GT remains a
valid positive, but near candidates can also become positives when the
PDMS/EPDMS-style quality target says they are plausible for the same visual
future. Only image/time/trajectory swaps stay hard negatives.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_structured_rules_pdms_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_multisolution_pdms_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_multisolution_pdms_vnext"

# Core change: ranking is over consistency targets, not over the GT label.
cfg["group_ranking_target_mode"] = "soft"
cfg["group_ranking_min_target_gap"] = 0.06
cfg["lambda_group_ranking"] = 0.20

# Hard suppression is only for visually mismatched sources. Near perturbations
# are not hard negatives anymore; they are judged by their soft target.
cfg["group_hard_negative_target_mode"] = "soft"
cfg["group_hard_negative_sources"] = [
    "image_swap",
    "time_shift_future",
    "traj_swap",
    "reverse_traj",
]
cfg["group_hard_negative_max_target"] = 0.30
cfg["lambda_group_hard_negative"] = 0.12

# Auxiliary heads should learn the same multi-positive consistency surface.
cfg["auxiliary_consistency_target_mode"] = "soft"
cfg["consistency_positive_mask_mode"] = "soft"
cfg["soft_positive_target_threshold"] = 0.55

# Keep explicit motion/path grounding, but prevent those losses from becoming
# a disguised GT selector.
cfg["progress_alignment_target_mode"] = "soft"
cfg["progress_alignment_min_target_gap"] = 0.06
cfg["lambda_progress_alignment"] = 0.07
cfg["lambda_path_evidence_consistency"] = 0.26
cfg["lambda_history_counterfactual"] = 0.10
cfg["lambda_trajectory_specific_grounding"] = 0.20

cfg["lambda_motion_rule_attribute"] = 0.20
cfg["lambda_motion_rule_match"] = 0.14
cfg["lambda_motion_rule_rank"] = 0.16
cfg["motion_rule_rank_min_target_gap"] = 0.05

# The useful checkpoint is now the soft validation objective. Hard top1 is no
# longer the objective because there may be multiple acceptable trajectories.
cfg["checkpoint_metric"] = "val_loss"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 2.0e-5

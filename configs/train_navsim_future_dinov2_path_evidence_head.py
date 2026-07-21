"""Dual-head path evidence continuation.

This run keeps the inherited global consistency critic as the decision head
and trains a separate path_evidence_logit as the scientific certificate. The
certificate is evaluated on low-IoU counterfactual groups with
--consistency-score-key path_evidence_logit.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_pathgrounded_strong import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_path_evidence_head"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_path_evidence_head"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_path_conditioned_evidence"] = True
cfg["dinov2"]["use_path_residual_score"] = True
cfg["dinov2"]["path_residual_mix"] = 0.0
cfg["dinov2"]["use_path_evidence_head"] = True
cfg["dinov2"]["mix_path_evidence_into_consistency"] = False
cfg["dinov2"]["path_conditioned_forward_m"] = 40.0
cfg["dinov2"]["path_conditioned_lateral_m"] = 10.0
cfg["dinov2"]["path_conditioned_width"] = 0.10

# Freeze the global critic surface and update only the path evidence branch.
cfg["trainable_parameter_prefixes"] = [
    "path_conditioned_traj_proj",
    "path_conditioned_fusion",
    "path_evidence_head",
]

# Main decision losses are intentionally disabled: the decision score is
# inherited from fullgroup_strong, while this run teaches a separate evidence
# instrument to react to the exact candidate path.
cfg["lambda_consistency"] = 0.0
cfg["lambda_validity"] = 0.0
cfg["lambda_speed_consistency"] = 0.0
cfg["lambda_steering_consistency"] = 0.0
cfg["lambda_progress_consistency"] = 0.0
cfg["lambda_temporal_coherence"] = 0.0
cfg["lambda_group_ranking"] = 0.0
cfg["lambda_group_hard_negative"] = 0.0
cfg["lambda_future_consistency_evidence"] = 0.0

# The evidence head still learns the coarse positive/negative boundary, then
# receives stronger candidate-vs-wrong path pressure.
cfg["lambda_path_evidence_consistency"] = 0.50
cfg["path_grounding_score_key"] = "path_evidence_logit"
cfg["trajectory_specific_grounding_score_key"] = "path_evidence_logit"
cfg["lambda_path_grounding"] = 0.12
cfg["path_grounding_margin"] = 0.025
cfg["lambda_path_sky_contrast"] = 0.08
cfg["path_sky_contrast_margin"] = 0.03
cfg["lambda_trajectory_specific_grounding"] = 0.35
cfg["trajectory_specific_grounding_margin"] = 0.055
cfg["trajectory_specific_wrong_selection"] = "mask_iou"
cfg["trajectory_specific_grounding_exclusive"] = True

cfg["checkpoint_metric"] = "val_c_score_gap"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 1.0e-4

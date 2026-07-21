"""History-conditioned transition continuation for IAC.

This run targets a specific failure mode: candidates that look plausible from
future frames alone but are inconsistent with the historical state. It keeps
the fused path-evidence judge and adds a history counterfactual loss so the
score must drop when the history frames are swapped with another sample.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_path_evidence_fused_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_history_transition_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_history_transition_vnext"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["path_evidence_use_transition_context"] = True

# The counterfactual score is the actual fused decision surface, so this loss
# teaches the deployed judge that history matters rather than only teaching an
# auxiliary certificate.
cfg["lambda_history_counterfactual"] = 0.12
cfg["history_counterfactual_margin"] = 0.08
cfg["history_counterfactual_score_key"] = "consistency_logit"
cfg["history_counterfactual_positive_only"] = True
cfg["history_counterfactual_swap_ego"] = False

cfg["trainable_parameter_prefixes"] = [
    "path_conditioned_traj_proj",
    "path_conditioned_temporal_head",
    "path_conditioned_fusion",
    "path_residual_head",
    "path_evidence_head",
    "path_evidence_gate_head",
    "progress_alignment_head",
    "consistency_head",
]

# Slightly reduce generic ranking pressure so the new history loss is not
# drowned by the old GT-vs-perturb objective.
cfg["lambda_group_ranking"] = 0.24
cfg["lambda_group_hard_negative"] = 0.08
cfg["lambda_path_evidence_consistency"] = 0.30
cfg["lambda_progress_alignment"] = 0.10

cfg["checkpoint_metric"] = "val_iac_precision"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 3.5e-5

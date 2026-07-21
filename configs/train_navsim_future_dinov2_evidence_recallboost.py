"""Consistency-recall boosted variant of the NAVSIM future evidence config."""

from __future__ import annotations

from configs.train_navsim_future_dinov2_evidence import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_evidence_recallboost"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_evidence_recallboost"

# Keep the ranking objective as the main learning signal; do not over-push
# the pointwise BCE because it collapses calibration.
cfg["consistency_positive_weight"] = 3.0
cfg["consistency_class_balanced_loss"] = False
cfg["consistency_negative_loss_weight"] = 1.0

# Let the learned ranking / hierarchical heads carry the harder signal.
cfg["lambda_future_consistency_evidence"] = 0.25
cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["future_consistency_mix"] = 0.7

# Keep ranking useful, but let consistency get more gradient budget.
cfg["lambda_group_ranking"] = 0.45
cfg["lambda_group_hard_negative"] = 0.22
cfg["group_hard_negative_margin"] = 0.08
cfg["group_hard_negative_target"] = 0.24
cfg["lambda_hierarchical_consistency"] = 0.35
cfg["dinov2"]["use_hierarchical_consistency"] = True
cfg["dinov2"]["find_unused_parameters"] = True

# Select checkpoints by evaluator usefulness rather than raw BCE loss.
# This favors models that separate GT future/action pairs from counterfactual
# pairs, which is the core IAC benchmark objective.
cfg["checkpoint_metric"] = "val_iac_precision"

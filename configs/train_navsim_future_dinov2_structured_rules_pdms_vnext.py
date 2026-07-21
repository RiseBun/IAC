"""Structured rules with PDMS-style candidate-quality supervision.

The core consistency target is still image/history/future/trajectory matching.
PDMS/EPDMS-style scores only soften trajectory-near candidates that are drawn
from the same visual future. They must not lift image/time/traj swaps, because
those are visual-consistency negatives by construction.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_structured_rules_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_structured_rules_pdms_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_structured_rules_pdms_vnext"

cfg["train_index"] = "indices_navsim_future/consistency_train_pdms_proxy.jsonl"
cfg["val_index"] = "indices_navsim_future/consistency_val_pdms_proxy.jsonl"

cfg["candidate_quality_score_fields"] = [
    "epdms_score",
    "pdms_score",
    "planning_score",
    "epdms_proxy_score",
    "pdms_proxy_score",
    "candidate_quality_score",
]
cfg["candidate_quality_score_weight"] = 0.30
cfg["candidate_quality_allowed_sources"] = [
    "gt_pos",
    "perturb_speed",
    "perturb_lateral",
    "perturb_heading",
]

# PDMS-style soft labels already add positive mass to plausible near candidates.
# Keep the hard GT ranking objective present but slightly less dominant.
cfg["lambda_group_ranking"] = 0.26
cfg["lambda_group_hard_negative"] = 0.10
cfg["lambda_motion_rule_rank"] = 0.14
cfg["motion_rule_rank_min_target_gap"] = 0.06

cfg["checkpoint_metric"] = "val_iac_precision"
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 2.0e-5

"""Multi-solution IAC training with official NAVSIM PDM supervision."""

from __future__ import annotations

from configs.train_navsim_future_dinov2_multisolution_pdms_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_multisolution_official_pdms_vnext"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_multisolution_official_pdms_vnext"

cfg["train_index"] = "indices_navsim_future/consistency_train_official_pdms.jsonl"
cfg["val_index"] = "indices_navsim_future/consistency_val_official_pdms.jsonl"

cfg["candidate_quality_score_fields"] = [
    "official_epdms_score",
    "epdms_score",
    "official_pdm_score",
    "pdms_score",
    "planning_score",
    "candidate_quality_score",
]
cfg["candidate_quality_score_weight"] = 1.0
cfg["candidate_quality_target_mode"] = "override_non_gt_preserve_gt"
cfg["checkpoint_metric"] = "val_loss"

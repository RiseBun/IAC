"""Hierarchical consistency variant for NAVSIM future evidence."""

from __future__ import annotations

from configs.train_navsim_future_dinov2_evidence import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_hierarchical"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_hierarchical"
cfg["lambda_hierarchical_consistency"] = 0.35
cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"].update(
    use_hierarchical_consistency=True,
)

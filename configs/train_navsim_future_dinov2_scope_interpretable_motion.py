"""Opt-in SCOPE config with a named, additive motion evidence ledger."""

from __future__ import annotations

from configs.train_navsim_future_dinov2_scope_motion_head import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_scope_interpretable_motion"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_scope_interpretable_motion"

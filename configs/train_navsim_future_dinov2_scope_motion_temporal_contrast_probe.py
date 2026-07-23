"""Candidate-blind DINO motion probe with temporal-control contrast.

This config tests whether the image-side motion head can become sensitive to
future-frame order.  It keeps the base IAC scorer frozen and trains only the
scope motion head/comparator.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_scope_motion_probe_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_scope_motion_temporal_contrast_probe"
cfg["work_dir"] = (
    "work_dirs/iac_navsim_future_dinov2_scope_motion_temporal_contrast_probe"
)

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["scope_motion_temporal_controls"] = [
    "reverse_future",
    "roll_future",
    "shuffle_future",
    "zero_future",
]
cfg["dinov2"]["scope_motion_temporal_controls_train_only"] = True

cfg["scope_motion_temporal_controls"] = list(
    cfg["dinov2"]["scope_motion_temporal_controls"]
)
cfg["lambda_scope_motion_temporal_contrast"] = 0.35
cfg["scope_motion_temporal_contrast_margin"] = 0.18
cfg["scope_motion_temporal_contrast_min_target"] = 0.999

cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 5.0e-5

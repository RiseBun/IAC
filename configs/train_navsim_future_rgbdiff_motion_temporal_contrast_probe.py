"""RGB temporal-difference visual motion probe.

This experiment asks whether direct frame-difference evidence can provide the
dynamic image signal that frozen DINO frame tokens did not expose reliably.
The base IAC scorer remains frozen; only the RGB-diff motion head and its
trajectory comparator are trained.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_scope_motion_temporal_contrast_probe import (
    cfg as base,
)


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_rgbdiff_motion_temporal_contrast_probe"
cfg["work_dir"] = "work_dirs/iac_navsim_future_rgbdiff_motion_temporal_contrast_probe"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_scope_rgb_diff_motion_head"] = True
cfg["dinov2"]["scope_rgb_diff_spatial_size"] = 96

cfg["trainable_parameter_prefixes"] = [
    "scope_rgb_diff_motion_head",
    "scope_motion_comparator",
]

cfg["lambda_scope_motion_temporal_contrast"] = 0.45
cfg["scope_motion_temporal_contrast_margin"] = 0.20

cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 7.5e-5

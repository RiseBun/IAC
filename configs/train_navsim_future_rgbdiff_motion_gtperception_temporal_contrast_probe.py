"""RGB-diff perception-first motion probe.

The earlier RGB-diff probe proved that direct frame differences expose dynamic
visual evidence. This variant removes a supervision conflict: the visual
attribute regressor is trained only on gt_pos rows, so the same future image is
not asked to regress every candidate perturbation/mismatch trajectory. The
candidate comparator still learns hard visual-time mismatch signals.
"""

from __future__ import annotations

from configs.train_navsim_future_rgbdiff_motion_temporal_contrast_probe import (
    cfg as base,
)


cfg = dict(base)
cfg["experiment_name"] = (
    "iac_navsim_future_rgbdiff_motion_gtperception_temporal_contrast_probe"
)
cfg["work_dir"] = (
    "work_dirs/iac_navsim_future_rgbdiff_motion_gtperception_temporal_contrast_probe"
)

# Perception-first supervision: the image-only motion estimate should describe
# the actual future motion, not whichever candidate row happens to be sampled.
cfg["motion_rule_attribute_weight_mode"] = "gt_positive"
cfg["lambda_motion_rule_attribute"] = 0.60

# Keep mismatch/listwise pressure, but make it secondary to clean visual motion
# prediction.
cfg["lambda_motion_rule_match"] = 0.12
cfg["lambda_motion_rule_rank"] = 0.22
cfg["lambda_scope_motion_temporal_contrast"] = 0.55
cfg["scope_motion_temporal_contrast_margin"] = 0.24

cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 7.5e-5

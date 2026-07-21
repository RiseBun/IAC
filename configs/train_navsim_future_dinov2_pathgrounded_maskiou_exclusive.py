"""Mask-IoU hard wrong path grounding continuation.

The low-IoU diagnostic showed that the current model is path-grounded but not
yet exact-trajectory grounded. This config makes the trajectory-specific loss
use wrong paths whose projected image masks have the lowest IoU with the
positive candidate path, then applies the fair exclusive/equal-area objective.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_pathgrounded_strong import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_pathgrounded_maskiou_exclusive"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_pathgrounded_maskiou_exclusive"

cfg["trajectory_specific_wrong_selection"] = "mask_iou"
cfg["trajectory_specific_grounding_exclusive"] = True

# Push exact-path grounding harder, while keeping the generic path-vs-sky
# pressure active but not dominant.
cfg["lambda_trajectory_specific_grounding"] = 0.22
cfg["trajectory_specific_grounding_margin"] = 0.045
cfg["lambda_path_grounding"] = 0.08

# Retain false-positive control from the best full-group strong model.
cfg["lambda_group_ranking"] = 0.45
cfg["lambda_group_hard_negative"] = 0.24
cfg["group_hard_negative_margin"] = 0.08
cfg["group_hard_negative_target"] = 0.22


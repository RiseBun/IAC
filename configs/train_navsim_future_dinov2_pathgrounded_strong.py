"""Stronger trajectory-specific path grounding continuation.

This keeps the same path ROI protocol as the pathgrounded config, but raises
the pressure that positive rows must rely more on the candidate path than on a
same-group wrong trajectory path.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_pathgrounded import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_pathgrounded_strong"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_pathgrounded_strong"

cfg["lambda_trajectory_specific_grounding"] = 0.20
cfg["trajectory_specific_grounding_margin"] = 0.04

# Keep generic path grounding present, but make the candidate-vs-wrong path
# intervention the dominant new causal pressure in this continuation.
cfg["lambda_path_grounding"] = 0.10

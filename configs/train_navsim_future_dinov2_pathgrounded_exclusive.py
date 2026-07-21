"""Exclusive/equal-area trajectory-specific path grounding.

This trains the candidate-vs-wrong path intervention on non-overlapping,
equal-area path regions, matching the fair trajectory-specific causal metric.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_pathgrounded_strong import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_pathgrounded_exclusive"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_pathgrounded_exclusive"

cfg["trajectory_specific_grounding_exclusive"] = True

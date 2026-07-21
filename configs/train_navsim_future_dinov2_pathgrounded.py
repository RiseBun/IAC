"""Path-grounded continuation for NAVSIM future evidence.

The goal is to make consistency depend on future-image evidence around the
candidate driving path, not on trajectory geometry shortcuts alone.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_sourceaware import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_pathgrounded"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_pathgrounded"

# Causal training pressure:
# - path mask should reduce positive consistency scores;
# - area-matched sky/background mask should preserve them.
cfg["lambda_path_grounding"] = 0.12
cfg["path_grounding_margin"] = 0.025
cfg["path_grounding_sky_weight"] = 1.0
cfg["path_grounding_path_width"] = 0.10
cfg["path_grounding_sky_ratio"] = 0.25
cfg["path_grounding_positive_only"] = True
cfg["path_grounding_trajectory_mode"] = "positions"
cfg["path_grounding_projection_mode"] = "fixed"
cfg["path_grounding_forward_m"] = 40.0
cfg["path_grounding_lateral_m"] = 10.0

# Stronger causal pressure: for positive rows, masking the current candidate
# path should hurt more than masking a same-group wrong candidate path.
cfg["lambda_trajectory_specific_grounding"] = 0.08
cfg["trajectory_specific_grounding_margin"] = 0.01

# Keep selection aligned with the previous source-aware run, then use the
# path-causal benchmark as the deciding diagnostic.
cfg["checkpoint_metric"] = "val_iac_precision"

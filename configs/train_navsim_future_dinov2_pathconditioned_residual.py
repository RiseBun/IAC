"""Residual path-conditioned evidence continuation.

Unlike the first path-conditioned config, this does not concatenate path
evidence into the main fusion input. It keeps the inherited global critic
intact and adds a small residual path score:

    consistency = global_consistency + mix * path_residual

This preserves the old fusion weights while still giving the model a
candidate-path-specific visual evidence channel.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_pathconditioned import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_pathconditioned_residual"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_pathconditioned_residual"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_path_conditioned_evidence"] = True
cfg["dinov2"]["use_path_residual_score"] = True
cfg["dinov2"]["path_residual_mix"] = 0.35

# Because the main fusion head is preserved, we can use a slightly stronger
# exact-path pressure without resetting the global decision surface.
cfg["lambda_trajectory_specific_grounding"] = 0.20
cfg["trajectory_specific_grounding_margin"] = 0.045
cfg["lambda_path_grounding"] = 0.08


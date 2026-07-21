"""Path-conditioned visual evidence continuation.

This is the structural follow-up to the low-IoU failure: instead of only
regularizing masked scores, the model gets an explicit branch that pools DINOv2
future-image patch tokens along the candidate trajectory path and fuses that
path evidence with the trajectory embedding.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_pathgrounded_maskiou_exclusive import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_pathconditioned"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_pathconditioned"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_path_conditioned_evidence"] = True
cfg["dinov2"]["path_conditioned_forward_m"] = 40.0
cfg["dinov2"]["path_conditioned_lateral_m"] = 10.0
cfg["dinov2"]["path_conditioned_width"] = 0.10

# Keep the fair exact-path pressure active, but let the new branch carry the
# burden instead of further increasing scalar loss weights.
cfg["trajectory_specific_wrong_selection"] = "mask_iou"
cfg["trajectory_specific_grounding_exclusive"] = True
cfg["lambda_trajectory_specific_grounding"] = 0.18
cfg["trajectory_specific_grounding_margin"] = 0.04
cfg["lambda_path_grounding"] = 0.08


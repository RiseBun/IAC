"""DINOv2 IAC with visual-only progress alignment auxiliary loss.

This config keeps the v3.2 ambiguity-aware protocol and adds a small
image-progress head. The head sees only visual history/future features
(`z_hist`, `z_fut`, `z_fut - z_hist`), then the training loss ranks its
alignment error against candidate trajectory progress inside each group.

Purpose:
- strengthen image-driven time/progress evidence;
- avoid source-label shortcuts;
- avoid a trajectory-geometry shortcut inside the auxiliary head.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_ambiguity_aware import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_progress_alignment"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_progress_alignment"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"]["use_progress_alignment_head"] = True

# Keep the first run conservative. This loss should help hard visual/time
# negatives without overriding the already working ambiguity-aware objective.
cfg["lambda_progress_alignment"] = 0.15
cfg["progress_alignment_mode"] = "final_displacement"
cfg["progress_alignment_scale"] = 40.0
cfg["progress_alignment_hard_margin"] = 0.05
cfg["progress_alignment_near_margin"] = 0.005
cfg["progress_alignment_near_weight"] = 0.05
cfg["progress_alignment_hard_sources"] = [
    "image_swap",
    "time_shift_future",
    "traj_swap",
    "reverse_traj",
]
cfg["progress_alignment_near_sources"] = [
    "perturb_speed",
    "perturb_lateral",
    "perturb_heading",
]

# Fine-tune gently from the current best checkpoint first.
cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 3.0e-5
cfg["epochs"] = 1
cfg["save_interval"] = 1

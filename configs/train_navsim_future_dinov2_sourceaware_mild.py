"""Mild source-aware continuation for NAVSIM future evidence.

This tests the smallest useful intervention from the false-positive audit:
include `perturb_heading` and `perturb_lateral` in hard-negative sampling and
give geometry perturbations moderate source-specific pressure.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_evidence_recallboost import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_sourceaware_mild"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_sourceaware_mild"

cfg["lambda_group_hard_negative"] = 0.14
cfg["group_hard_negative_margin"] = 0.04
cfg["group_hard_negative_target"] = 0.30

cfg["consistency_source_weights"] = dict(cfg.get("consistency_source_weights", {}))
cfg["consistency_source_weights"].update(
    dict(
        image_swap=0.7,
        traj_swap=2.8,
        time_shift_future=3.2,
        perturb_heading=2.8,
        perturb_lateral=2.6,
        perturb_speed=4.2,
        reverse_traj=2.8,
    )
)

cfg["consistency_source_margins"] = dict(cfg.get("consistency_source_margins", {}))
cfg["consistency_source_margins"].update(
    dict(
        traj_swap=0.30,
        time_shift_future=0.34,
        perturb_heading=0.34,
        perturb_lateral=0.34,
        perturb_speed=0.40,
        reverse_traj=0.30,
    )
)

cfg["ranking"] = dict(cfg.get("ranking", {}))
cfg["ranking"].update(
    dict(
        max_negatives_per_group=6,
        hard_negative_sources=[
            "perturb_speed",
            "perturb_heading",
            "perturb_lateral",
            "time_shift_future",
            "traj_swap",
            "reverse_traj",
        ],
    )
)

cfg["checkpoint_metric"] = "val_iac_precision"

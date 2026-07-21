"""Source-aware hard-negative continuation for NAVSIM future evidence.

The current false positives are dominated by geometry perturbation sources
(`perturb_speed`, `perturb_heading`, `perturb_lateral`), while `image_swap`
is already mostly solved. This config keeps the successful recallboost line
but pushes ranking and hard-negative pressure toward those confused sources.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_evidence_recallboost import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_sourceaware"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_sourceaware"

# Keep the hard-negative objective present but not as blunt as the precision
# smoke; source weights/margins below do the targeting.
cfg["lambda_group_hard_negative"] = 0.18
cfg["group_hard_negative_margin"] = 0.06
cfg["group_hard_negative_target"] = 0.28

cfg["consistency_source_weights"] = dict(cfg.get("consistency_source_weights", {}))
cfg["consistency_source_weights"].update(
    dict(
        image_swap=0.6,
        traj_swap=2.5,
        time_shift_future=3.0,
        perturb_heading=3.5,
        perturb_lateral=3.5,
        perturb_speed=4.5,
        reverse_traj=2.5,
    )
)

cfg["consistency_source_margins"] = dict(cfg.get("consistency_source_margins", {}))
cfg["consistency_source_margins"].update(
    dict(
        traj_swap=0.28,
        time_shift_future=0.32,
        perturb_heading=0.38,
        perturb_lateral=0.38,
        perturb_speed=0.42,
        reverse_traj=0.28,
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

# Select for precision/TNR pressure, while threshold-sweep analysis still
# reports balanced accuracy and F1 for direct comparison.
cfg["checkpoint_metric"] = "val_iac_precision"

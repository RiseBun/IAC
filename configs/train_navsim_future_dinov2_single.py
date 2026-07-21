"""DINOv2 critic on the audited NAVSIM strict-future index.

This config is the production entry for the new NAVSIM project line:
history images and future images are disjoint, and future images come from
true future frames instead of history-tail replay.
"""

from __future__ import annotations

import os
from pathlib import Path

from configs.train_dinov2_v5_minimal import cfg as base


project_root = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: str | Path) -> str:
    return str(Path(os.environ.get(name, str(default))).expanduser())


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_single"
cfg["work_dir"] = str(project_root / "work_dirs" / "iac_navsim_future_dinov2_single")

# Strict NAVSIM index built with --future-image-policy future.
cfg["train_index"] = _env_path(
    "NAVSIM_FUTURE_TRAIN_INDEX",
    project_root / "indices_navsim_future" / "consistency_train.jsonl",
)
cfg["val_index"] = _env_path(
    "NAVSIM_FUTURE_VAL_INDEX",
    project_root / "indices_navsim_future" / "consistency_val.jsonl",
)
cfg["image_root"] = _env_path(
    "NAVSIM_IMAGE_ROOT",
    "/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_sensor_blobs",
)
cfg["camera_roots"] = [
    _env_path(
        "NAVSIM_CAMERA_ROOT",
        "/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_sensor_blobs/trainval",
    ),
]

# Kept for train.py compatibility; NAVSIM rows already carry image paths.
cfg["mini_db_root"] = _env_path(
    "NUPLAN_DB_ROOT",
    "/mnt/slurmfs-3090node3_msp/public_data/nuplan/dataset/nuplan-v1.1/trainval",
)

# Conservative defaults for one 4090-class GPU. Override from CLI/script for
# larger runs after the first score-distribution check.
cfg["epochs"] = 1
cfg["batch_size"] = 8
cfg["num_workers"] = 2
cfg["persistent_workers"] = True
cfg["prefetch_factor"] = 2
cfg["amp"] = True

# Strict-future consistency is the bottleneck. Keep validity useful but avoid
# letting the easy kinematic task dominate the early DINOv2 probe.
cfg["lambda_validity"] = 0.25
cfg["consistency_positive_weight"] = 3.0
cfg["consistency_class_balanced_loss"] = False
cfg["consistency_negative_loss_weight"] = 1.0
cfg["validity_negative_weight"] = 8.0
cfg["lambda_future_traj_geometry"] = 0.5

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"].update(
    enabled=True,
    model_name="dinov2_vits14",
    layer_index=11,
    freeze=True,
    use_explicit_distance=True,
    use_traj_geometry_features=True,
    use_future_traj_geometry_prediction=True,
    layer_mode="single",
    layer_indices=[11],
)

cfg["model"] = dict(cfg.get("model", {}))
cfg["model"].update(
    temporal_encoder="mean",
    use_action_visual_interaction=False,
)

cfg["ranking"] = dict(cfg.get("ranking", {}))
cfg["ranking"].update(
    enabled=True,
    group_batches=True,
    max_negatives_per_group=4,
    hard_negative_sources=[
        "perturb_speed",
        "time_shift_future",
        "traj_swap",
        "reverse_traj",
    ],
)
cfg["lambda_group_ranking"] = 0.35
cfg["group_ranking_margin"] = 0.2

cfg["difficulty_sampling"] = dict(cfg.get("difficulty_sampling", {}))
cfg["difficulty_sampling"].update(
    enabled=True,
    mix=(0.10, 0.15, 0.35, 0.40),
    positive_ratio=0.25,
    num_samples_per_epoch=0,
)

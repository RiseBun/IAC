from pathlib import Path

from configs.train_consistency_mini import cfg as base


project_root = Path(__file__).resolve().parent.parent

cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_stable"
cfg["work_dir"] = str(project_root / "work_dirs" / "iac_navsim_future_stable")

# Strict NAVSIM index built with --future-image-policy future.
cfg["train_index"] = str(project_root / "indices_navsim_future" / "consistency_train.jsonl")
cfg["val_index"] = str(project_root / "indices_navsim_future" / "consistency_val.jsonl")
cfg["image_root"] = "/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_sensor_blobs"
cfg["mini_db_root"] = "/mnt/slurmfs-3090node3_msp/public_data/nuplan/dataset/nuplan-v1.1/trainval"
cfg["camera_roots"] = [
    "/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_sensor_blobs/trainval",
]

# Conservative single-GPU baseline. This config is meant to validate the real
# future-frame data path before reintroducing ranking loss or DINOv2.
cfg["batch_size"] = 2
cfg["num_workers"] = 0
cfg["persistent_workers"] = False
cfg["prefetch_factor"] = 2

cfg["lambda_group_ranking"] = 0.0
cfg["ranking"] = dict(cfg.get("ranking", {}))
cfg["ranking"]["enabled"] = False
cfg["ranking"]["group_batches"] = False

cfg["difficulty_sampling"] = dict(cfg.get("difficulty_sampling", {}))
cfg["difficulty_sampling"]["enabled"] = False

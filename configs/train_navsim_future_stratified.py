from pathlib import Path

from configs.train_navsim_future_stable import cfg as base


cfg = dict(base)
project_root = Path(__file__).resolve().parent.parent
cfg["experiment_name"] = "iac_navsim_future_stratified"
cfg["work_dir"] = str(project_root / "work_dirs" / "iac_navsim_future_stratified")

# Fix the root cause exposed by the first two future-frame runs:
# default sampling sees one positive for six negatives, while a large global
# positive weight flips the model to all-positive. Instead, sample a balanced
# stream and keep the BCE weights neutral.
cfg["consistency_positive_weight"] = 1.0
cfg["lambda_consistency"] = 1.0
cfg["lambda_validity"] = 0.5
cfg["consistency_source_weights"] = {
    "gt_pos": 1.0,
    "image_swap": 1.0,
    "traj_swap": 1.0,
    "time_shift_future": 1.0,
    "perturb_lateral": 1.0,
    "perturb_heading": 1.0,
    "perturb_speed": 1.0,
}

cfg["difficulty_sampling"] = dict(cfg.get("difficulty_sampling", {}))
cfg["difficulty_sampling"]["enabled"] = True
cfg["difficulty_sampling"]["positive_ratio"] = 0.5
cfg["difficulty_sampling"]["mix"] = (0.25, 0.25, 0.25, 0.25)
cfg["difficulty_sampling"]["num_samples_per_epoch"] = 12000

cfg["lambda_group_ranking"] = 0.0
cfg["ranking"] = dict(cfg.get("ranking", {}))
cfg["ranking"]["enabled"] = False
cfg["ranking"]["group_batches"] = False

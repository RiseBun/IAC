from pathlib import Path

from configs.train_navsim_future_stable import cfg as base


cfg = dict(base)
project_root = Path(__file__).resolve().parent.parent
cfg["experiment_name"] = "iac_navsim_future_balanced"
cfg["work_dir"] = str(project_root / "work_dirs" / "iac_navsim_future_balanced")

# NAVSIM future index has one positive and six negatives per anchor. The base
# config also weights hard negatives, so the effective negative mass is about
# 12.5x one positive. Match that mass to avoid the consistency head collapsing
# to the class prior.
cfg["consistency_positive_weight"] = 12.5

# Keep the first balanced run simple: classification only, no ranking sampler.
cfg["lambda_group_ranking"] = 0.0
cfg["ranking"] = dict(cfg.get("ranking", {}))
cfg["ranking"]["enabled"] = False
cfg["ranking"]["group_batches"] = False

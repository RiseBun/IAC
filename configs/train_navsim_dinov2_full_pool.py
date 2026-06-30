from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

_base_path = Path(__file__).resolve().parent / "train_dinov2_v5_multilayer.py"
_spec = importlib.util.spec_from_file_location("_base", _base_path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

cfg = copy.deepcopy(_module.cfg)

cfg["experiment_name"] = "iac_navsim_dinov2_multilayer_full_pool"
cfg["work_dir"] = str(
    Path(__file__).resolve().parent.parent
    / "work_dirs"
    / "iac_navsim_dinov2_multilayer_full_pool"
)

cfg["epochs"] = 30
cfg["batch_size"] = 2
cfg["num_workers"] = 0
cfg["save_interval"] = 1
cfg["log_interval"] = 20

# 这是正式 epoch 预算，不是 debug cap：
# 每个 epoch 从 full NAVSIM train index 里抽 20000 samples。
cfg["difficulty_sampling"] = dict(cfg.get("difficulty_sampling", {}))
cfg["difficulty_sampling"]["enabled"] = True
cfg["difficulty_sampling"]["num_samples_per_epoch"] = 20000
cfg["difficulty_sampling"]["positive_ratio"] = 0.25

# 保留 DINOv2 multilayer + ranking 设定。
cfg["ranking"] = dict(cfg.get("ranking", {}))
cfg["ranking"]["enabled"] = True
cfg["ranking"]["group_batches"] = True
cfg["ranking"]["max_negatives_per_group"] = 3

"""v5 ablation: CNN backbone with D1-D4 sampler, no explicit distance."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


_base_path = Path(__file__).resolve().parent / "train_dinov2_v5_minimal.py"
_spec = importlib.util.spec_from_file_location("_v5_minimal_base", _base_path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

cfg = copy.deepcopy(_module.cfg)
cfg["experiment_name"] = "nuplan_iac_v5_d1d4"
cfg["work_dir"] = str(Path(__file__).resolve().parent.parent / "work_dirs" / "iac_v5_d1d4")
cfg["dinov2"]["enabled"] = False
cfg["dinov2"]["use_explicit_distance"] = False
cfg["difficulty_sampling"]["enabled"] = True

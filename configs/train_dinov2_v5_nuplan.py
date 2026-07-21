"""Explicit nuPlan entry for the DINOv2 v5 critic."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from configs.train_dinov2_v5_minimal import cfg as base


cfg = copy.deepcopy(base)
cfg["experiment_name"] = "nuplan_iac_dinov2_v5"
cfg["work_dir"] = str(project_root / "work_dirs" / "iac_dinov2_v5_nuplan")

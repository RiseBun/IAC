"""v5 ablation: DINOv2 multi-layer fusion.

This config keeps the v5 minimal training recipe unchanged except for the
DINOv2 image encoder:
  - single layer [11] -> multi-layer [6, 7, 8, 9, 10, 11]
  - learnable softmax fusion over layer features
  - no AvgPool / Ridge / geometric regularizer yet

Use this to test whether multi-layer DINO features improve IAC hard-negative
ranking before adding more complex layer-selection modules.
"""

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

cfg["experiment_name"] = "nuplan_iac_dinov2_v5_multilayer"
cfg["work_dir"] = str(
    Path(__file__).resolve().parent.parent
    / "work_dirs"
    / "iac_dinov2_v5_multilayer"
)

cfg["dinov2"]["layer_indices"] = [6, 7, 8, 9, 10, 11]
cfg["dinov2"].pop("layer_index", None)

# Keep the first multi-layer experiment intentionally simple and interpretable.
cfg["dinov2"]["fusion"] = "softmax"
cfg["dinov2"]["use_avgpool"] = False
cfg["dinov2"]["use_ridge_init"] = False

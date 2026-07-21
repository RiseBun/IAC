"""NAVSIM future DINOv2 with multi-layer / gated selection enabled."""

from __future__ import annotations

from configs.train_navsim_future_dinov2 import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_multilayer"
cfg["work_dir"] = str(
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "work_dirs" / "iac_navsim_future_dinov2_multilayer"
)

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"].update(
    layer_mode="gated",
    layer_indices=[3, 6, 9, 11],
    layer_gate_hidden=128,
    layer_residual_scale=1.0,
)


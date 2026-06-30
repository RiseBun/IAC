"""NAVSIM strict-future DINOv2 plus config.

This variant keeps the strict future-frame index, but upgrades the visual path
with multi-layer DINOv2 fusion and keeps the temporal/action interaction
features on.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2 import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_plus"
cfg["work_dir"] = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "work_dirs" / "iac_navsim_future_dinov2_plus")

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"].update(
    enabled=True,
    model_name="dinov2_vits14",
    layer_indices=[6, 7, 8, 9, 10, 11],
    freeze=True,
    use_explicit_distance=True,
    use_motion_features=True,
)
cfg["dinov2"].pop("layer_index", None)

cfg["model"] = dict(cfg.get("model", {}))
cfg["model"].update(
    temporal_encoder="gru",
    use_action_visual_interaction=True,
)

cfg["batch_size"] = 4
cfg["num_workers"] = 2
cfg["prefetch_factor"] = 2
cfg["epochs"] = 1
cfg["amp"] = True

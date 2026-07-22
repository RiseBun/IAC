"""IAC experiment: frozen DINO plus a candidate-blind temporal motion head.

The config keeps IAC's current structured/listwise training protocol.  The
visual head receives only per-frame DINO features; candidate attributes enter a
separate uncertainty-aware comparator.
"""

from __future__ import annotations

from configs.train_navsim_future_dinov2_structured_rules_vnext import cfg as base


cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_future_dinov2_scope_motion_head"
cfg["work_dir"] = "work_dirs/iac_navsim_future_dinov2_scope_motion_head"

cfg["dinov2"] = dict(cfg.get("dinov2", {}))
cfg["dinov2"].update(
    use_scope_motion_head=True,
    # Disable the old deterministic visual-rule MLP.  The wrapper still emits
    # the same IAC output keys, so existing losses remain available.
    use_learned_motion_rules=False,
    motion_rule_segment_count=3,
    motion_rule_attr_dim=36,
    motion_rule_mix=0.14,
    scope_motion_hidden_dim=256,
    scope_motion_num_layers=2,
    scope_motion_num_heads=4,
    scope_motion_dropout=0.10,
    scope_motion_max_frames=32,
)

# The candidate-blind visual target must come from exact/strong positives.  Do
# not train it toward every near-neighbour candidate in a support set.
cfg["lambda_motion_rule_attribute"] = 0.22
cfg["motion_rule_attribute_weight_mode"] = "threshold"
cfg["motion_rule_attribute_min_target"] = 0.999
cfg["lambda_motion_rule_match"] = 0.12
cfg["lambda_motion_rule_rank"] = 0.12
cfg["motion_rule_match_use_soft_targets"] = True
cfg["motion_rule_rank_margin"] = 0.16
cfg["motion_rule_rank_min_target_gap"] = 0.08

prefixes = list(cfg.get("trainable_parameter_prefixes", []))
for prefix in ("scope_motion_head", "scope_motion_comparator"):
    if prefix not in prefixes:
        prefixes.append(prefix)
cfg["trainable_parameter_prefixes"] = prefixes

cfg["optimizer"] = dict(cfg.get("optimizer", {}))
cfg["optimizer"]["lr"] = 2.5e-5

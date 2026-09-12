from __future__ import annotations

from tools.audit_metric_credibility import _direction_row


def test_direction_gate_uses_source_bootstrap_lower_bound(tmp_path):
    path = tmp_path / "mas_yaw_demo.json"
    path.write_text(
        '{"confirmation":{"normal":{"pair_coverage":0.95,"direction_accuracy":0.8,"source_bootstrap_ci95":[0.76,0.84]}}}',
        encoding="utf-8",
    )
    row = _direction_row(path, "MAS")
    assert row["credible_for_declared_yaw_scope"] is True

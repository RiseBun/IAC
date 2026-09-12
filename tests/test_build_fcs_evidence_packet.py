from __future__ import annotations

from tools.build_fcs_evidence_packet import build


def test_rollout_only_summary_cannot_be_called_fcs():
    report = build({
        "model_id": "drivewam",
        "coverage": 1.0,
        "task_success_rate": 0.51,
        "scored_rows": 978,
        "action_sources": ["drivewam_native_action_head"],
        "independent_realized_state": True,
        "action_injection_verified": True,
    })
    assert report["validation"]["status"] == "unavailable"
    assert report["packet"]["evidence_status"] == "unavailable"
    assert report["packet"]["failure_reason"] == "paired_future_only_intervention_missing"

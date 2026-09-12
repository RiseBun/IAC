from __future__ import annotations

from tools.build_gs_evidence_packets import build_packets
from iac_new.metric_evidence_contract import validate_metric_evidence_table


def _row(interval: int, generated: float, reference: float) -> dict:
    return {
        "source_key": "s1",
        "branch_role": "left",
        "interval_index": interval,
        "values": {
            "median_flow_magnitude_px": {"generated": generated, "logged_gt": reference},
            "horizontal_flow_center": {"generated": 0.1, "logged_gt": 0.1},
            "vertical_flow_center": {"generated": 0.2, "logged_gt": 0.2},
            "divergence": {"generated": 0.3, "logged_gt": 0.3},
            "curl": {"generated": 0.0, "logged_gt": 0.0},
        },
    }


def test_gs_builder_emits_valid_source_packet():
    rows = [_row(i, 10.0 + i, 10.0 + i) for i in range(4)]
    packets = build_packets(
        rows,
        reference_source="logged_navsim_future",
        generated_representation="generated_flow_manifest",
        reference_representation="logged_flow_manifest",
        scales={
            "median_flow_magnitude_px": 1.0,
            "horizontal_flow_center": 1.0,
            "vertical_flow_center": 1.0,
            "divergence": 1.0,
            "curl": 1.0,
        },
    )
    report = validate_metric_evidence_table(packets, "GS")
    assert report["valid"]
    assert report["status_counts"]["valid"] == 1


def test_gs_builder_preserves_missing_expected_branch_as_unavailable():
    rows = [_row(i, 10.0 + i, 10.0 + i) for i in range(4)]
    packets = build_packets(
        rows,
        reference_source="logged_navsim_future",
        generated_representation="generated_flow_structure_manifest",
        reference_representation="logged_future_flow_structure_manifest",
        scales={key: 1.0 for key in ("median_flow_magnitude_px", "horizontal_flow_center", "vertical_flow_center", "divergence", "curl")},
        expected_keys=["s1::left", "s2::right"],
    )
    report = validate_metric_evidence_table(packets, "GS")
    assert report["status_counts"]["valid"] == 1
    assert report["status_counts"]["unavailable"] == 1

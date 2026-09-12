from __future__ import annotations

from tools.build_metric_evidence_packets import build_mas, build_rcs
from iac_new.metric_evidence_contract import validate_metric_evidence_table


def _score(value: float = 0.4, coverage: float = 1.0) -> dict:
    return {
        "score": value,
        "interval_coverage": coverage,
        "fixed_support_fraction": 0.8,
        "projection_valid_fraction": 0.9,
    }


def test_mas_builder_requires_explicit_action_source_and_builds_valid_packets():
    rows = [{"source_key": "s1", "mas_left": _score(), "mas_right": _score(0.2)}]
    packets = build_mas(rows, model_id="Epona", action_source="native_action_head", threshold=0.75)
    report = validate_metric_evidence_table(packets, "MAS")
    assert report["valid"]
    assert report["status_counts"]["valid"] == 2


def test_rcs_is_unavailable_without_raw_action_delta():
    rows = [{"source_key": "s1", "normal": _score(), "reversed": _score(0.1), "zero_contrast": _score(0.2)}]
    packets = build_rcs(rows, model_id="Epona", action_source="native_action_head", records={}, threshold=0.75)
    report = validate_metric_evidence_table(packets, "RCS")
    assert report["valid"]
    assert report["status_counts"]["unavailable"] == 1


def test_rcs_packet_is_valid_with_explicit_paired_trajectory():
    rows = [{"source_key": "s1", "normal": _score(), "reversed": _score(0.1), "zero_contrast": _score(0.2)}]
    records = {
        ("s1", "left"): {"trajectory": [[1.0, 0.2, 0.1]]},
        ("s1", "right"): {"trajectory": [[0.5, -0.1, 0.0]]},
    }
    packets = build_rcs(rows, model_id="Epona", action_source="native_action_head", records=records, threshold=0.75)
    report = validate_metric_evidence_table(packets, "RCS")
    assert report["valid"]
    assert report["status_counts"]["valid"] == 1

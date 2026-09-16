import unittest

from tools.audit_longitudinal_real import audit


def _row(sample_id, intervals):
    return {
        "sample_id": sample_id,
        "intervals": intervals,
        "progress": {"action_interval_distance_m": [0.2, 2.0, 5.0, 8.0]},
    }


def _interval(predicted, predicted_bin, reference_bin, *, uncertain=False):
    return {
        "status": "uncertain" if uncertain else "scored",
        "distance_m_raw": predicted,
        "distance_bin": predicted_bin,
        "reference_distance_bin": reference_bin,
        "geometry_quality": {
            "reasons": ["weak_inlier_support"] if uncertain else [],
            "matches": 100,
            "inlier_fraction": 0.05 if uncertain else 0.5,
            "median_reprojection_error_px": 1.0,
        },
    }


class LongitudinalRealAuditTest(unittest.TestCase):
    def test_reports_coverage_and_ordinal_accuracy(self):
        rows = [_row("source-a::left", [
            _interval(0.1, "stop", "stop"),
            _interval(2.5, "very_short", "very_short"),
            _interval(3.5, "short", "short"),
            _interval(4.0, "short", "medium", uncertain=True),
        ])]
        result = audit(rows, draws=20)
        self.assertEqual(result["intervals_total"], 4)
        self.assertEqual(result["intervals_scored"], 3)
        self.assertEqual(result["coverage"], 0.75)
        self.assertEqual(result["exact_bin_accuracy"], 1.0)
        self.assertEqual(result["adjacent_bin_accuracy"], 1.0)
        self.assertEqual(result["abstention_reasons"]["weak_inlier_support"], 1)

    def test_adjacent_bin_is_distinct_from_exact_bin(self):
        rows = [_row("source-a::left", [
            _interval(0.8, "very_short", "stop"),
            _interval(3.5, "short", "very_short"),
            _interval(7.0, "medium", "short"),
            _interval(10.0, "long", "medium"),
        ])]
        result = audit(rows, draws=20)
        self.assertEqual(result["exact_bin_accuracy"], 0.0)
        self.assertEqual(result["adjacent_bin_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

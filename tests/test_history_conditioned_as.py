import unittest
import math

from iac_new.history_conditioned_as import aggregate, score_row, validate_benchmark_v3_record


def _record():
    return {
        "sample_id": "x",
        "history_frame_paths": ["h0", "h1", "h2", "h3"],
        "future_frame_paths": ["f0", "f1", "f2", "f3"],
        "history_times_s": [-1.5, -1.0, -0.5, 0.0],
        "future_times_s": [1.0, 2.0, 3.0, 4.0],
        "action_trajectory": [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        "camera_to_ego": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }


class HistoryConditionedASTest(unittest.TestCase):
    def test_fixed_contract(self):
        result = validate_benchmark_v3_record(_record())
        self.assertTrue(result["valid"])
        self.assertTrue(result["history_context_available"])

    def test_invalid_contract_abstains(self):
        record = _record()
        record["history_frame_paths"] = []
        result = score_row({}, record)
        self.assertEqual(result["status"], "uncertain")

    def test_missing_visual_intervals_are_explicit_abstentions(self):
        record = _record()
        record["action_trajectory"][-1][2] = 0.1
        row = score_row({"intervals": []}, record)
        self.assertFalse(row["visual_probe_contract"]["valid"])
        self.assertEqual(len(row["intervals"]), 4)
        self.assertTrue(row["yaw"]["applicable"])
        self.assertIsNone(row["yaw"]["direction_match"])
        result = aggregate([row], mode="wam")
        self.assertEqual(result["AS_progress_effective"], 0.0)
        self.assertEqual(result["AS_yaw_effective"], 0.0)
        self.assertEqual(result["AS_overall"], 0.0)

    def test_frozen_visual_threshold_is_consumed(self):
        record = _record()
        record["action_trajectory"][-1][2] = 0.1
        row = score_row(
            {"intervals": []},
            record,
            config={"yaw": {"straight_threshold_rad": 0.2}},
        )
        self.assertFalse(row["yaw"]["applicable"])

    def test_aggregate_has_separate_components(self):
        row = score_row({"intervals": []}, _record())
        result = aggregate([row], mode="real")
        self.assertIn("AS_progress_mean", result)
        self.assertIn("AS_yaw_direction", result)

    def test_outside_tolerance_remains_in_progress_denominator(self):
        def interval(distance):
            return {
                "reloc3r_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "estimators": {
                    "all": {
                        "match": {"selected_matches": 80},
                        "motion": {
                            "available": True,
                            "longitudinal_m": distance,
                            "inlier_fraction": 0.5,
                            "median_reprojection_error_px": 1.0,
                        },
                    }
                },
            }

        weak = interval(2.0)
        weak["estimators"]["all"]["match"]["selected_matches"] = 0
        weak["estimators"]["all"]["motion"]["inlier_fraction"] = 0.0
        row = score_row(
            {"intervals": [interval(2.0), interval(20.0), weak, weak]},
            _record(),
        )
        self.assertEqual(row["progress"]["quality_scored_count"], 2)
        self.assertEqual(row["progress"]["accepted_rate"], 0.5)
        result = aggregate([row], mode="real")
        self.assertEqual(result["AS_progress_mean"], 0.5)
        # The fixed contract expects four intervals.  One accepted interval
        # therefore contributes 1/4 to the coverage-aware progress score.
        self.assertEqual(result["accepted_interval_count"], 1)
        self.assertEqual(result["expected_interval_count"], 4)
        self.assertEqual(result["AS_progress_effective"], 0.25)

    def test_overall_as_penalizes_progress_abstention(self):
        def interval(distance, *, usable=True):
            return {
                "reloc3r_rotation": [
                    [math.cos(0.01), -math.sin(0.01), 0],
                    [math.sin(0.01), math.cos(0.01), 0],
                    [0, 0, 1],
                ],
                "estimators": {
                    "all": {
                        "match": {"selected_matches": 80 if usable else 0},
                        "motion": {
                            "available": True,
                            "longitudinal_m": distance,
                            "inlier_fraction": 0.5 if usable else 0.0,
                            "median_reprojection_error_px": 1.0,
                        },
                    }
                },
            }

        record = _record()
        record["action_trajectory"][-1][2] = 0.1
        row = score_row(
            {
                "intervals": [
                    interval(2.0),
                    interval(2.0),
                    interval(2.0, usable=False),
                    interval(2.0, usable=False),
                ]
            },
            record,
        )
        result = aggregate([row], mode="wam")
        self.assertEqual(result["AS_progress_mean"], 1.0)
        self.assertEqual(result["AS_progress_effective"], 0.5)
        self.assertEqual(result["AS_yaw_effective"], 1.0)
        self.assertAlmostEqual(result["AS_overall"], math.sqrt(0.5))
        self.assertAlmostEqual(result["AS_overall_100"], 100.0 * math.sqrt(0.5))


if __name__ == "__main__":
    unittest.main()

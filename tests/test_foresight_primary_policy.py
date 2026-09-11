import unittest

from iac_new.foresight_metrics import evaluate_cfac, evaluate_fau, evaluate_grounding_score


def _profile(*, yaw: float, lateral: float, source: str) -> dict:
    return {
        "source": source,
        "times_s": [1.0, 2.0],
        "rows": [
            {
                "time_s": 1.0,
                "yaw_rate_radps": yaw,
                "lateral_speed_mps": lateral,
                "curvature_1pm": 0.0,
                "shape_status": "usable",
                "observability": 1.0,
            },
            {
                "time_s": 2.0,
                "yaw_rate_radps": yaw,
                "lateral_speed_mps": lateral,
                "curvature_1pm": 0.0,
                "shape_status": "usable",
                "observability": 1.0,
            },
        ],
    }


class ForesightPrimaryPolicyTest(unittest.TestCase):
    def test_cfac_primary_score_uses_yaw_only(self) -> None:
        imagined = _profile(yaw=0.2, lateral=10.0, source="image_only_candidate_blind_decoder")
        action = _profile(yaw=0.2, lateral=-10.0, source="native_action_head")
        result = evaluate_cfac(imagined, action)
        self.assertEqual(result["primary_fields"], ["yaw_rate_radps"])
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertLess(result["components"]["lateral_speed_mps"]["direction_score"], 1.0)

    def test_fau_primary_score_uses_yaw_only(self) -> None:
        imagined = _profile(yaw=0.2, lateral=10.0, source="image_only_candidate_blind_decoder")
        action = _profile(yaw=0.2, lateral=-10.0, source="native_action_head")
        truth = _profile(yaw=0.2, lateral=0.0, source="ground_truth_future")
        result = evaluate_fau(imagined, action, truth)
        self.assertEqual(result["primary_fields"], ["yaw_rate_radps"])
        self.assertAlmostEqual(result["fau_f"], 1.0)
        self.assertAlmostEqual(result["fau_a"], 1.0)
        self.assertLess(
            result["components"]["lateral_speed_mps"]["fau_f_score"],
            1.0,
        )

    def test_canonical_metric_aliases_are_emitted(self) -> None:
        imagined = _profile(yaw=0.2, lateral=0.0, source="image_only_candidate_blind_decoder")
        action = _profile(yaw=0.2, lateral=0.0, source="native_action_head")
        truth = _profile(yaw=0.2, lateral=0.0, source="ground_truth_future")
        self.assertEqual(evaluate_cfac(imagined, action)["metric_id"], "MAS")
        grounding = evaluate_grounding_score(imagined, action, truth)
        self.assertEqual(grounding["metric_id"], "GS")
        self.assertEqual(grounding["legacy_metric"], "FAU")


if __name__ == "__main__":
    unittest.main()

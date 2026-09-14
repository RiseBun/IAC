import unittest

from tools.score_flow_token_mas_rcs import score_rcs


def branch(source, group, role, action_yaw, visual_yaw, *, available=True):
    return {
        "source_key": source,
        "sample_id": f"{group}::{role}",
        "counterfactual_group_id": group,
        "branch_role": role,
        "action_yaw": action_yaw,
        "action_extent_m": 5.0,
        "action_stop": False,
        "visual_yaw": visual_yaw,
        "yaw_available": available,
        "motion_features": {"median_q25": 1.0},
    }


class FlowTokenRcsTest(unittest.TestCase):
    def test_same_source_direction_response_and_reversed_control(self):
        rows = [
            branch("s0", "g0", "left", 0.2, 0.1),
            branch("s0", "g0", "right", -0.2, -0.1),
        ]
        result = score_rcs(
            rows, orientation=1, action_yaw_delta=0.01, visual_yaw_delta=0.001
        )
        direction = result["direction"]
        self.assertEqual(direction["scored"], 1)
        self.assertEqual(direction["score"], 1.0)
        self.assertEqual(direction["reversed_action_score"], 0.0)

    def test_unavailable_branch_reduces_coverage_without_zero_fill(self):
        rows = [
            branch("s0", "g0", "left", 0.2, 0.1, available=False),
            branch("s0", "g0", "right", -0.2, -0.1),
        ]
        result = score_rcs(
            rows, orientation=1, action_yaw_delta=0.01, visual_yaw_delta=0.001
        )
        direction = result["direction"]
        self.assertEqual(direction["scored"], 0)
        self.assertEqual(direction["measurement_pair_coverage"], 0.0)
        self.assertIsNone(direction["score"])

    def test_nonmaterial_action_pair_is_not_scored_as_failure(self):
        rows = [
            branch("s0", "g0", "left", 0.002, 0.1),
            branch("s0", "g0", "right", -0.002, -0.1),
        ]
        result = score_rcs(
            rows, orientation=1, action_yaw_delta=0.01, visual_yaw_delta=0.001
        )
        self.assertEqual(result["direction"]["scored"], 0)
        self.assertEqual(result["pair_exclusion_reasons"]["nonmaterial_action_yaw_delta"], 1)


if __name__ == "__main__":
    unittest.main()

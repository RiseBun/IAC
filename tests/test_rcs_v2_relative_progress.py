import unittest

from tools.score_rcs_relative_progress import score


CONFIG = {
    "protocol": "test-rcs-v2",
    "intervention_contract": {
        "same_source_within_group_required": True,
        "same_history_required": True,
        "same_nuisance_seed_required": True,
        "actual_regeneration_required": True,
        "action_injection_verified_required": True,
        "future_images_source": "epona_generated",
        "one_counterfactual_group_per_source_for_formal_report": True,
    },
    "relative_progress": {
        "minimum_action_delta_m": 0.5,
        "interval_indices": [1, 2, 3],
        "minimum_quality_intervals": 2,
        "minimum_inlier_fraction": 0.2,
    },
    "statistics": {
        "bootstrap_draws": 100,
        "bootstrap_seed": 7,
        "minimum_independent_sources_for_formal_report": 2,
        "promotion_criteria": {
            "coverage_min": 0.9,
            "endpoint_accuracy_min": 0.75,
            "endpoint_accuracy_ci95_lower_min": 0.0,
        },
    },
}


def manifest_group(group_id, source):
    common = {
        "counterfactual_group_id": group_id,
        "source_sample": source,
        "history_frame_paths": [f"{group_id}-history.png"],
        "history_fingerprint": f"{group_id}-history",
        "nuisance_seed": 1,
        "action_injection_verified": True,
        "future_images_source": "epona_generated",
    }
    return [
        {
            **common,
            "sample_id": f"{group_id}-low",
            "branch_role": "low",
            "future_frame_paths": [f"{group_id}-low.png"],
            "action_trajectory": [[0.0, 0.0]],
        },
        {
            **common,
            "sample_id": f"{group_id}-high",
            "branch_role": "high",
            "future_frame_paths": [f"{group_id}-high.png"],
            "action_trajectory": [[1.0, 0.0]],
        },
    ]


def visual_row(sample_id, value):
    interval = {
        "estimators": {
            "all": {"motion": {"longitudinal_m": value, "inlier_fraction": 0.5}}
        }
    }
    return {"sample_id": sample_id, "intervals": [{}, interval, interval, interval]}


class RelativeProgressIndependenceTest(unittest.TestCase):
    def test_repeated_source_is_pilot_even_with_enough_groups(self):
        manifest = manifest_group("g1", "same-log") + manifest_group("g2", "same-log")
        visual = [
            visual_row("g1-low", 0.0),
            visual_row("g1-high", 1.0),
            visual_row("g2-low", 0.0),
            visual_row("g2-high", 1.0),
        ]
        report = score(manifest, visual, CONFIG)
        self.assertEqual(report["independent_eligible_sources"], 1)
        self.assertFalse(report["independence_requirement_met"])
        self.assertEqual(report["status"], "pilot")
        self.assertFalse(report["promotion_pass"])

    def test_independent_sources_can_enter_formal_scoring(self):
        manifest = manifest_group("g1", "log-1") + manifest_group("g2", "log-2")
        visual = [
            visual_row("g1-low", 0.0),
            visual_row("g1-high", 1.0),
            visual_row("g2-low", 0.0),
            visual_row("g2-high", 1.0),
        ]
        report = score(manifest, visual, CONFIG)
        self.assertEqual(report["independent_eligible_sources"], 2)
        self.assertTrue(report["independence_requirement_met"])
        self.assertEqual(report["status"], "formal")
        self.assertTrue(report["promotion_pass"])


if __name__ == "__main__":
    unittest.main()

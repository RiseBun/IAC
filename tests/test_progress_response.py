from iac_new.progress_response import action_progress, score_progress_response


def test_action_progress_supports_both_manifest_shapes():
    assert action_progress({"action_trajectory": [[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]}) == 4.0
    assert action_progress({"predicted_action_trajectory": [[[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]}) == 1.0


def test_score_requires_visual_response_for_credit():
    manifest = [
        {"sample_id": "slow", "counterfactual_group_id": "g", "speed_role": "slow", "action_trajectory": [[1.0, 0, 0]]},
        {"sample_id": "fast", "counterfactual_group_id": "g", "speed_role": "fast", "action_trajectory": [[2.0, 0, 0]]},
    ]
    def row(sample_id, value):
        interval = {"estimators": {"all": {"motion": {"longitudinal_m": value, "inlier_fraction": 0.5}}}}
        return {"sample_id": sample_id, "intervals": [{}, interval, interval]}
    report = score_progress_response(manifest, [row("slow", 1.0), row("fast", 2.0)])
    assert report["scored_pairs"] == 1
    assert report["ordering_accuracy"] == 1.0

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_as_cross_model_comparability.py"
SPEC = importlib.util.spec_from_file_location("audit_as_cross_model_comparability", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(source: str, action_source: str = "wam_action_head") -> dict:
    return {
        "sample_id": source,
        "source_key": source,
        "history_frame_paths": ["h"] * 4,
        "future_frame_paths": ["f"] * 4,
        "history_times_s": [-1.5, -1.0, -0.5, 0.0],
        "future_times_s": [1.0, 2.0, 3.0, 4.0],
        "action_trajectory": [[0.0, 0.0, 0.0]] * 4,
        "action_trajectory_source": action_source,
        "future_images_source": "wam_generated",
        "wam_model_id": "model",
        "intrinsics": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "camera_to_ego": [[1.0, 0.0, 0.0, 0.0]] * 4,
        "metadata": {"candidate_blind_image_branch": True},
    }


def _report() -> dict:
    return {
        "protocol": "as-v1",
        "visual_protocol": "visual-v1",
        "aggregate": {"input_lineage_audit": {"status": "passed", "formal_metric_eligible": True}},
    }


class ComparabilityAuditTest(unittest.TestCase):
    def _write_case(self, root: Path, sources_a, sources_b, action_b="wam_action_head", selection_b="pool") -> Path:
        for label, sources, action in (("a", sources_a, "wam_action_head"), ("b", sources_b, action_b)):
            (root / f"{label}.jsonl").write_text(
                "".join(json.dumps(_row(source, action)) + "\n" for source in sources), encoding="utf-8"
            )
            (root / f"{label}_report.json").write_text(json.dumps(_report()), encoding="utf-8")
        config = {
            "comparison_unit": "source",
            "minimum_common_sources_for_formal_comparison": 1,
            "models": [
                {"label": "a", "manifest": "a.jsonl", "report": "a_report.json", "action_contract": "own_native", "selection_policy": "pool"},
                {"label": "b", "manifest": "b.jsonl", "report": "b_report.json", "action_contract": "own_native", "selection_policy": selection_b},
            ]
        }
        path = root / "configs" / "audit.json"
        path.parent.mkdir()
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_accepts_same_source_pool_and_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_case(root, ["s1", "s2"], ["s1", "s2"])
            result = MODULE.audit(config)
            self.assertTrue(result["strict_cross_model_leaderboard_eligible"])
            self.assertEqual(result["all_model_common_source_count"], 2)

    def test_rejects_non_native_or_different_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_case(root, ["s1", "s2"], ["s2", "s3"], action_b="injected_action")
            result = MODULE.audit(config)
            self.assertFalse(result["strict_cross_model_leaderboard_eligible"])
            self.assertFalse(result["models"][1]["gates"]["own_native_action"])
            self.assertFalse(result["joint_gates"]["same_declared_source_pool"])

    def test_rejects_pool_below_formal_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_case(root, ["s1"], ["s1"])
            payload = json.loads(config.read_text(encoding="utf-8"))
            payload["minimum_common_sources_for_formal_comparison"] = 30
            config.write_text(json.dumps(payload), encoding="utf-8")
            result = MODULE.audit(config)
            self.assertFalse(result["strict_cross_model_leaderboard_eligible"])
            self.assertFalse(result["joint_gates"]["minimum_common_sources"])

    def test_rejects_action_prefilter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_case(root, ["s1"], ["s1"], selection_b="action_materiality_prefilter")
            result = MODULE.audit(config)
            self.assertFalse(result["models"][1]["individually_eligible"])


if __name__ == "__main__":
    unittest.main()

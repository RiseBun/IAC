import json
import tempfile
import unittest
from pathlib import Path

from tools.analyse_joint_source_table import analyse


class JointSourceTableTest(unittest.TestCase):
    def test_join_is_source_level_and_missing_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "as.json").write_text(json.dumps({"rows": [
                {"sample_id": "s1::left", "status": "scored", "as_components": {"composite_mean": 0.8}, "yaw": {"applicable": True, "direction_match": True}, "intervals": [{"status": "scored", "acceptable": True}]},
                {"sample_id": "s1::right", "status": "scored", "as_components": {"composite_mean": 0.6}, "yaw": {"applicable": True, "direction_match": True}, "intervals": [{"status": "scored", "acceptable": True}]},
            ]}), encoding="utf-8")
            (root / "execution.jsonl").write_text(json.dumps({"source_key": "s1", "task_success": True, "task_score": 0.9}) + "\n" + json.dumps({"source_key": "s2", "task_success": False, "task_score": 0.1}) + "\n", encoding="utf-8")
            report = analyse(root / "as.json", root / "execution.jsonl")
            self.assertEqual(report["joined_source_count"], 1)
            self.assertEqual(report["correlations"][0]["n_sources"], 1)
            self.assertIsNone(report["correlations"][0]["spearman"])
            self.assertIsNone(report["correlations"][0]["log_cluster_bootstrap_ci95"])
            s2 = next(row for row in report["table"] if row["source_key"] == "s2")
            self.assertNotIn("as_score", s2)


if __name__ == "__main__":
    unittest.main()

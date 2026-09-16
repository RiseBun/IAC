import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_action_alignment import audit


class ActionAlignmentTest(unittest.TestCase):
    def test_mismatch_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"source_key": "s", "branch_role": "left", "action_trajectory": [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]}
            rollout = {"source_key": "s", "action_trajectory": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], "future_times_s": [1, 2, 3, 4]}
            (root / "m.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            (root / "r.jsonl").write_text(json.dumps(rollout) + "\n", encoding="utf-8")
            report = audit(root / "m.jsonl", root / "r.jsonl")
            self.assertEqual(report["status"], "failed_action_mismatch")
            self.assertEqual(report["exact_matches"], 0)


if __name__ == "__main__":
    unittest.main()

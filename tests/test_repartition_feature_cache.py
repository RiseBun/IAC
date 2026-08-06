from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "repair") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "repair"))

from repartition_feature_cache import repartition_cache  # noqa: E402


class FeatureCacheRepartitionTest(unittest.TestCase):
    def test_rows_and_tensors_follow_requested_sample_order(self) -> None:
        first = {
            "sample_id": ["a", "b"],
            "source_type": ["gt", "swap"],
            "x": torch.tensor([[1.0], [2.0]]),
            "metadata": {"rows": 2, "feature": "x"},
        }
        second = {
            "sample_id": ["c", "d"],
            "source_type": ["gt", "swap"],
            "x": torch.tensor([[3.0], [4.0]]),
            "metadata": {"rows": 2, "feature": "x"},
        }
        result = repartition_cache([first, second], ["d", "a", "c"])
        self.assertEqual(result["sample_id"], ["d", "a", "c"])
        self.assertEqual(result["source_type"], ["swap", "gt", "gt"])
        self.assertTrue(torch.equal(result["x"], torch.tensor([[4.0], [1.0], [3.0]])))
        self.assertEqual(result["metadata"]["rows"], 3)
        self.assertTrue(result["metadata"]["repartitioned_without_feature_reextraction"])


if __name__ == "__main__":
    unittest.main()

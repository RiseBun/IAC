from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "audit") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "audit"))

from audit_ordered_motion_support import audit_rows  # noqa: E402


class OrderedMotionSupportAuditTest(unittest.TestCase):
    def test_precision_uses_only_precomputed_decisions(self) -> None:
        rows = [
            {"source_type": "gt_pos", "ordered_motion_support_state": "supported"},
            {"source_type": "image_swap", "ordered_motion_support_state": "supported"},
            {"source_type": "traj_swap", "ordered_motion_support_state": "unsupported"},
            {"source_type": "gt_pos", "ordered_motion_support_state": "insufficient_evidence"},
        ]
        result = audit_rows(
            rows,
            acceptable_sources={"gt_pos"},
            hard_sources={"image_swap", "traj_swap"},
        )
        self.assertEqual(result["supported_precision"], 0.5)
        self.assertEqual(result["unsupported_precision"], 1.0)
        self.assertEqual(result["per_source"]["gt_pos"]["rows"], 2)


if __name__ == "__main__":
    unittest.main()

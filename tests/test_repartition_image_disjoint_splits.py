from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "repair") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "repair"))

from repartition_image_disjoint_splits import repartition  # noqa: E402


def _row(group: str, source: str, image: str) -> dict:
    return {
        "sample_id": f"{group}__{source}",
        "source_type": source,
        "scene_name": group,
        "future_images": [image],
    }


class ImageDisjointRepartitionTest(unittest.TestCase):
    def test_shared_image_components_move_atomically(self) -> None:
        left = [_row("a", "gt_pos", "shared.png"), _row("b", "gt_pos", "b.png")]
        right = [_row("c", "gt_pos", "shared.png"), _row("d", "gt_pos", "d.png")]
        output_left, output_right, summary = repartition(left, right)
        left_groups = {row["sample_id"].split("__", 1)[0] for row in output_left}
        right_groups = {row["sample_id"].split("__", 1)[0] for row in output_right}
        self.assertEqual(len(left_groups), 2)
        self.assertEqual(len(right_groups), 2)
        self.assertTrue({"a", "c"} <= left_groups or {"a", "c"} <= right_groups)
        self.assertEqual(summary["moved_groups"], 2)
        self.assertEqual(summary["post_repartition_image_overlap"], 0)


if __name__ == "__main__":
    unittest.main()

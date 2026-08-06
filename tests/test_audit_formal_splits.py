from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (
    PROJECT_ROOT,
    PROJECT_ROOT / "pipeline",
    PROJECT_ROOT / "training",
    PROJECT_ROOT / "audit",
    PROJECT_ROOT / "repair",
    PROJECT_ROOT / "ordered_motion",
):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from audit_formal_splits import audit_splits  # noqa: E402


SOURCES = (
    "gt_pos",
    "image_swap",
    "perturb_heading",
    "perturb_lateral",
    "perturb_speed",
    "time_shift_future",
    "traj_swap",
)


def _rows(prefix: str, scene: str, image_prefix: str) -> list[dict]:
    result = []
    for index, source in enumerate(SOURCES):
        result.append(
            {
                "group_id": f"{prefix}_group",
                "sample_id": f"{prefix}_{index}",
                "source_type": source,
                "scene_id": scene,
                "history_images": [f"{image_prefix}_history.png"],
                "future_images": [f"{image_prefix}_future_{step}.png" for step in range(8)],
                "candidate_traj": [[float(step), 0.0, 0.0] for step in range(8)],
                "horizon_seconds": 4.0,
            }
        )
    return result


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class FormalSplitAuditTest(unittest.TestCase):
    def test_group_id_can_be_derived_from_stable_sample_id(self) -> None:
        rows = _rows("train", "scene_train", "train")
        for row in rows:
            row.pop("group_id")
            row["sample_id"] = f"train_group__{row['source_type']}"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            _write(train, rows)
            _write(val, _rows("val", "scene_val", "val"))
            result = audit_splits(
                [("train", train), ("val", val)],
                horizon="4s",
            )
        self.assertTrue(result["formal_evidence_ready"])
        self.assertEqual(result["splits"]["train"]["groups"], 1)
        self.assertEqual(
            result["splits"]["train"]["schema"]["derived_group_id_rows"],
            len(rows),
        )

    def test_disjoint_complete_splits_are_formal_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, evaluation = root / "train.jsonl", root / "eval.jsonl"
            _write(train, _rows("train", "scene_train", "train"))
            _write(evaluation, _rows("eval", "scene_eval", "eval"))
            summary = audit_splits(
                [("train", train), ("evaluation", evaluation)],
                horizon="4s",
            )
            self.assertTrue(summary["formal_evidence_ready"])

    def test_missing_scene_identity_is_not_treated_as_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, evaluation = root / "train.jsonl", root / "eval.jsonl"
            train_rows = _rows("train", "scene_train", "train")
            for row in train_rows:
                row.pop("scene_id")
            _write(train, train_rows)
            _write(evaluation, _rows("eval", "scene_eval", "eval"))
            summary = audit_splits(
                [("train", train), ("evaluation", evaluation)],
                horizon="4s",
            )
            self.assertFalse(summary["formal_evidence_ready"])
            self.assertEqual(
                summary["splits"]["train"]["schema"]["missing_scene_id_rows"],
                len(SOURCES),
            )

    def test_scene_overlap_and_incomplete_groups_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, evaluation = root / "train.jsonl", root / "eval.jsonl"
            _write(train, _rows("train", "shared_scene", "train"))
            evaluation_rows = _rows("eval", "shared_scene", "eval")[:-1]
            _write(evaluation, evaluation_rows)
            summary = audit_splits(
                [("train", train), ("evaluation", evaluation)],
                horizon="4s",
            )
            self.assertFalse(summary["formal_evidence_ready"])
            self.assertEqual(
                summary["pairs"]["train__vs__evaluation"]["scene_id_overlap"],
                1,
            )
            self.assertEqual(
                summary["splits"]["evaluation"]["group_protocol"]["incomplete_groups"],
                1,
            )


if __name__ == "__main__":
    unittest.main()

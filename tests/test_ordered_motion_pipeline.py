from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
for value in (PROJECT_ROOT, TOOLS_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from audit_ordered_motion_alignment import audit  # noqa: E402
from train_ordered_motion_alignment import train  # noqa: E402


class PortablePipelineTest(unittest.TestCase):
    @staticmethod
    def _rows_and_cache(prefix: str, group_count: int) -> tuple[list[dict], dict]:
        rows: list[dict] = []
        sample_ids: list[str] = []
        features: list[torch.Tensor] = []
        for group_index in range(group_count):
            scale = 0.8 + 0.1 * group_index
            gt_traj = [
                [scale * (step + 1), 0.05 * step, 0.01 * step]
                for step in range(4)
            ]
            speed_traj = [
                [1.5 * scale * (step + 1), 0.05 * step, 0.01 * step]
                for step in range(4)
            ]
            time = torch.linspace(0.0, 1.0, 16).unsqueeze(-1)
            visual = torch.cat(
                [
                    scale * time,
                    torch.sin(time * 3.14159),
                    torch.cos(time * 3.14159),
                    time.square(),
                    time.repeat(1, 4),
                ],
                dim=-1,
            )[:, :8]
            for source_name, trajectory in (
                ("gt_pos", gt_traj),
                ("perturb_speed", speed_traj),
            ):
                sample_id = f"{prefix}_g{group_index}_{source_name}"
                rows.append(
                    {
                        "group_id": f"{prefix}_g{group_index}",
                        "sample_id": sample_id,
                        "source_type": source_name,
                        "candidate_traj": trajectory,
                    }
                )
                sample_ids.append(sample_id)
                features.append(visual)
        return rows, {
            "sample_id": sample_ids,
            "x_tokens": torch.stack(features),
            "metadata": {
                "kind": "synthetic_ordered_vjepa_cache",
                "token_summary_size": 16,
            },
        }

    def test_train_score_and_control_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, count in (("train", 8), ("val", 4), ("eval", 4)):
                rows, cache = self._rows_and_cache(name, count)
                (root / f"{name}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                torch.save(cache, root / f"{name}.pt")

            model_path = root / "ordered.pt"
            train(
                argparse.Namespace(
                    train_rows=str(root / "train.jsonl"),
                    train_cache=str(root / "train.pt"),
                    val_rows=str(root / "val.jsonl"),
                    val_cache=str(root / "val.pt"),
                    output_model=str(model_path),
                    output_summary=str(root / "train_summary.json"),
                    feature_key="x_tokens",
                    positive_selector="gt_only",
                    segment_count=4,
                    hidden_dim=16,
                    bandwidth=0.24,
                    dropout=0.0,
                    epochs=3,
                    patience=2,
                    min_delta=1e-5,
                    batch_size=4,
                    num_workers=0,
                    lr=1e-3,
                    weight_decay=0.0,
                    grad_clip=5.0,
                    seed=19,
                    max_train_rows=0,
                    max_val_rows=0,
                    device="cpu",
                )
            )
            summary = audit(
                argparse.Namespace(
                    model=str(model_path),
                    rows=str(root / "eval.jsonl"),
                    visual_cache=str(root / "eval.pt"),
                    output_summary=str(root / "audit.json"),
                    output_ledger=str(root / "ledger.jsonl"),
                    feature_key="x_tokens",
                    batch_size=8,
                    device="cpu",
                    seed=19,
                    acceptable_sources=(
                        "gt_pos,perturb_speed,perturb_lateral,perturb_heading"
                    ),
                    hard_sources="image_swap,time_shift_future,traj_swap",
                )
            )
            self.assertTrue(model_path.exists())
            self.assertEqual(summary["protocol"]["rows"], 8)
            self.assertGreater(
                summary["control_score_delta_from_normal"][
                    "reverse_compressed_visual_time"
                ]["mean_absolute_score_delta"],
                1e-6,
            )
            self.assertTrue((root / "ledger.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

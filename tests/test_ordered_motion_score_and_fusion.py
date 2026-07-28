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

from iac_extensions.ordered_motion_alignment import (  # noqa: E402
    OrderedMotionAlignment,
    OrderedMotionConfig,
    save_bundle,
)
from make_temporal_control_rows import transform  # noqa: E402
from score_ordered_motion_alignment import score  # noqa: E402
from tune_fuse_ordered_motion import tune_and_apply  # noqa: E402


class ScoreAndFusionSmokeTest(unittest.TestCase):
    def test_score_raw_control_and_validation_tuned_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = OrderedMotionAlignment(
                OrderedMotionConfig(
                    visual_dim=6,
                    hidden_dim=12,
                    segment_count=3,
                    dropout=0.0,
                )
            )
            model_path = root / "model.pt"
            save_bundle(
                model_path,
                model=model,
                target_mean=torch.zeros(1, 1, 4),
                target_std=torch.ones(1, 1, 4),
                metadata={"test": True},
            )

            rows = []
            sample_ids = []
            features = []
            base_rows = []
            for group_index in range(2):
                visual = torch.randn(16, 6)
                for source_index, source in enumerate(("gt_pos", "image_swap")):
                    sample_id = f"g{group_index}_{source}"
                    row = {
                        "group_id": f"g{group_index}",
                        "sample_id": sample_id,
                        "source_type": source,
                        "candidate_traj": [
                            [1.0 + source_index, 0.0, 0.0],
                            [2.0 + source_index, 0.1, 0.05],
                            [3.0 + source_index, 0.2, 0.10],
                        ],
                        "history_images": ["h0.jpg", "h1.jpg"],
                        "future_images": ["f0.jpg", "f1.jpg"],
                    }
                    rows.append(row)
                    sample_ids.append(sample_id)
                    features.append(visual)
                    base_rows.append({**row, "base_score": 0.6 - 0.1 * source_index})

            rows_path = root / "rows.jsonl"
            rows_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            base_path = root / "base.jsonl"
            base_path.write_text(
                "".join(json.dumps(row) + "\n" for row in base_rows),
                encoding="utf-8",
            )
            cache_path = root / "cache.pt"
            torch.save(
                {
                    "sample_id": sample_ids,
                    "x_tokens": torch.stack(features),
                    "metadata": {"token_summary_size": 16},
                },
                cache_path,
            )
            scores_path = root / "scores.jsonl"
            summary = score(
                argparse.Namespace(
                    model=str(model_path),
                    rows=str(rows_path),
                    visual_cache=str(cache_path),
                    output_scores=str(scores_path),
                    output_summary=str(root / "score_summary.json"),
                    feature_key="x_tokens",
                    batch_size=4,
                    device="cpu",
                    include_segment_ledger=True,
                    acceptable_sources="gt_pos",
                    hard_sources="image_swap",
                )
            )
            self.assertEqual(summary["matched_rows"], 4)
            first_scored = json.loads(scores_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("ordered_motion_segment_ledger", first_scored)

            transform(
                argparse.Namespace(
                    input_rows=str(rows_path),
                    output_rows=str(root / "reverse_rows.jsonl"),
                    output_summary=str(root / "reverse_summary.json"),
                    control="reverse",
                    seed=13,
                )
            )
            reversed_first = json.loads(
                (root / "reverse_rows.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(reversed_first["history_images"], ["f1.jpg", "f0.jpg"])
            self.assertEqual(reversed_first["future_images"], ["h1.jpg", "h0.jpg"])

            fusion = tune_and_apply(
                argparse.Namespace(
                    val_primary=str(base_path),
                    val_evidence=str(scores_path),
                    eval_primary=str(base_path),
                    eval_evidence=str(scores_path),
                    primary_key="base_score",
                    energy_key="ordered_motion_energy",
                    beta_grid="0,0.1",
                    threshold_grid="0,0.5",
                    hard_weight=1.0,
                    strict_weight=0.05,
                    acceptable_sources="gt_pos",
                    hard_sources="image_swap",
                    output_scores=str(root / "fused.jsonl"),
                    output_summary=str(root / "fusion.json"),
                )
            )
            self.assertIn("selected", fusion)
            self.assertTrue((root / "fused.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

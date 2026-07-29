from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
for value in (PROJECT_ROOT, TOOLS_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from audit_fused_ordered_motion_controls import audit as audit_fused  # noqa: E402
from audit_ordered_motion_splits import audit as audit_splits  # noqa: E402
from summarize_ordered_motion_decision import summarize  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class OrderedMotionDecisionAuditTest(unittest.TestCase):
    def test_split_audit_detects_scene_and_image_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [
                {
                    "sample_id": "train_gt",
                    "group_id": "train",
                    "source_type": "gt_pos",
                    "scene_name": "scene_a",
                    "history_images": ["shared.jpg"],
                    "future_images": ["train.jpg"],
                }
            ]
            val = [
                {
                    "sample_id": "val_gt",
                    "group_id": "val",
                    "source_type": "gt_pos",
                    "scene_name": "scene_b",
                    "history_images": ["val.jpg"],
                    "future_images": [],
                }
            ]
            evaluation = [
                {
                    "sample_id": "eval_gt",
                    "group_id": "eval",
                    "source_type": "gt_pos",
                    "scene_name": "scene_a",
                    "history_images": ["shared.jpg"],
                    "future_images": [],
                }
            ]
            for name, rows in (
                ("train", train),
                ("val", val),
                ("eval", evaluation),
            ):
                _write_jsonl(root / f"{name}.jsonl", rows)
            result = audit_splits(
                argparse.Namespace(
                    split=[
                        f"train={root / 'train.jsonl'}",
                        f"validation={root / 'val.jsonl'}",
                        f"evaluation={root / 'eval.jsonl'}",
                    ],
                    output_summary=str(root / "split.json"),
                    require_strict_disjoint=False,
                )
            )
            pair = result["pairs"]["train__vs__evaluation"]
            self.assertEqual(pair["scene_overlap"], 1)
            self.assertEqual(pair["image_path_overlap"], 1)
            self.assertFalse(
                result["all_pairs_strict_scene_and_image_disjoint"]
            )

    def test_fused_controls_use_one_fixed_validation_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary: list[dict] = []
            ledger: list[dict] = []
            for group in ("g0", "g1"):
                for source, base, normal_rank, reverse_rank in (
                    ("gt_pos", 0.6, -0.1, -1.0),
                    ("image_swap", 0.7, -10.0, -0.1),
                ):
                    sample_id = f"{group}_{source}"
                    primary.append(
                        {
                            "sample_id": sample_id,
                            "group_id": group,
                            "source_type": source,
                            "base": base,
                        }
                    )
                    ledger.append(
                        {
                            "sample_id": sample_id,
                            "group_id": group,
                            "normal_ordered_motion_rank_score": normal_rank,
                            "control_rank_scores": {
                                "reverse_compressed_visual_time": reverse_rank,
                                "permute_compressed_visual_time": reverse_rank,
                                "reverse_trajectory_segments": reverse_rank,
                                "permute_trajectory_segments": reverse_rank,
                                "candidate_derangement": reverse_rank,
                                "visual_group_derangement": reverse_rank,
                            },
                        }
                    )
            _write_jsonl(root / "primary.jsonl", primary)
            _write_jsonl(root / "ledger.jsonl", ledger)
            (root / "fusion.json").write_text(
                json.dumps(
                    {
                        "selected": {
                            "beta": 0.2,
                            "threshold": 0.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = audit_fused(
                argparse.Namespace(
                    primary_rows=str(root / "primary.jsonl"),
                    primary_key="base",
                    control_ledger=str(root / "ledger.jsonl"),
                    fusion_summary=str(root / "fusion.json"),
                    output_summary=str(root / "audit.json"),
                    acceptable_sources="gt_pos",
                    hard_sources="image_swap",
                )
            )
            self.assertEqual(
                result["metrics"]["normal"]["strict_gt_top1"],
                1.0,
            )
            self.assertTrue(
                result["decision_diagnostics"][
                    "normal_beats_every_order_control_mrr"
                ]
            )

    def test_multi_seed_summary_separates_engineering_and_formal_gates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_args: list[str] = []
            for seed in (1, 2, 3):
                run = root / f"seed_{seed}"
                run.mkdir()
                (run / "fusion_summary.json").write_text(
                    json.dumps(
                        {
                            "eval_base_metrics": {
                                "strict_gt_top1": 0.4,
                                "acceptable_top1": 0.9,
                                "hard_mismatch_top1": 0.1,
                                "mrr_gt": 0.6,
                                "pairwise_gt_win": {
                                    "perturb_speed": 0.6,
                                    "time_shift_future": 0.7,
                                },
                            },
                            "eval_fused_metrics": {
                                "strict_gt_top1": 0.42,
                                "acceptable_top1": 0.91,
                                "hard_mismatch_top1": 0.09,
                                "mrr_gt": 0.62,
                                "pairwise_gt_win": {
                                    "perturb_speed": 0.61,
                                    "time_shift_future": 0.71,
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "fused_control_audit.json").write_text(
                    json.dumps(
                        {
                            "metrics": {
                                "normal": {
                                    "strict_gt_top1": 0.42,
                                    "acceptable_top1": 0.91,
                                    "mrr_gt": 0.62,
                                }
                            },
                            "decision_diagnostics": {
                                "normal_minus_best_order_control_mrr": 0.01,
                                "normal_beats_every_order_control_mrr": True,
                                "normal_beats_every_identity_control_mrr": True,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "split_independence_audit.json").write_text(
                    json.dumps(
                        {
                            "all_pairs_strict_scene_and_image_disjoint": False
                        }
                    ),
                    encoding="utf-8",
                )
                run_args.append(f"seed_{seed}={run}")
            result = summarize(
                argparse.Namespace(
                    run=run_args,
                    output_summary=str(root / "decision.json"),
                )
            )
            self.assertTrue(
                result["preregistered_gates"][
                    "advance_corrected_time_head"
                ]
            )
            self.assertFalse(
                result["interpretation"]["formal_evidence_ready"]
            )


if __name__ == "__main__":
    unittest.main()

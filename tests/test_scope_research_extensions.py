from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from iac_extensions.dino_motion_head import (
    CandidateBlindDinoMotionHead,
    UncertaintyAwareTrajectoryComparator,
    uncertainty_weighted_motion_loss,
)
from iac_extensions.flow_evidence import (
    PAIR_FEATURE_DIM,
    ClassicFlowExtractor,
    RidgeSpeedHead,
    flow_statistics,
    speed_energy,
    trajectory_speed_targets,
)


class DinoMotionHeadTest(unittest.TestCase):
    def test_candidate_blind_head_shapes_and_gradients(self) -> None:
        torch.manual_seed(7)
        head = CandidateBlindDinoMotionHead(
            24,
            36,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            segment_count=3,
        )
        history = torch.randn(3, 4, 24, requires_grad=True)
        future = torch.randn(3, 4, 24, requires_grad=True)
        output = head(history, future)
        self.assertEqual(output["mean"].shape, (3, 36))
        self.assertEqual(output["log_variance"].shape, (3, 36))
        self.assertTrue(torch.all(output["mean"].abs() <= 1.0))
        output["mean"].sum().backward()
        self.assertIsNotNone(history.grad)
        self.assertIsNotNone(future.grad)

    def test_comparator_prefers_matching_candidate(self) -> None:
        comparator = UncertaintyAwareTrajectoryComparator(6)
        visual = torch.tensor([[0.1, -0.2, 0.3, 0.4, 0.0, 0.5]])
        log_variance = torch.zeros_like(visual)
        matching = comparator(visual, log_variance, visual)
        distant = comparator(visual, log_variance, visual + 0.5)
        self.assertLess(
            float(matching["energy"].detach()), float(distant["energy"].detach())
        )
        self.assertGreater(
            float(matching["logit"].detach()), float(distant["logit"].detach())
        )
        self.assertTrue(torch.all(distant["component_energy"] >= 0.0))

    def test_uncertainty_loss_is_finite_and_non_negative(self) -> None:
        prediction = torch.tensor([[0.0, 0.5]])
        target = torch.tensor([[0.1, -0.2]])
        value = uncertainty_weighted_motion_loss(
            prediction, torch.zeros_like(prediction), target
        )
        self.assertTrue(torch.isfinite(value))
        self.assertGreaterEqual(float(value), 0.0)


class FlowEvidenceTest(unittest.TestCase):
    def test_flow_statistics_contract(self) -> None:
        height, width = 48, 64
        yy, xx = np.mgrid[0:height, 0:width]
        flow = np.stack(
            [(xx - width / 2) * 0.01, (yy - height / 2) * 0.02], axis=-1
        ).astype(np.float32)
        features = flow_statistics(flow)
        self.assertEqual(features.shape, (PAIR_FEATURE_DIM,))
        self.assertTrue(np.isfinite(features).all())

    def test_dis_sequence_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(4)
            base = rng.integers(0, 256, size=(72, 96), dtype=np.uint8)
            paths = []
            for index in range(8):
                image = np.roll(base, shift=index, axis=1)
                path = root / f"frame_{index}.png"
                self.assertTrue(cv2.imwrite(str(path), image))
                paths.append(path)
            extractor = ClassicFlowExtractor("dis", width=64, height=48)
            features = extractor.sequence_features(paths)
            self.assertEqual(features.shape, (PAIR_FEATURE_DIM * 10,))
            self.assertTrue(np.isfinite(features).all())

    def test_speed_targets_and_ridge_roundtrip(self) -> None:
        trajectory = np.stack(
            [np.arange(1, 9, dtype=np.float32), np.zeros(8, np.float32)], axis=1
        )
        target = trajectory_speed_targets(trajectory)
        np.testing.assert_allclose(target[:5], 2.0, atol=1e-6)
        self.assertAlmostEqual(float(target[5]), 0.0, places=6)

        rng = np.random.default_rng(5)
        features = rng.normal(size=(80, 10))
        weights = rng.normal(size=(10, 6))
        targets = features @ weights + rng.normal(scale=0.01, size=(80, 6))
        model = RidgeSpeedHead.fit(
            features[:60], targets[:60], features[60:], targets[60:]
        )
        prediction = model.predict(features[60:])
        self.assertEqual(prediction.shape, (20, 6))
        energy_same = speed_energy(prediction, prediction, model)
        energy_far = speed_energy(prediction, prediction + 1.0, model)
        self.assertTrue(np.all(energy_same < energy_far))

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ridge.npz"
            model.save(path)
            restored = RidgeSpeedHead.load(path)
            np.testing.assert_allclose(
                model.predict(features[60:]), restored.predict(features[60:])
            )


if __name__ == "__main__":
    unittest.main()

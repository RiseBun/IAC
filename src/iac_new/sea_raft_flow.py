"""SEA-RAFT adapter with the same calibrated flow contract as RAFT-Large."""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from .flow import (
    FlowObservation,
    RaftFlowExtractor,
    forward_backward_mask,
    long_range_flow_residual,
)
from .geometry import scale_intrinsics


class SeaRaftFlowExtractor:
    def __init__(
        self,
        *,
        checkpoint: str,
        device: str,
        iters: int = 12,
        forward_backward: bool = True,
        fb_abs_threshold_px: float = 1.5,
        fb_relative_threshold: float = 0.05,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.iters = int(iters)
        self.model_size = "sea_raft_m"
        self.use_forward_backward = bool(forward_backward)
        self.fb_abs_threshold_px = float(fb_abs_threshold_px)
        self.fb_relative_threshold = float(fb_relative_threshold)
        checkpoint_path = Path(checkpoint).resolve()
        roots = []
        if os.environ.get("SEA_RAFT_ROOT"):
            roots.append(Path(os.environ["SEA_RAFT_ROOT"]).expanduser())
        roots.extend([
            checkpoint_path.parents[2] / "SEA-RAFT",
            checkpoint_path.parents[1] / "SEA-RAFT",
            Path.home() / "SEA-RAFT",
        ])
        sea_root = next((path.resolve() for path in roots if path.exists()), None)
        if sea_root is None:
            raise FileNotFoundError(
                "SEA-RAFT source tree not found; set SEA_RAFT_ROOT explicitly"
            )
        sys.path.insert(0, str(sea_root))
        sys.path.insert(0, str(sea_root / "core"))
        from raft import RAFT
        from safetensors.torch import load_file

        args = Namespace(
            use_var=True, var_min=0.0, var_max=10.0, pretrain="resnet34",
            initial_dim=64, block_dims=[64, 128, 256], radius=4, dim=128,
            num_blocks=2, iters=self.iters, image_size=[432, 960], scale=0,
            batch_size=1, epsilon=1e-8, lr=1e-4, wdecay=1e-5, dropout=0,
            clip=1.0, gamma=0.85, num_steps=10000, restore_ckpt=None,
            coarse_config=None, skip_pretrain=True,
        )
        model = RAFT(args)
        model.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=False)
        self.model = model.to(device).eval()

    @staticmethod
    def _read_images(
        paths: list[str],
        intrinsics: np.ndarray,
        distortion: np.ndarray,
        target_size: tuple[int, int],
        allow_mixed_source_sizes: bool = False,
        intrinsics_source_size: tuple[int, int] | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray, tuple[int, int]]:
        return RaftFlowExtractor._read_images(
            paths,
            intrinsics,
            distortion,
            target_size,
            allow_mixed_source_sizes=allow_mixed_source_sizes,
            intrinsics_source_size=intrinsics_source_size,
        )

    def _infer_pairs(
        self,
        first: list[np.ndarray],
        second: list[np.ndarray],
        *,
        return_uncertainty: bool = False,
        uncertainty_tail: int = 8,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if len(first) != len(second):
            raise ValueError("SEA-RAFT pair lists must have equal length")
        outputs = []
        uncertainties = []
        torch = self.torch
        for left, right in zip(first, second):
            image1 = torch.from_numpy(left.copy()).permute(2, 0, 1).float()[None].to(self.device)
            image2 = torch.from_numpy(right.copy()).permute(2, 0, 1).float()[None].to(self.device)
            with torch.inference_mode():
                predictions = self.model(image1, image2, iters=self.iters, test_mode=True)["flow"]
            prediction = predictions[-1]
            outputs.append(prediction[0].permute(1, 2, 0).float().cpu().numpy())
            if return_uncertainty:
                tail = predictions[-max(2, min(int(uncertainty_tail), len(predictions))):]
                stack = torch.stack(tail, dim=0)
                spread = torch.sqrt(torch.mean((stack - stack[-1:]) ** 2, dim=0))
                uncertainties.append(spread[0, 0].float().cpu().numpy())
        flow = np.stack(outputs, axis=0).astype(np.float32)
        uncertainty = (
            np.stack(uncertainties, axis=0).astype(np.float32)
            if return_uncertainty else None
        )
        return flow, uncertainty

    def observe(
        self,
        frame_paths: list[str],
        intrinsics: np.ndarray,
        distortion: np.ndarray,
        target_size: tuple[int, int],
        inference_size: tuple[int, int] | None = None,
        allow_mixed_source_sizes: bool = False,
        intrinsics_source_size: tuple[int, int] | None = None,
        return_uncertainty: bool = False,
        uncertainty_tail: int = 8,
        long_range_consistency: bool = False,
    ) -> FlowObservation:
        if len(frame_paths) < 2:
            raise ValueError("at least two video frames are required")
        inference_size = target_size if inference_size is None else tuple(inference_size)
        images, _, source_size = self._read_images(
            frame_paths,
            np.asarray(intrinsics, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
            inference_size,
            allow_mixed_source_sizes=allow_mixed_source_sizes,
            intrinsics_source_size=intrinsics_source_size,
        )
        forward, uncertainty = self._infer_pairs(
            images[:-1], images[1:], return_uncertainty=return_uncertainty,
            uncertainty_tail=uncertainty_tail,
        )
        long_range_residual = None
        if long_range_consistency and len(images) >= 3:
            skip_forward, _ = self._infer_pairs(images[:-2], images[2:])
            residuals = [
                long_range_flow_residual(direct, skip)
                for direct, skip in zip(forward[:-1], skip_forward)
            ]
            residuals.append(np.full(forward[-1].shape[:2], np.nan, dtype=np.float32))
            long_range_residual = np.stack(residuals, axis=0)
        backward = None
        if self.use_forward_backward:
            backward, _ = self._infer_pairs(images[1:], images[:-1])

        if inference_size != target_size:
            def resize_flow(value: np.ndarray) -> np.ndarray:
                resized = cv2.resize(value, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
                resized[..., 0] *= target_size[0] / float(inference_size[0])
                resized[..., 1] *= target_size[1] / float(inference_size[1])
                return resized

            forward = np.stack([resize_flow(value) for value in forward], axis=0)
            if backward is not None:
                backward = np.stack([resize_flow(value) for value in backward], axis=0)
            if uncertainty is not None:
                uncertainty = np.stack([
                    cv2.resize(value, target_size, interpolation=cv2.INTER_LINEAR)
                    for value in uncertainty
                ], axis=0).astype(np.float32)
            if long_range_residual is not None:
                long_range_residual = np.stack([
                    cv2.resize(value, target_size, interpolation=cv2.INTER_LINEAR)
                    for value in long_range_residual
                ], axis=0).astype(np.float32)

        masks = None
        if backward is not None:
            masks = np.stack([
                forward_backward_mask(
                    fwd, bwd,
                    absolute_threshold_px=self.fb_abs_threshold_px,
                    relative_threshold=self.fb_relative_threshold,
                )
                for fwd, bwd in zip(forward, backward)
            ], axis=0)
        calibration_size = source_size if intrinsics_source_size is None else intrinsics_source_size
        return FlowObservation(
            forward=forward,
            consistency_masks=masks,
            intrinsics=scale_intrinsics(intrinsics, calibration_size, target_size),
            source_size=source_size,
            target_size=target_size,
            refinement_uncertainty=uncertainty,
            long_range_residual=long_range_residual,
        )

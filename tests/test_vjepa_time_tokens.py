from __future__ import annotations

import argparse
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

from iac_extensions.vjepa_time_tokens import (  # noqa: E402
    legacy_chunks_to_time_tokens,
    pool_flattened_vjepa_time_tokens,
)
from migrate_vjepa_time_tokens import migrate  # noqa: E402


class VJEPATimeTokenTest(unittest.TestCase):
    def test_shape_aware_pooling_keeps_time_and_removes_space(self) -> None:
        grid = torch.zeros(2, 4, 2, 3, 5)
        for time_index in range(4):
            grid[:, time_index] = float(time_index * 10)
            grid[:, time_index, 0, 0] += 3.0
        hidden = grid.reshape(2, 4 * 2 * 3, 5)
        pooled, layout = pool_flattened_vjepa_time_tokens(
            hidden,
            num_frames=8,
            image_height=32,
            image_width=48,
            tubelet_size=2,
            patch_size=16,
        )
        self.assertEqual(tuple(pooled.shape), (2, 4, 5))
        self.assertEqual(layout["temporal_tokens"], 4)
        torch.testing.assert_close(
            pooled[0, :, 0],
            torch.tensor([0.5, 10.5, 20.5, 30.5]),
        )

    def test_legacy_sixteen_chunks_recover_four_time_tokens(self) -> None:
        values = torch.arange(4, dtype=torch.float32).repeat_interleave(4)
        chunks = values.view(1, 16, 1)
        restored, metadata = legacy_chunks_to_time_tokens(
            chunks,
            native_temporal_tokens=4,
        )
        torch.testing.assert_close(
            restored.flatten(),
            torch.arange(4, dtype=torch.float32),
        )
        self.assertEqual(metadata["legacy_chunks_per_time"], 4)

    def test_legacy_sixteen_chunks_can_be_temporal_windows_of_32(self) -> None:
        chunks = torch.arange(16, dtype=torch.float32).view(1, 16, 1)
        restored, metadata = legacy_chunks_to_time_tokens(
            chunks,
            native_temporal_tokens=32,
        )
        torch.testing.assert_close(restored, chunks)
        self.assertEqual(metadata["native_times_per_output"], 2)

    def test_migration_writes_compact_time_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "legacy.pt"
            output_path = root / "time.pt"
            chunks = (
                torch.arange(4, dtype=torch.float32)
                .repeat_interleave(4)
                .view(1, 16, 1)
            )
            torch.save(
                {
                    "sample_id": ["sample"],
                    "x": torch.randn(1, 3072),
                    "x_tokens": chunks,
                    "metadata": {"num_frames": 8},
                },
                source_path,
            )
            summary = migrate(
                argparse.Namespace(
                    input_cache=str(source_path),
                    output_cache=str(output_path),
                    output_summary=str(root / "summary.json"),
                    source_key="x_tokens",
                    output_key="x_time_tokens",
                    num_frames=0,
                    tubelet_size=2,
                    preserve_all=False,
                )
            )
            migrated = torch.load(
                output_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertNotIn("x", migrated)
            self.assertEqual(summary["output_shape"], [1, 4, 1])
            torch.testing.assert_close(
                migrated["x_time_tokens"].flatten(),
                torch.arange(4, dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()

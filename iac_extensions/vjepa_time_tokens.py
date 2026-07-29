"""Shape-aware temporal pooling for flattened V-JEPA patch tokens.

Hugging Face V-JEPA2 flattens the Conv3D output grid from ``(T, H, W)`` to a
single token axis.  A generic chunk of that flat axis is therefore not
necessarily a time token.  The helpers here restore the explicit grid before
pooling spatial patches.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch


def _pair(value: Any, *, name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        result = (int(value), int(value))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [int(item) for item in value]
        if len(items) == 1:
            result = (items[0], items[0])
        elif len(items) == 2:
            result = (items[0], items[1])
        else:
            raise ValueError(f"{name} must contain one or two values")
    else:
        raise ValueError(f"{name} must be an int or a one/two-value sequence")
    if min(result) <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def infer_vjepa_patch_layout(
    *,
    token_count: int,
    num_frames: int,
    image_height: int,
    image_width: int,
    tubelet_size: int,
    patch_size: Any,
) -> Dict[str, int]:
    """Infer and validate the explicit ``(T,H,W)`` patch-token layout."""

    if min(
        int(token_count),
        int(num_frames),
        int(image_height),
        int(image_width),
        int(tubelet_size),
    ) <= 0:
        raise ValueError("token/layout dimensions must be positive")
    patch_height, patch_width = _pair(patch_size, name="patch_size")
    if num_frames % tubelet_size:
        raise ValueError(
            f"num_frames={num_frames} is not divisible by "
            f"tubelet_size={tubelet_size}"
        )
    if image_height % patch_height or image_width % patch_width:
        raise ValueError(
            "processed image size is not divisible by the V-JEPA patch size: "
            f"{image_height}x{image_width} vs {patch_height}x{patch_width}"
        )
    temporal = num_frames // tubelet_size
    spatial_height = image_height // patch_height
    spatial_width = image_width // patch_width
    expected = temporal * spatial_height * spatial_width
    if int(token_count) != expected:
        raise ValueError(
            "flattened V-JEPA token count does not match the inferred "
            f"(T,H,W) layout: got {token_count}, expected {expected}="
            f"{temporal}*{spatial_height}*{spatial_width}"
        )
    return {
        "temporal_tokens": temporal,
        "spatial_height": spatial_height,
        "spatial_width": spatial_width,
        "spatial_tokens_per_time": spatial_height * spatial_width,
        "flattened_tokens": expected,
        "num_frames": int(num_frames),
        "image_height": int(image_height),
        "image_width": int(image_width),
        "tubelet_size": int(tubelet_size),
        "patch_height": patch_height,
        "patch_width": patch_width,
    }


def pool_flattened_vjepa_time_tokens(
    hidden: torch.Tensor,
    *,
    num_frames: int,
    image_height: int,
    image_width: int,
    tubelet_size: int,
    patch_size: Any,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Return true time tokens by restoring ``(T,H,W)`` and pooling ``H,W``."""

    if hidden.ndim != 3:
        raise ValueError(
            "hidden must have shape (batch,flattened_token,feature), "
            f"got {tuple(hidden.shape)}"
        )
    layout = infer_vjepa_patch_layout(
        token_count=int(hidden.shape[1]),
        num_frames=int(num_frames),
        image_height=int(image_height),
        image_width=int(image_width),
        tubelet_size=int(tubelet_size),
        patch_size=patch_size,
    )
    grid = hidden.reshape(
        hidden.shape[0],
        layout["temporal_tokens"],
        layout["spatial_height"],
        layout["spatial_width"],
        hidden.shape[-1],
    )
    return grid.mean(dim=(2, 3)), layout


def legacy_chunks_to_time_tokens(
    chunks: torch.Tensor,
    *,
    native_temporal_tokens: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Recover time-wise spatial means from legacy contiguous flat chunks.

    This is exact when the legacy chunks were created by equally chunking the
    T-major flattened ``(T,H,W)`` axis and each temporal slice spans an integer
    number of chunks.  The old IAC cache used 16 equal chunks for four V-JEPA
    tubelets, so averaging each consecutive group of four recovers the same
    spatial mean as shape-aware pooling.
    """

    if chunks.ndim != 3:
        raise ValueError(
            "legacy chunks must have shape (row,chunk,feature), "
            f"got {tuple(chunks.shape)}"
        )
    if native_temporal_tokens < 2:
        raise ValueError("native_temporal_tokens must be at least two")
    chunk_count = int(chunks.shape[1])
    native_temporal_tokens = int(native_temporal_tokens)
    if chunk_count % native_temporal_tokens == 0:
        chunks_per_time = chunk_count // native_temporal_tokens
        restored = chunks.reshape(
            chunks.shape[0],
            native_temporal_tokens,
            chunks_per_time,
            chunks.shape[-1],
        ).mean(dim=2)
        return restored, {
            "legacy_chunk_count": chunk_count,
            "native_temporal_tokens": native_temporal_tokens,
            "output_temporal_tokens": native_temporal_tokens,
            "legacy_chunks_per_time": chunks_per_time,
            "native_times_per_output": 1,
        }
    if native_temporal_tokens % chunk_count == 0:
        native_times_per_output = native_temporal_tokens // chunk_count
        return chunks.clone(), {
            "legacy_chunk_count": chunk_count,
            "native_temporal_tokens": native_temporal_tokens,
            "output_temporal_tokens": chunk_count,
            "legacy_chunks_per_time": 1,
            "native_times_per_output": native_times_per_output,
        }
    raise ValueError(
        "legacy chunks cannot be separated into pure temporal windows: "
        f"chunk_count={chunk_count}, "
        f"native_temporal_tokens={native_temporal_tokens}"
    )

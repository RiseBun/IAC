#!/usr/bin/env python3
"""Build a compact true-time-token cache from an existing V-JEPA cache.

The source cache is never modified.  If it already contains ``x_time_tokens``,
that key is validated and copied.  Otherwise the tool reconstructs pure
temporal windows from legacy equal chunks of the T-major flattened V-JEPA
``(T,H,W)`` patch axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_extensions.vjepa_time_tokens import (  # noqa: E402
    legacy_chunks_to_time_tokens,
)
from ordered_motion_common import sha256, write_json  # noqa: E402


def _metadata_int(
    metadata: Dict[str, Any],
    key: str,
    fallback: int,
) -> int:
    value = metadata.get(key, fallback)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cache metadata {key!r} is not an integer") from exc
    if result <= 0:
        raise ValueError(f"cache metadata {key!r} must be positive")
    return result


def migrate(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input_cache)
    output_path = Path(args.output_cache)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output cache paths must differ")
    source = torch.load(input_path, map_location="cpu", weights_only=False)
    metadata = dict(source.get("metadata", {}))
    output_key = str(args.output_key)
    if output_key in source:
        time_tokens = source[output_key]
        if not isinstance(time_tokens, torch.Tensor) or time_tokens.ndim != 3:
            raise ValueError(
                f"{output_key} must have shape (row,time,feature)"
            )
        conversion = {
            "mode": "copied_existing_shape_aware_time_tokens",
            "output_temporal_tokens": int(time_tokens.shape[1]),
        }
    else:
        source_key = str(args.source_key)
        if source_key not in source:
            raise KeyError(
                f"cache has neither {output_key!r} nor {source_key!r}"
            )
        legacy = source[source_key]
        if not isinstance(legacy, torch.Tensor) or legacy.ndim != 3:
            raise ValueError(
                f"{source_key} must have shape (row,chunk,feature)"
            )
        num_frames = (
            int(args.num_frames)
            if int(args.num_frames) > 0
            else _metadata_int(metadata, "num_frames", 0)
        )
        tubelet_size = int(args.tubelet_size)
        if tubelet_size <= 0 or num_frames % tubelet_size:
            raise ValueError(
                f"num_frames={num_frames} must be divisible by "
                f"tubelet_size={tubelet_size}"
            )
        native_temporal_tokens = num_frames // tubelet_size
        time_tokens, details = legacy_chunks_to_time_tokens(
            legacy,
            native_temporal_tokens=native_temporal_tokens,
        )
        conversion = {
            "mode": "recovered_from_legacy_flat_T_H_W_chunks",
            "source_key": source_key,
            "num_frames": num_frames,
            "tubelet_size": tubelet_size,
            **details,
        }

    sample_ids = list(source.get("sample_id", []))
    if len(sample_ids) != int(time_tokens.shape[0]):
        raise ValueError(
            f"sample_id count {len(sample_ids)} != feature rows "
            f"{time_tokens.shape[0]}"
        )
    output_metadata = {
        **metadata,
        "time_token_key": output_key,
        "time_token_count": int(time_tokens.shape[1]),
        "time_token_feature_dim": int(time_tokens.shape[-1]),
        "time_token_layout": {
            "kind": "shape_aware_vjepa_time_windows",
            "source_grid": "T_H_W",
            "spatial_pooling": "mean",
            **conversion,
        },
    }
    output: Dict[str, Any] = {
        "sample_id": sample_ids,
        output_key: time_tokens,
        "metadata": output_metadata,
    }
    for key in ("group_id", "source_type", "y"):
        if key in source:
            output[key] = source[key]
    if bool(args.preserve_all):
        output = dict(source)
        output[output_key] = time_tokens
        output["metadata"] = output_metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    summary = {
        "kind": "vjepa_time_token_cache_migration",
        "input_cache": str(input_path),
        "input_sha256": sha256(input_path),
        "output_cache": str(output_path),
        "output_sha256": sha256(output_path),
        "rows": int(time_tokens.shape[0]),
        "output_shape": list(time_tokens.shape),
        "output_key": output_key,
        "preserve_all": bool(args.preserve_all),
        "conversion": conversion,
    }
    if args.output_summary:
        write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--source-key", default="x_tokens")
    parser.add_argument("--output-key", default="x_time_tokens")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="Override cache metadata num_frames; 0 reads metadata.",
    )
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--preserve-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    migrate(parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract frozen V-JEPA2 video features for IAC visual-side probes.

This extractor is candidate-blind with respect to visual encoding: it only sees
the ordered history/future image sequence stored in each index row. Candidate
trajectory features are left to downstream agreement scorers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_extensions.vjepa_time_tokens import (
    pool_flattened_vjepa_time_tokens,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object row")
                rows.append(value)
    return rows


def _source(row: Dict[str, Any]) -> str:
    for key in ("source_type", "action_type", "wam_name", "sample_type", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _group_id(row: Dict[str, Any], fallback: str) -> str:
    value = row.get("group_id") or row.get("anchor_id")
    if value is not None:
        return str(value)
    sample_id = str(row.get("sample_id", fallback))
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def _select_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    max_groups: int,
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    selected = list(rows)
    if max_groups > 0:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[_group_id(row, str(index))].append(row)
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        selected = []
        for key in keys[:max_groups]:
            selected.extend(groups[key])
    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _pad_tail(values: Sequence[object], count: int) -> List[object]:
    items = list(values[-count:])
    if not items:
        raise ValueError("row does not contain a usable image sequence")
    if len(items) < count:
        items = [items[0]] * (count - len(items)) + items
    return items


def _row_paths(
    row: Dict[str, Any],
    *,
    image_root: Path,
    history_num_frames: int,
    future_num_frames: int,
    video_mode: str,
) -> List[Path]:
    history = _pad_tail(row.get("history_images", []), history_num_frames)
    future = _pad_tail(row.get("future_images", []), future_num_frames)
    if video_mode == "history_future":
        values = [*history, *future]
    elif video_mode == "future_only":
        values = future
    elif video_mode == "history_only":
        values = history
    else:
        raise ValueError(f"unknown video mode: {video_mode}")
    return [_resolve(image_root, value) for value in values]


def _resample_paths(paths: Sequence[Path], num_frames: int) -> List[Path]:
    if len(paths) < 1:
        raise ValueError("empty path sequence")
    if num_frames <= 0 or len(paths) == num_frames:
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, num_frames)
    return [paths[int(round(index))] for index in indices]


def _load_video(paths: Sequence[Path]) -> torch.Tensor:
    frames = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        frames.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(frames, dim=0)


def _pool_hidden(hidden: torch.Tensor, mode: str) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError(f"expected hidden shape (B,N,D), got {tuple(hidden.shape)}")
    mean = hidden.mean(dim=1)
    if mode == "mean":
        return mean
    std = hidden.std(dim=1)
    if mode == "mean_std":
        return torch.cat([mean, std], dim=-1)
    chunks = torch.chunk(hidden, chunks=4, dim=1)
    first = chunks[0].mean(dim=1)
    last = chunks[-1].mean(dim=1)
    if mode == "mean_std_diff":
        return torch.cat([mean, std, last - first], dim=-1)
    if mode == "mean_std_first_last_diff":
        return torch.cat([mean, std, first, last, last - first], dim=-1)
    raise ValueError(f"unknown pooling mode: {mode}")


def _token_summary(hidden: torch.Tensor, count: int) -> torch.Tensor:
    """Return legacy flat spatiotemporal chunks for old-gate compatibility."""

    if count <= 0:
        raise ValueError("token summary count must be positive")
    chunks = torch.chunk(hidden, chunks=count, dim=1)
    return torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)


def _batched(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument(
        "--video-mode",
        choices=("history_future", "future_only", "history_only"),
        default="history_future",
    )
    parser.add_argument("--history-num-frames", type=int, default=4)
    parser.add_argument("--future-num-frames", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument(
        "--pooling",
        choices=("mean", "mean_std", "mean_std_diff", "mean_std_first_last_diff"),
        default="mean_std_diff",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--token-summary-size",
        type=int,
        default=0,
        help=(
            "If >0, also save legacy flattened spatiotemporal chunks as "
            "x_tokens for old-gate compatibility. True time tokens are always "
            "saved separately as x_time_tokens."
        ),
    )
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError as exc:
        raise SystemExit(
            "transformers with V-JEPA2 support is required; install/upgrade transformers"
        ) from exc

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(args.dtype)
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    elif device.type == "cuda":
        load_kwargs["torch_dtype"] = torch.float16

    processor = AutoVideoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = AutoModel.from_pretrained(args.model_name, **load_kwargs).to(device)
    model.eval()

    rows = _select_rows(
        _read_jsonl(Path(args.index)),
        max_groups=int(args.max_groups),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    image_root = Path(args.image_root)
    features: List[torch.Tensor] = []
    token_features: List[torch.Tensor] = []
    time_token_features: List[torch.Tensor] = []
    time_token_layout: Dict[str, int] | None = None
    sample_ids: List[str] = []
    group_ids: List[str] = []
    source_types: List[str] = []
    trajectories: List[torch.Tensor] = []
    start = time.time()

    with torch.inference_mode():
        for batch_idx, batch_rows in enumerate(_batched(rows, int(args.batch_size)), start=1):
            videos = []
            for row in batch_rows:
                paths = _row_paths(
                    row,
                    image_root=image_root,
                    history_num_frames=int(args.history_num_frames),
                    future_num_frames=int(args.future_num_frames),
                    video_mode=str(args.video_mode),
                )
                videos.append(_load_video(_resample_paths(paths, int(args.num_frames))))
            inputs = processor(videos, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            try:
                outputs = model(**inputs, skip_predictor=True)
            except TypeError:
                outputs = model(**inputs)
            pooled = _pool_hidden(outputs.last_hidden_state.float(), str(args.pooling))
            features.append(pooled.cpu())
            processed = inputs.get("pixel_values_videos")
            if processed is None or processed.ndim != 5:
                raise ValueError(
                    "V-JEPA processor output must contain pixel_values_videos "
                    "with shape (batch,time,channel,height,width)"
                )
            temporal, current_layout = pool_flattened_vjepa_time_tokens(
                outputs.last_hidden_state.float(),
                num_frames=int(processed.shape[1]),
                image_height=int(processed.shape[-2]),
                image_width=int(processed.shape[-1]),
                tubelet_size=int(model.config.tubelet_size),
                patch_size=model.config.patch_size,
            )
            if (
                time_token_layout is not None
                and current_layout != time_token_layout
            ):
                raise ValueError(
                    "processed V-JEPA token layout changed between batches: "
                    f"{time_token_layout} vs {current_layout}"
                )
            time_token_layout = current_layout
            time_token_features.append(temporal.cpu())
            if int(args.token_summary_size) > 0:
                token_features.append(_token_summary(outputs.last_hidden_state.float(), int(args.token_summary_size)).cpu())
            for row in batch_rows:
                sample_ids.append(str(row.get("sample_id", len(sample_ids))))
                group_ids.append(_group_id(row, str(len(group_ids))))
                source_types.append(_source(row))
                trajectories.append(torch.tensor(row.get("candidate_traj", []), dtype=torch.float32))
            if args.log_every > 0 and (
                batch_idx == 1
                or batch_idx % int(args.log_every) == 0
                or len(sample_ids) == len(rows)
            ):
                elapsed = max(time.time() - start, 1e-6)
                print(
                    json.dumps(
                        {
                            "kind": "vjepa_feature_progress",
                            "output": args.output,
                            "batches": batch_idx,
                            "samples": len(sample_ids),
                            "total": len(rows),
                            "samples_per_sec": len(sample_ids) / elapsed,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    max_steps = max((int(t.shape[0]) if t.ndim == 2 else 0) for t in trajectories) if trajectories else 0
    max_dims = max((int(t.shape[1]) if t.ndim == 2 else 0) for t in trajectories) if trajectories else 0
    y = torch.zeros((len(trajectories), max_steps, max_dims), dtype=torch.float32)
    for index, traj in enumerate(trajectories):
        if traj.ndim == 2 and max_steps and max_dims:
            y[index, : traj.shape[0], : traj.shape[1]] = traj

    out = {
        "x": torch.cat(features, dim=0),
        "x_time_tokens": torch.cat(time_token_features, dim=0),
        "y": y,
        "sample_id": sample_ids,
        "group_id": group_ids,
        "source_type": source_types,
        "metadata": {
            "kind": "vjepa_video_feature_cache",
            "model_name": args.model_name,
            "video_mode": args.video_mode,
            "num_frames": int(args.num_frames),
            "history_num_frames": int(args.history_num_frames),
            "future_num_frames": int(args.future_num_frames),
            "pooling": args.pooling,
            "feature_dim": int(torch.cat(features, dim=0).shape[1]) if features else 0,
            "rows": len(rows),
            "time_token_key": "x_time_tokens",
            "time_token_layout": {
                "kind": "shape_aware_vjepa_time_tokens",
                "source_grid": "T_H_W",
                "spatial_pooling": "mean",
                **(time_token_layout or {}),
            },
        },
    }
    if token_features:
        out["x_tokens"] = torch.cat(token_features, dim=0)
        out["metadata"]["token_summary_size"] = int(args.token_summary_size)
        out["metadata"]["token_feature_dim"] = int(out["x_tokens"].shape[-1])
        out["metadata"]["x_tokens_semantics"] = (
            "legacy_equal_chunks_of_flattened_T_H_W_not_pure_time"
        )
    out["metadata"]["time_token_count"] = int(out["x_time_tokens"].shape[1])
    out["metadata"]["time_token_feature_dim"] = int(
        out["x_time_tokens"].shape[-1]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    print(json.dumps(out["metadata"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

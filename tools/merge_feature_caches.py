#!/usr/bin/env python3
"""Merge feature cache shards produced by IAC feature extractors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _items(cache: Dict[str, Any], key: str) -> List[Any]:
    value = cache.get(key, [])
    return list(value)


def main() -> None:
    args = parse_args()
    caches = [torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs]
    if not caches:
        raise ValueError("no input caches")
    x = torch.cat([cache["x"].float() for cache in caches], dim=0)
    token_tensors = [cache.get("x_tokens") for cache in caches if cache.get("x_tokens") is not None]
    y_tensors = [cache.get("y") for cache in caches if cache.get("y") is not None]
    y = None
    if y_tensors:
        max_steps = max(int(t.shape[1]) for t in y_tensors)
        max_dims = max(int(t.shape[2]) for t in y_tensors)
        y = torch.zeros((x.shape[0], max_steps, max_dims), dtype=torch.float32)
        offset = 0
        for tensor in y_tensors:
            tensor = tensor.float()
            y[offset : offset + tensor.shape[0], : tensor.shape[1], : tensor.shape[2]] = tensor
            offset += tensor.shape[0]
    out = {
        "x": x,
        "sample_id": sum((_items(cache, "sample_id") for cache in caches), []),
        "group_id": sum((_items(cache, "group_id") for cache in caches), []),
        "source_type": sum((_items(cache, "source_type") for cache in caches), []),
        "metadata": dict(caches[0].get("metadata", {})),
    }
    if token_tensors:
        out["x_tokens"] = torch.cat([tensor.float() for tensor in token_tensors], dim=0)
    if y is not None:
        out["y"] = y
    out["metadata"].update(
        {
            "kind": f"merged_{out['metadata'].get('kind', 'feature_cache')}",
            "rows": int(x.shape[0]),
            "feature_dim": int(x.shape[1]) if x.ndim == 2 else None,
            "num_shards": len(caches),
            "inputs": [str(path) for path in args.inputs],
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    print(json.dumps(out["metadata"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

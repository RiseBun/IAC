#!/usr/bin/env python3
"""Apply a trained clean V-JEPA trajectory-cross-attention mismatch gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

import _pathfix  # noqa: F401

from train_visual_mismatch_gate_scorer import (
    MismatchGate,
    _load_dataset,
    _score_rows,
    _write_jsonl,
)


def _load_gate(path: Path) -> Dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(bundle.get("metadata", {}))
    train_args = dict(metadata.get("args", {}))
    model = MismatchGate(
        int(metadata["visual_dim"]),
        int(metadata["scalar_dim"]),
        int(train_args.get("visual_hidden_dim", 32)),
        int(train_args.get("hidden_dim", 64)),
        float(train_args.get("dropout", 0.0)),
        str(metadata.get("interaction_kind", train_args.get("interaction_kind", "traj_cross_attention"))),
        int((metadata.get("traj_shape") or [8, 5])[-1]),
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    bundle["model"] = model
    bundle["metadata"] = metadata
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--visual-cache", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--visual-cache-key", default=None)
    parser.add_argument("--scalar-feature-mode", choices=["full", "zero"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    bundle = _load_gate(Path(args.model))
    metadata = bundle.get("metadata", {})
    train_args = metadata.get("args", {})
    feature_key = str(args.visual_cache_key or train_args.get("visual_cache_key") or "x")
    scalar_mode = str(args.scalar_feature_mode or metadata.get("scalar_feature_mode") or "zero")
    rows, visual, scalar, traj = _load_dataset(
        Path(args.rows),
        Path(args.visual_cache),
        feature_key=feature_key,
        scalar_feature_mode=scalar_mode,
    )
    scored = _score_rows(bundle, rows, visual, scalar, traj)
    _write_jsonl(Path(args.output_scores), scored)
    print(
        json.dumps(
            {
                "rows": len(scored),
                "output": str(args.output_scores),
                "visual_cache_key": feature_key,
                "scalar_feature_mode": scalar_mode,
                "model_kind": metadata.get("kind"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

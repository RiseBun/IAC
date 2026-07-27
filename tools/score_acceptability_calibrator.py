#!/usr/bin/env python3
"""Apply the trained v3 IAC acceptability calibrator to a score JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from train_iac_acceptability_calibrator import (
    Calibrator,
    _apply,
    _parse_paths,
)


def _parse_sources(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _load_model(path: Path, device: torch.device) -> tuple[Calibrator, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(bundle.get("metadata", {}))
    feature_dim = int(metadata.get("feature_dim") or bundle["mean"].numel())
    hidden_dim = int(metadata.get("hidden_dim", 0))
    model = Calibrator(feature_dim, hidden_dim, dropout=0.0)
    model.load_state_dict(bundle["model"])
    model.to(device)
    model.eval()
    return model, bundle["mean"], bundle["std"], metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--primary-scores", required=True)
    parser.add_argument("--aux", default="", help="Comma-separated auxiliary score JSONLs aligned with primary.")
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--source-key", default="source_type")
    parser.add_argument(
        "--acceptable-sources",
        default="gt_pos,perturb_speed,perturb_lateral,perturb_heading",
    )
    parser.add_argument(
        "--hard-sources",
        default="image_swap,time_shift_future,traj_swap,reverse_traj,high_pdm_image_mismatch",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, mean, std, metadata = _load_model(Path(args.model), device)
    acceptable_sources = _parse_sources(args.acceptable_sources)
    hard_sources = _parse_sources(args.hard_sources)
    summary = _apply(
        model,
        Path(args.primary_scores),
        _parse_paths(args.aux),
        mean=mean,
        std=std,
        group_key=str(args.group_key),
        source_key=str(args.source_key),
        acceptable_sources=acceptable_sources,
        hard_sources=hard_sources,
        output_scores=Path(args.output_scores),
        device=device,
    )
    record: Dict[str, Any] = {
        "model": str(args.model),
        "model_metadata": metadata,
        "eval": summary,
    }
    if args.output_summary:
        out = Path(args.output_summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record["eval"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

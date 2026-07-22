#!/usr/bin/env python3
"""Fit or apply the candidate-blind optical-flow speed head.

Examples
--------
Fit only on exact positive image/trajectory pairs::

    python tools/flow_speed_head.py fit --train-index train.jsonl \
      --val-index val.jsonl --image-root /data/navsim \
      --output work_dirs/flow_speed/flow_speed_head.npz

Append visual predictions and lower-is-better energies to an IAC index::

    python tools/flow_speed_head.py apply --index test.jsonl \
      --image-root /data/navsim --model work_dirs/flow_speed/flow_speed_head.npz \
      --output work_dirs/flow_speed/test_with_flow.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iac_extensions.flow_evidence import (
    ClassicFlowExtractor,
    RidgeSpeedHead,
    SPEED_NAMES,
    spearman_correlation,
    speed_energy,
    trajectory_speed_targets,
    visual_sequence,
)


_THREAD_LOCAL = threading.local()


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def deterministic_subset(
    rows: Sequence[Dict[str, object]], maximum: int, seed: int
) -> List[Dict[str, object]]:
    if maximum <= 0 or len(rows) <= maximum:
        return list(rows)
    indices = np.random.default_rng(seed).choice(len(rows), size=maximum, replace=False)
    return [rows[int(index)] for index in np.sort(indices)]


def positive_rows(
    rows: Sequence[Dict[str, object]], minimum_label: float
) -> List[Dict[str, object]]:
    return [
        row
        for row in rows
        if float(row.get("consistency_label", 0.0)) >= minimum_label
    ]


def sequence_digest(
    sequence: Sequence[str], method: str, width: int, height: int
) -> str:
    digest = hashlib.sha256(f"iac-flow-v1\0{method}\0{width}x{height}\0".encode())
    for value in sequence:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def extract_one(
    task: Tuple[Tuple[str, ...], str, int, int, str]
) -> Tuple[Tuple[str, ...], np.ndarray]:
    sequence, method, width, height, cache_value = task
    cache_path = Path(cache_value) if cache_value else None
    if cache_path is not None and cache_path.exists():
        value = np.load(cache_path)
        return sequence, value.astype(np.float32, copy=False)
    settings = (method, width, height)
    if getattr(_THREAD_LOCAL, "settings", None) != settings:
        _THREAD_LOCAL.extractor = ClassicFlowExtractor(
            method, width=width, height=height
        )
        _THREAD_LOCAL.settings = settings
    value = _THREAD_LOCAL.extractor.sequence_features(sequence)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npy")
        np.save(temporary, value)
        temporary.replace(cache_path)
    return sequence, value


def extract_features(
    sequences: Sequence[Tuple[str, ...]],
    *,
    method: str,
    width: int,
    height: int,
    workers: int,
    cache_dir: Path | None,
) -> np.ndarray:
    unique = list(dict.fromkeys(sequences))
    tasks = []
    for sequence in unique:
        cache_path = ""
        if cache_dir is not None:
            digest = sequence_digest(sequence, method, width, height)
            cache_path = str(cache_dir / f"{digest}.npy")
        tasks.append((sequence, method, width, height, cache_path))
    lookup: Dict[Tuple[str, ...], np.ndarray] = {}
    if workers <= 1:
        iterator = map(extract_one, tasks)
        for index, (sequence, value) in enumerate(iterator, start=1):
            lookup[sequence] = value
            if index % 1000 == 0 or index == len(tasks):
                print(f"flow sequences {index}/{len(tasks)}", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for index, (sequence, value) in enumerate(
                executor.map(extract_one, tasks), start=1
            ):
                lookup[sequence] = value
                if index % 1000 == 0 or index == len(tasks):
                    print(f"flow sequences {index}/{len(tasks)}", flush=True)
    return np.stack([lookup[sequence] for sequence in sequences])


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, object]:
    result: Dict[str, object] = {}
    correlations = []
    for column, name in enumerate(SPEED_NAMES):
        truth = target[:, column]
        estimate = prediction[:, column]
        correlation = spearman_correlation(truth, estimate)
        correlations.append(correlation)
        result[name] = {
            "mae": float(np.mean(np.abs(truth - estimate))),
            "rmse": float(np.sqrt(np.mean(np.square(truth - estimate)))),
            "srocc": correlation if np.isfinite(correlation) else None,
        }
    finite = [value for value in correlations if np.isfinite(value)]
    result["mean_srocc"] = float(np.mean(finite)) if finite else None
    return result


def metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".json")


def run_fit(args: argparse.Namespace) -> None:
    train_rows = positive_rows(read_jsonl(Path(args.train_index)), args.min_label)
    validation_rows = positive_rows(read_jsonl(Path(args.val_index)), args.min_label)
    train_rows = deterministic_subset(train_rows, args.max_train_positives, args.seed)
    validation_rows = deterministic_subset(
        validation_rows, args.max_val_positives, args.seed + 1
    )
    if not train_rows or not validation_rows:
        raise ValueError("no exact positive rows remain after filtering")
    image_root = Path(args.image_root)
    train_sequences = [visual_sequence(row, image_root) for row in train_rows]
    validation_sequences = [visual_sequence(row, image_root) for row in validation_rows]
    all_features = extract_features(
        [*train_sequences, *validation_sequences],
        method=args.method,
        width=args.width,
        height=args.height,
        workers=args.workers,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    split = len(train_rows)
    train_features = all_features[:split]
    validation_features = all_features[split:]
    train_targets = np.stack(
        [trajectory_speed_targets(row["candidate_traj"]) for row in train_rows]
    )
    validation_targets = np.stack(
        [trajectory_speed_targets(row["candidate_traj"]) for row in validation_rows]
    )
    model = RidgeSpeedHead.fit(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
    )
    output = Path(args.output)
    model.save(output)
    train_prediction = model.predict(train_features)
    validation_prediction = model.predict(validation_features)
    metadata = {
        "format": "iac_candidate_blind_flow_speed_head_v1",
        "method": args.method,
        "width": args.width,
        "height": args.height,
        "minimum_positive_label": args.min_label,
        "feature_dim": int(train_features.shape[1]),
        "speed_targets": list(SPEED_NAMES),
        "alpha": model.alpha,
        "counts": {
            "train_positive_rows": len(train_rows),
            "validation_positive_rows": len(validation_rows),
            "train_unique_visuals": len(set(train_sequences)),
            "validation_unique_visuals": len(set(validation_sequences)),
        },
        "train_metrics": regression_metrics(train_targets, train_prediction),
        "validation_metrics": regression_metrics(
            validation_targets, validation_prediction
        ),
    }
    write_json(metadata_path(output), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def load_extractor_settings(
    model_path: Path, args: argparse.Namespace
) -> Tuple[str, int, int]:
    path = metadata_path(model_path)
    if path.exists():
        metadata = json.loads(path.read_text(encoding="utf-8"))
    else:
        metadata = {}
    method = args.method or str(metadata.get("method", "dis"))
    width = args.width or int(metadata.get("width", 256))
    height = args.height or int(metadata.get("height", 144))
    return method, width, height


def run_apply(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.index))
    model_path = Path(args.model)
    model = RidgeSpeedHead.load(model_path)
    method, width, height = load_extractor_settings(model_path, args)
    sequences = [visual_sequence(row, args.image_root) for row in rows]
    features = extract_features(
        sequences,
        method=method,
        width=width,
        height=height,
        workers=args.workers,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    prediction = model.predict(features)
    candidate_targets = np.stack(
        [trajectory_speed_targets(row["candidate_traj"]) for row in rows]
    )
    energies = speed_energy(prediction, candidate_targets, model)
    output_rows = []
    for row, visual, target, energy in zip(
        rows, prediction, candidate_targets, energies
    ):
        value = dict(row)
        value["flow_speed_prediction"] = [float(item) for item in visual]
        value["flow_speed_candidate"] = [float(item) for item in target]
        value["flow_speed_energy"] = float(energy)
        output_rows.append(value)
    write_jsonl(Path(args.output), output_rows)
    summary = {
        "rows": len(rows),
        "unique_visuals": len(set(sequences)),
        "method": method,
        "resolution": [width, height],
        "flow_speed_energy_mean": float(np.mean(energies)),
        "flow_speed_energy_std": float(np.std(energies)),
        "output": str(Path(args.output)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="fit a visual-only speed predictor")
    fit.add_argument("--train-index", required=True)
    fit.add_argument("--val-index", required=True)
    fit.add_argument("--image-root", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--method", choices=("dis", "farneback"), default="dis")
    fit.add_argument("--width", type=int, default=256)
    fit.add_argument("--height", type=int, default=144)
    fit.add_argument("--workers", type=int, default=4)
    fit.add_argument("--cache-dir", default="")
    fit.add_argument("--min-label", type=float, default=0.999)
    fit.add_argument("--max-train-positives", type=int, default=0)
    fit.add_argument("--max-val-positives", type=int, default=0)
    fit.add_argument("--seed", type=int, default=42)
    fit.set_defaults(handler=run_fit)

    apply = subparsers.add_parser("apply", help="append flow energy to an index")
    apply.add_argument("--index", required=True)
    apply.add_argument("--image-root", required=True)
    apply.add_argument("--model", required=True)
    apply.add_argument("--output", required=True)
    apply.add_argument("--method", choices=("dis", "farneback"), default=None)
    apply.add_argument("--width", type=int, default=None)
    apply.add_argument("--height", type=int, default=None)
    apply.add_argument("--workers", type=int, default=4)
    apply.add_argument("--cache-dir", default="")
    apply.set_defaults(handler=run_apply)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

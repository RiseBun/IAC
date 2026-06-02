#!/usr/bin/env python3
"""Difficulty-stratified sampler for IAC training (iWorld-Bench §3.2.2).

iWorld-Bench designs 4 difficulty levels D1..D4 based on how many
independent perturbation axes are applied simultaneously. We mirror
that idea for the IAC critic's training data:

  D1 (single-axis): one perturbation, easy negatives
        - image_swap, traj_swap, perturb_lateral, perturb_heading,
          perturb_speed, time_shift_future, reverse_traj
  D2 (2-axis):      two perturbations composed
        - perturb_lateral+heading, perturb_heading+speed,
          perturb_lateral+speed, traj_swap+perturb_lateral, ...
  D3 (3-axis):      three perturbations composed
  D4 (adversarial): all 4 axes + reverse

The sampler groups negative samples by their difficulty bucket and
yields batches that respect a configurable mix ratio (default: 30%
D1, 30% D2, 25% D3, 15% D4, 0% positives mixed at 25% of total).

Why:
  - In v4 we observed `perturb_speed` and `time_shift_future` had the
    lowest negative recall (0.79/0.81) while `perturb_lateral` was
    near-perfect (0.99). The critic was probably seeing a
    lateral-heavy stream and overfitting to it. Stratification
    rebalances the gradient.
  - Composing D2/D3/D4 negatives at training time creates 'compositional
    robustness' that D1 alone can't teach.

This file is *consumed* by train.py. The sampler is wrapped on top of
the existing DistributedSampler.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from torch.utils.data import Sampler

from train import ConsistencyDataset


# ───────────────────────── Difficulty assignment ─────────────────────────


# Map each source_type to a baseline difficulty axis count.
_BASE_AXES: Dict[str, int] = {
    "gt_pos": 0,
    "image_swap": 1,
    "traj_swap": 1,
    "time_shift_future": 1,
    "reverse_traj": 1,
    "perturb_lateral": 1,
    "perturb_heading": 1,
    "perturb_speed": 1,
    # Compositional
    "perturb_lateral+heading": 2,
    "perturb_lateral+speed": 2,
    "perturb_heading+speed": 2,
    "perturb_lateral+heading+speed": 3,
    "all_perturb+reverse": 4,
}


def assign_difficulty(sample: dict) -> int:
    """Return a difficulty bucket 1..4 for a sample.

    Composed negatives carry the explicit D in their source_type; if not
    present we fall back to the source_type's base axis count.
    """
    st = sample.get("source_type", "unknown")
    if sample.get("consistency_label", 1) == 1:
        return 0  # positives, not part of difficulty mix
    if st in _BASE_AXES:
        return max(1, _BASE_AXES[st])
    # Default unknown negatives → D1 (safest)
    return 1


def bucket_samples(samples: List[dict]) -> Dict[int, List[int]]:
    """Group sample indices by difficulty bucket. Returns {0..4: [indices]}."""
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        buckets[assign_difficulty(s)].append(i)
    return buckets


# ───────────────────────── Sampler ─────────────────────────


class DifficultyStratifiedSampler(Sampler[int]):
    """Yield sample indices respecting a D1..D4 mix ratio.

    A new epoch = a fixed total of `num_samples_per_epoch` indices,
    drawn with replacement according to the configured mix.

    Positives (D0) are added at fixed ratio (default 25% of total).
    """

    def __init__(
        self,
        samples: List[dict],
        num_samples_per_epoch: int,
        mix: Tuple[float, float, float, float] = (0.30, 0.30, 0.25, 0.15),
        positive_ratio: float = 0.25,
        seed: int = 0,
    ) -> None:
        if abs(sum(mix) - 1.0) > 1e-3:
            raise ValueError(f"mix must sum to 1.0, got {sum(mix)}")
        self.samples = samples
        self.num_samples = int(num_samples_per_epoch)
        self.mix = mix
        self.positive_ratio = float(positive_ratio)
        self.seed = int(seed)
        self.buckets = bucket_samples(samples)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        positives = self.buckets.get(0, [])
        d1, d2, d3, d4 = (
            self.buckets.get(1, []),
            self.buckets.get(2, []),
            self.buckets.get(3, []),
            self.buckets.get(4, []),
        )
        # Number of positive indices
        n_pos = int(self.num_samples * self.positive_ratio)
        n_neg_total = self.num_samples - n_pos
        # Split negatives by D1..D4 mix
        per_d = [int(round(n_neg_total * m)) for m in self.mix]
        # Adjust rounding error
        per_d[0] += n_neg_total - sum(per_d)

        out: List[int] = []
        for n, bucket in zip([n_pos, *per_d], [positives, d1, d2, d3, d4]):
            if not bucket or n <= 0:
                # If a bucket is empty (e.g. no D4 yet), borrow from D1
                if n > 0 and d1:
                    out.extend(rng.choices(d1, k=n))
                continue
            out.extend(rng.choices(bucket, k=n))
        rng.shuffle(out)
        yield from out[: self.num_samples]


# ───────────────────────── D2/D3/D4 synthetic composers (for training-only) ─────────────────────────


def compose_perturb_negatives(
    anchors_samples: List[dict],
    rng: random.Random,
    lateral_range: Tuple[float, float] = (0.5, 2.0),
    heading_range: Tuple[float, float] = (5.0, 15.0),
    speed_range: Tuple[float, float] = (0.7, 1.3),
    time_shift_steps: int = 2,
    targets: Tuple[str, ...] = (
        "perturb_lateral+heading",
        "perturb_lateral+speed",
        "perturb_heading+speed",
        "perturb_lateral+heading+speed",
    ),
) -> List[dict]:
    """Generate composed D2/D3 negatives by combining two or more
    single-axis perturbations on the same anchor.

    Returns new sample dicts that are then merged into the train JSONL
    at build time. (The Composition logic reuses the *GT trajectory*
    from the anchor and the *GT future_images*; the perturbations only
    modify the candidate_traj, so consistency_label=0.)
    """
    from build_consistency_index import perturb_trajectory  # type: ignore

    out: List[dict] = []
    axis_map = {
        "lateral": ("lateral", lateral_range),
        "heading": ("heading", heading_range),
        "speed": ("speed", speed_range),
    }
    # perturb_trajectory lives in tools/build_consistency_index.py.
    # We add the tools/ directory to sys.path so the import works
    # whether the sampler is invoked from the project root or from
    # inside tools/ (e.g. by build_consistency_index.py itself).
    import sys
    from pathlib import Path
    _TOOLS = Path(__file__).resolve().parent / "tools"
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    try:
        from build_consistency_index import perturb_trajectory  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Could not import perturb_trajectory from tools/build_consistency_index.py: {e}. "
            "Make sure IAC/ is on sys.path and tools/ contains build_consistency_index.py."
        ) from e

    out: List[dict] = []
    axis_map = {
        "lateral": ("lateral", lateral_range),
        "heading": ("heading", heading_range),
        "speed": ("speed", speed_range),
    }
    for anchor in anchors_samples:
        base_traj = anchor["candidate_traj"]
        ego_state = anchor["ego_state"]
        for target in targets:
            axes = [a.strip() for a in target.replace("perturb_", "").split("+")]
            if not all(a in axis_map for a in axes):
                continue
            new_traj = [list(pt) for pt in base_traj]
            magnitude = 0.0
            for axis in axes:
                kind, rng_range = axis_map[axis]
                new_traj, mag = perturb_trajectory(
                    new_traj, kind, rng, lateral_range, heading_range, speed_range
                )
                magnitude += mag
            row = dict(anchor)
            row["sample_id"] = f"{anchor['sample_id']}__{target}"
            row["candidate_traj"] = new_traj
            row["consistency_label"] = 0
            row["source_type"] = target
            row["negative_family"] = "composed"
            row["perturb_magnitude"] = magnitude
            # validity recomputed downstream by with_validity()
            row["validity_label"] = 0
            row["validity_reason"] = "composed_perturb"
            out.append(row)
    return out


__all__ = [
    "DifficultyStratifiedSampler",
    "assign_difficulty",
    "bucket_samples",
    "compose_perturb_negatives",
]

"""Candidate-blind optical-flow evidence for IAC trajectory ranking.

The extractor sees only the history/future image sequence.  A small ridge
model maps the resulting flow statistics to six interpretable speed targets.
Candidate trajectories are introduced only by :func:`speed_energy`, after the
visual prediction has been made.  This mirrors the anti-shortcut boundary used
by the DINO motion head.

The implementation is intentionally classical and dependency-light: DIS was
the method retained by our validation gate; Farneback remains available as a
sanity-check baseline.  This module does not claim that optical flow replaces
DINO or IAC's learned scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, Union

import cv2
import numpy as np


PathLike = Union[str, Path]
SPEED_NAMES: Tuple[str, ...] = (
    "speed_h25_mps",
    "speed_h50_mps",
    "speed_h75_mps",
    "speed_h100_mps",
    "mean_speed_mps",
    "delta_speed_mps",
)


def _pair_feature_names() -> Tuple[str, ...]:
    names = [
        "median_u",
        "median_v",
        "median_magnitude",
        "q75_magnitude",
        "q90_magnitude",
        "std_magnitude",
        "median_radial",
        "q75_radial",
        "median_abs_radial",
        "median_divergence",
        "q75_abs_divergence",
        "affine_u_bias",
        "affine_u_x",
        "affine_u_y",
        "affine_v_bias",
        "affine_v_x",
        "affine_v_y",
        "valid_fraction",
    ]
    for row in range(3):
        for column in range(3):
            for value in ("u", "v", "magnitude", "radial"):
                names.append(f"grid_{row}_{column}_{value}")
    return tuple(names)


PAIR_FEATURE_NAMES = _pair_feature_names()
PAIR_FEATURE_DIM = len(PAIR_FEATURE_NAMES)


def flow_statistics(flow: np.ndarray) -> np.ndarray:
    """Summarize dense flow with robust global and 3x3 road-grid statistics."""

    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"expected flow shape (H,W,2), got {flow.shape}")
    height, width = flow.shape[:2]
    u = flow[..., 0].astype(np.float32, copy=False)
    v = flow[..., 1].astype(np.float32, copy=False)
    yy, xx = np.mgrid[0:height, 0:width]
    xn = (xx.astype(np.float32) - width * 0.5) / max(float(width), 1.0)
    yn = (yy.astype(np.float32) - height * 0.42) / max(float(height), 1.0)
    magnitude = np.sqrt(u * u + v * v)
    radius = np.sqrt(xn * xn + yn * yn) + 1e-4
    radial = (u * xn + v * yn) / radius
    divergence = np.gradient(u, axis=1) + np.gradient(v, axis=0)
    roi = yy >= int(round(height * 0.25))
    finite = roi & np.isfinite(u) & np.isfinite(v) & np.isfinite(magnitude)
    if int(finite.sum()) < 64:
        values = np.full(PAIR_FEATURE_DIM, np.nan, dtype=np.float32)
        values[17] = 0.0
        return values

    cutoff = float(np.quantile(magnitude[finite], 0.997))
    finite &= magnitude <= max(cutoff, 1e-4)

    def median(array: np.ndarray, mask: np.ndarray = finite) -> float:
        selected = array[mask]
        return float(np.median(selected)) if selected.size else float("nan")

    def quantile(array: np.ndarray, value: float, mask: np.ndarray = finite) -> float:
        selected = array[mask]
        return float(np.quantile(selected, value)) if selected.size else float("nan")

    sample = finite & ((xx % 8) == 0) & ((yy % 8) == 0)
    design = np.stack(
        [np.ones_like(xn[sample]), xn[sample], yn[sample]], axis=1
    )
    try:
        coefficient_u = np.linalg.lstsq(design, u[sample], rcond=None)[0]
        coefficient_v = np.linalg.lstsq(design, v[sample], rcond=None)[0]
    except np.linalg.LinAlgError:
        coefficient_u = np.full(3, np.nan, dtype=np.float32)
        coefficient_v = np.full(3, np.nan, dtype=np.float32)

    values = [
        median(u),
        median(v),
        median(magnitude),
        quantile(magnitude, 0.75),
        quantile(magnitude, 0.90),
        float(np.std(magnitude[finite])),
        median(radial),
        quantile(radial, 0.75),
        median(np.abs(radial)),
        median(divergence),
        quantile(np.abs(divergence), 0.75),
        *[float(value) for value in coefficient_u],
        *[float(value) for value in coefficient_v],
        float(finite.sum() / max(int(roi.sum()), 1)),
    ]
    y_edges = np.linspace(int(round(height * 0.25)), height, 4, dtype=int)
    x_edges = np.linspace(0, width, 4, dtype=int)
    for row in range(3):
        for column in range(3):
            cell = finite.copy()
            cell &= yy >= y_edges[row]
            cell &= yy < y_edges[row + 1]
            cell &= xx >= x_edges[column]
            cell &= xx < x_edges[column + 1]
            values.extend(
                [
                    median(u, cell),
                    median(v, cell),
                    median(magnitude, cell),
                    median(radial, cell),
                ]
            )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (PAIR_FEATURE_DIM,):
        raise AssertionError(f"flow feature contract changed: {result.shape}")
    return result


class ClassicFlowExtractor:
    """Extract DIS or Farneback evidence from adjacent image frames."""

    def __init__(
        self,
        method: str = "dis",
        *,
        width: int = 256,
        height: int = 144,
    ) -> None:
        if method not in {"dis", "farneback"}:
            raise ValueError("method must be 'dis' or 'farneback'")
        if width < 16 or height < 16:
            raise ValueError("flow resolution is too small")
        self.method = method
        self.width = int(width)
        self.height = int(height)
        self._dis = None
        if method == "dis":
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            self._dis.setFinestScale(2)

    def _read_gray(self, path: PathLike) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(str(path))
        return cv2.resize(
            image,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

    def pair_features(self, first: PathLike, second: PathLike) -> np.ndarray:
        first_image = self._read_gray(first)
        second_image = self._read_gray(second)
        if self.method == "dis":
            assert self._dis is not None
            flow = self._dis.calc(first_image, second_image, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                first_image,
                second_image,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
        return flow_statistics(flow)

    def sequence_features(self, frames: Sequence[PathLike]) -> np.ndarray:
        """Return flattened pairs plus mean/std/change sequence summaries.

        IAC/NAVSIM normally supplies four history and four future frames.  The
        method supports any sequence with at least two frames; model metadata
        records the resulting feature dimension so incompatible inputs fail
        clearly at prediction time.
        """

        if len(frames) < 2:
            raise ValueError("at least two frames are required")
        pair_values = np.stack(
            [
                self.pair_features(first, second)
                for first, second in zip(frames[:-1], frames[1:])
            ]
        )
        return np.concatenate(
            [
                pair_values.reshape(-1),
                np.nanmean(pair_values, axis=0),
                np.nanstd(pair_values, axis=0),
                pair_values[-1] - pair_values[0],
            ]
        ).astype(np.float32)


def visual_sequence(
    row: Mapping[str, object], image_root: PathLike = "."
) -> Tuple[str, ...]:
    """Resolve the ordered history+future frame paths of one IAC index row."""

    root = Path(image_root)
    raw_values = list(row.get("history_images", [])) + list(
        row.get("future_images", [])
    )
    if len(raw_values) < 2:
        raise ValueError("index row does not contain a usable visual sequence")
    values = []
    for raw in raw_values:
        path = Path(str(raw))
        if not path.is_absolute():
            path = root / path
        values.append(str(path))
    return tuple(values)


def trajectory_speed_targets(
    candidate_trajectory: Union[np.ndarray, Sequence[Sequence[float]]],
    *,
    step_time: float = 0.5,
) -> np.ndarray:
    """Return 25/50/75/100%-horizon, mean and delta speeds in m/s."""

    trajectory = np.asarray(candidate_trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or len(trajectory) < 4 or trajectory.shape[1] < 2:
        raise ValueError(f"invalid candidate trajectory shape {trajectory.shape}")
    if step_time <= 0:
        raise ValueError("step_time must be positive")
    indices = sorted(
        {
            min(len(trajectory) - 1, max(0, round(len(trajectory) * fraction) - 1))
            for fraction in (0.25, 0.50, 0.75, 1.00)
        }
    )
    if len(indices) != 4:
        indices = list(np.linspace(0, len(trajectory) - 1, 4, dtype=int))
    xy = trajectory[:, :2]
    increments = np.linalg.norm(
        np.diff(np.vstack([np.zeros((1, 2)), xy]), axis=0), axis=1
    )
    cumulative = np.cumsum(increments)
    segment_speeds = []
    previous_index = -1
    previous_path = 0.0
    for index in indices:
        elapsed = (index - previous_index) * step_time
        path = float(cumulative[index])
        segment_speeds.append((path - previous_path) / max(elapsed, 1e-6))
        previous_index = index
        previous_path = path
    mean_speed = float(cumulative[indices[-1]]) / ((indices[-1] + 1) * step_time)
    delta_speed = float(segment_speeds[-1] - segment_speeds[0])
    return np.asarray([*segment_speeds, mean_speed, delta_speed], dtype=np.float32)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2:
        return float("nan")
    target_rank = _rankdata(np.asarray(target, dtype=np.float64))
    prediction_rank = _rankdata(np.asarray(prediction, dtype=np.float64))
    if target_rank.std() < 1e-12 or prediction_rank.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(target_rank, prediction_rank)[0, 1])


@dataclass
class RidgeSpeedHead:
    """Serializable multi-output ridge model for visual speed prediction."""

    impute: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    weights: np.ndarray
    sigma: np.ndarray
    alpha: float

    @classmethod
    def fit(
        cls,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
        *,
        alphas: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    ) -> "RidgeSpeedHead":
        train_x = np.asarray(train_features, dtype=np.float64)
        validation_x = np.asarray(validation_features, dtype=np.float64)
        train_y = np.asarray(train_targets, dtype=np.float64)
        validation_y = np.asarray(validation_targets, dtype=np.float64)
        if train_x.ndim != 2 or validation_x.ndim != 2:
            raise ValueError("features must be two-dimensional")
        if train_y.ndim != 2 or train_y.shape[1] != len(SPEED_NAMES):
            raise ValueError(f"targets must have shape (N,{len(SPEED_NAMES)})")
        if train_x.shape[1] != validation_x.shape[1]:
            raise ValueError("train/validation feature dimensions differ")
        if not len(train_x) or not len(validation_x):
            raise ValueError("train and validation splits must be non-empty")

        with np.errstate(all="ignore"):
            impute = np.nanmedian(train_x, axis=0)
        impute[~np.isfinite(impute)] = 0.0
        train_x = np.where(np.isfinite(train_x), train_x, impute)
        validation_x = np.where(np.isfinite(validation_x), validation_x, impute)
        feature_mean = train_x.mean(axis=0)
        feature_std = np.maximum(train_x.std(axis=0), 1e-4)
        target_mean = train_y.mean(axis=0)
        target_std = np.maximum(train_y.std(axis=0), 1e-3)
        normalized_x = (train_x - feature_mean) / feature_std
        normalized_validation_x = (validation_x - feature_mean) / feature_std
        normalized_y = (train_y - target_mean) / target_std
        gram = normalized_x.T @ normalized_x
        cross = normalized_x.T @ normalized_y
        identity = np.eye(normalized_x.shape[1], dtype=np.float64)

        best_key = (-float("inf"), -float("inf"))
        best_weights = None
        best_alpha = None
        for alpha in alphas:
            alpha = float(alpha)
            if alpha <= 0:
                raise ValueError("ridge alphas must be positive")
            weights = np.linalg.solve(gram + alpha * identity, cross)
            prediction = (
                normalized_validation_x @ weights * target_std + target_mean
            )
            correlations = [
                spearman_correlation(validation_y[:, index], prediction[:, index])
                for index in range(len(SPEED_NAMES))
            ]
            finite_correlations = [value for value in correlations if np.isfinite(value)]
            mean_correlation = (
                float(np.mean(finite_correlations)) if finite_correlations else -1.0
            )
            rmse = float(np.sqrt(np.mean(np.square(validation_y - prediction))))
            key = (mean_correlation, -rmse)
            if key > best_key:
                best_key = key
                best_weights = weights
                best_alpha = alpha
        if best_weights is None or best_alpha is None:
            raise RuntimeError("ridge selection produced no valid model")

        train_prediction_normalized = normalized_x @ best_weights
        sigma = np.maximum(
            np.sqrt(np.mean(np.square(normalized_y - train_prediction_normalized), axis=0)),
            0.10,
        )
        return cls(
            impute=impute,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            weights=best_weights,
            sigma=sigma,
            alpha=best_alpha,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        one_sample = values.ndim == 1
        if one_sample:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError(
                f"expected feature width {len(self.feature_mean)}, got {values.shape}"
            )
        values = np.where(np.isfinite(values), values, self.impute)
        normalized = (values - self.feature_mean) / self.feature_std
        prediction = normalized @ self.weights * self.target_std + self.target_mean
        result = prediction.astype(np.float32)
        return result[0] if one_sample else result

    def save(self, path: PathLike) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            impute=self.impute,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            weights=self.weights,
            sigma=self.sigma,
            alpha=np.asarray(self.alpha),
        )

    @classmethod
    def load(cls, path: PathLike) -> "RidgeSpeedHead":
        with np.load(path) as payload:
            return cls(
                impute=payload["impute"],
                feature_mean=payload["feature_mean"],
                feature_std=payload["feature_std"],
                target_mean=payload["target_mean"],
                target_std=payload["target_std"],
                weights=payload["weights"],
                sigma=payload["sigma"],
                alpha=float(payload["alpha"]),
            )


def speed_energy(
    visual_prediction: np.ndarray,
    candidate_targets: np.ndarray,
    model: RidgeSpeedHead,
) -> np.ndarray:
    """Convert visual/candidate speed disagreement into lower-is-better energy."""

    prediction = np.asarray(visual_prediction, dtype=np.float64)
    candidate = np.asarray(candidate_targets, dtype=np.float64)
    if prediction.shape != candidate.shape or prediction.shape[-1] != len(SPEED_NAMES):
        raise ValueError("visual prediction and candidate speed targets must match")
    prediction_normalized = (prediction - model.target_mean) / model.target_std
    candidate_normalized = (candidate - model.target_mean) / model.target_std
    residual = (candidate_normalized - prediction_normalized) / model.sigma
    return np.mean(0.5 * np.square(residual) + np.log(model.sigma), axis=-1)

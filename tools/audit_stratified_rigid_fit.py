#!/usr/bin/env python3
"""Fit the best plane-rigid trajectory on fixed, spatially stratified flow evidence.

This is an audit, not a production decoder variant.  It deliberately changes
two measurement choices that confounded the released decoder:

* pixels are sampled across fixed image-space strata, independently of a
  candidate trajectory; and
* every selected source point stays in the denominator.  A candidate that
  projects a source point out of view receives the robust ceiling (4.0).

The optimizer uses a broad constant-speed/constant-curvature grid followed by
piecewise speed/curvature coordinate descent from the best grid seeds.  Its
primary comparison is the fitted flow energy against the zero-flow energy on
exactly the same pixels and weights.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_EVALUATOR_PATH = _ROOT / "scripts" / "evaluate_continuous_decoder.py"
_SPEC = importlib.util.spec_from_file_location("iac_evaluate_continuous_decoder", _EVALUATOR_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"unable to load evaluator from {_EVALUATOR_PATH}")
evaluator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluator)

import iac_new.trajectory_decode as trajectory_decode


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_pixel_key(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Deterministically break equal-weight ties without raster-order bias."""
    x = np.asarray(xx, dtype=np.uint64)
    y = np.asarray(yy, dtype=np.uint64)
    value = (x * np.uint64(0x9E3779B1)) ^ (y * np.uint64(0x85EBCA77))
    value ^= value >> np.uint64(16)
    return value


def _stratified_sample_pixels(
    observed_flows: np.ndarray,
    support_weights: np.ndarray,
    *,
    max_points: int,
    **_production_sampling_options: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select fixed evidence across 3 vertical by 6 horizontal strata."""
    observed = np.asarray(observed_flows, dtype=np.float32)
    weights = np.asarray(support_weights, dtype=np.float32)
    if observed.ndim != 4 or observed.shape[-1] != 2:
        raise ValueError("observed_flows must have shape [T,H,W,2]")
    if weights.shape != observed.shape[:-1]:
        raise ValueError("support_weights must have shape [T,H,W]")
    valid = np.isfinite(observed).all(axis=-1) & np.isfinite(weights) & (weights > 0.0)
    valid &= np.linalg.norm(observed, axis=-1) > 0.05
    eligible = valid.any(axis=0)
    height, width = eligible.shape
    yy, xx = np.nonzero(eligible)
    if len(xx) == 0:
        raise ValueError("no finite motion pixels are available for trajectory decoding")

    v = yy / max(height - 1, 1)
    u = xx / max(width - 1, 1)
    in_band = (v >= 0.55) & (v < 0.85)
    yy, xx, v, u = yy[in_band], xx[in_band], v[in_band], u[in_band]
    if len(xx) == 0:
        raise ValueError("no finite motion pixels are available in v=[0.55,0.85)")

    score_image = np.mean(np.where(valid, weights, 0.0), axis=0)
    scores = score_image[yy, xx]
    keys = _stable_pixel_key(xx, yy)
    row_bin = np.minimum(((v - 0.55) / 0.10).astype(np.int64), 2)
    col_bin = np.minimum((u * 6.0).astype(np.int64), 5)
    cell = row_bin * 6 + col_bin
    quota = int(np.ceil(int(max_points) / 18.0))
    selected_parts: list[np.ndarray] = []
    for cell_index in range(18):
        candidates = np.flatnonzero(cell == cell_index)
        if len(candidates) == 0:
            continue
        order = np.lexsort((keys[candidates], -scores[candidates]))
        selected_parts.append(candidates[order[:quota]])
    selected = (
        np.concatenate(selected_parts)
        if selected_parts
        else np.empty(0, dtype=np.int64)
    )
    if len(selected) > int(max_points):
        order = np.lexsort((keys[selected], -scores[selected]))
        selected = selected[order[: int(max_points)]]
    elif len(selected) < int(max_points):
        already = np.zeros(len(xx), dtype=bool)
        already[selected] = True
        remaining = np.flatnonzero(~already)
        order = np.lexsort((keys[remaining], -scores[remaining]))
        need = int(max_points) - len(selected)
        selected = np.concatenate([selected, remaining[order[:need]]])

    chosen_x = xx[selected]
    chosen_y = yy[selected]
    coords = np.stack([chosen_x, chosen_y], axis=1).astype(np.float64)
    return (
        coords,
        observed[:, chosen_y, chosen_x, :].astype(np.float64),
        weights[:, chosen_y, chosen_x].astype(np.float64),
    )


def _fixed_source_objective(
    trajectory: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    pixel_xy: np.ndarray,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    depths_m: np.ndarray | None,
    minimum_flow_scale_px: float,
    road_masks: np.ndarray | None = None,
    road_prior_weight: float = 0.0,
    road_half_width_m: float = 1.1,
    road_lateral_samples: int = 5,
    road_longitudinal_step_m: float = 0.5,
    speed_smoothness_weight: float = 0.0,
    curvature_smoothness_weight: float = 0.0,
    lateral_acceleration_weight: float = 0.0,
    future_times_s: np.ndarray | None = None,
    adaptive_plane_params: np.ndarray | None = None,
    minimum_projection_points: int = 30,
    minimum_projection_weight_fraction: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    predicted, valid = trajectory_decode._sparse_predicted_flows(
        trajectory,
        camera_to_ego,
        intrinsics,
        pixel_xy,
        image_size=image_size,
        depths_m=depths_m,
        adaptive_plane_params=adaptive_plane_params,
    )
    source = np.isfinite(observed).all(axis=-1) & np.isfinite(weights) & (weights > 0.0)
    projected = valid & np.isfinite(predicted).all(axis=-1)
    usable = source & projected
    residual = np.zeros(source.shape, dtype=np.float64)
    residual[usable] = np.linalg.norm(predicted[usable] - observed[usable], axis=-1)
    scale = np.maximum(np.linalg.norm(observed, axis=-1), float(minimum_flow_scale_px))
    robust = np.full(source.shape, 4.0, dtype=np.float64)
    robust[usable] = np.minimum(residual[usable] / scale[usable], 4.0)
    denominator = float(np.where(source, weights, 0.0).sum())
    flow_energy = (
        float(np.where(source, robust * weights, 0.0).sum() / denominator)
        if denominator > 1e-6
        else 4.0
    )
    support = trajectory_decode._projection_support(
        observed,
        predicted,
        valid,
        weights,
        minimum_points=minimum_projection_points,
        minimum_weight_fraction=minimum_projection_weight_fraction,
    )
    road_penalty = (
        trajectory_decode._road_prior_penalty(
            trajectory,
            road_masks,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            image_size=image_size,
            half_width_m=road_half_width_m,
            lateral_samples=road_lateral_samples,
            longitudinal_step_m=road_longitudinal_step_m,
        )
        if road_masks is not None and road_prior_weight > 0.0
        else 0.0
    )
    smoothness_penalty = (
        trajectory_decode._kinematic_smoothness_penalty(
            trajectory,
            np.arange(1, len(trajectory) + 1, dtype=np.float64)
            if future_times_s is None
            else np.asarray(future_times_s, dtype=np.float64),
            speed_weight=speed_smoothness_weight,
            curvature_weight=curvature_smoothness_weight,
            lateral_acceleration_weight=lateral_acceleration_weight,
        )
        if any(
            value > 0.0
            for value in (
                speed_smoothness_weight,
                curvature_smoothness_weight,
                lateral_acceleration_weight,
            )
        )
        else 0.0
    )
    return (
        flow_energy + float(road_prior_weight) * road_penalty + smoothness_penalty,
        predicted,
        valid,
        support,
    )


def _flow_energy_rows(
    trajectory: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    pixel_xy: np.ndarray,
    *,
    camera_to_ego: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    depths_m: np.ndarray | None,
    adaptive_plane_params: np.ndarray | None,
    minimum_flow_scale_px: float,
) -> dict[str, Any]:
    predicted, valid = trajectory_decode._sparse_predicted_flows(
        trajectory,
        camera_to_ego,
        intrinsics,
        pixel_xy,
        image_size=image_size,
        depths_m=depths_m,
        adaptive_plane_params=adaptive_plane_params,
    )
    source = np.isfinite(observed).all(axis=-1) & np.isfinite(weights) & (weights > 0.0)
    projected = valid & np.isfinite(predicted).all(axis=-1)
    usable = source & projected
    scale = np.maximum(np.linalg.norm(observed, axis=-1), float(minimum_flow_scale_px))
    fitted_robust = np.full(source.shape, 4.0, dtype=np.float64)
    fitted_robust[usable] = np.minimum(
        np.linalg.norm(predicted[usable] - observed[usable], axis=-1) / scale[usable],
        4.0,
    )
    zero_robust = np.minimum(np.linalg.norm(observed, axis=-1) / scale, 4.0)
    rows = []
    for index in range(len(observed)):
        denominator = float(np.where(source[index], weights[index], 0.0).sum())
        fitted = (
            float(np.where(source[index], fitted_robust[index] * weights[index], 0.0).sum() / denominator)
            if denominator > 1e-6
            else None
        )
        zero = (
            float(np.where(source[index], zero_robust[index] * weights[index], 0.0).sum() / denominator)
            if denominator > 1e-6
            else None
        )
        projected_weight = float(np.where(source[index] & projected[index], weights[index], 0.0).sum())
        rows.append({
            "interval_index": index,
            "source_points": int(source[index].sum()),
            "projected_points": int((source[index] & projected[index]).sum()),
            "source_weight": denominator,
            "projected_weight_fraction": projected_weight / denominator if denominator > 1e-6 else 0.0,
            "fitted_energy": fitted,
            "zero_flow_energy": zero,
            "delta_fitted_minus_zero": None if fitted is None else fitted - zero,
        })
    total_weight = float(np.where(source, weights, 0.0).sum())
    fitted_total = float(np.where(source, fitted_robust * weights, 0.0).sum() / total_weight)
    zero_total = float(np.where(source, zero_robust * weights, 0.0).sum() / total_weight)
    return {
        "fitted_energy": fitted_total,
        "zero_flow_energy": zero_total,
        "delta_fitted_minus_zero": fitted_total - zero_total,
        "fitted_better_than_zero": bool(fitted_total < zero_total),
        "by_interval": rows,
    }


def _fit_audit(kwargs: dict[str, Any], *, grid_seeds: int, local_iterations: int) -> dict[str, Any]:
    observed = np.asarray(kwargs["observed_flows"], dtype=np.float32)
    roi = np.asarray(kwargs["roi_mask"], dtype=bool)
    consistency = (
        np.ones(observed.shape[:-1], dtype=bool)
        if kwargs.get("consistency_masks") is None
        else np.asarray(kwargs["consistency_masks"], dtype=bool)
    )
    weights = (
        np.ones(observed.shape[:-1], dtype=np.float32)
        if kwargs.get("dynamic_weights") is None
        else np.asarray(kwargs["dynamic_weights"], dtype=np.float32)
    )
    weights = np.where(roi[None, ...] & consistency, np.maximum(weights, 0.0), 0.0)
    pixel_xy, sampled_observed, sampled_weights = _stratified_sample_pixels(
        observed, weights, max_points=int(kwargs["max_points"])
    )
    times = np.asarray(kwargs["future_times_s"], dtype=np.float64)
    speed_grid = np.asarray([0.2, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0])
    curvature_grid = np.asarray([-0.30, -0.20, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.20, 0.30])
    candidates: list[tuple[float, float, float]] = []
    for speed in speed_grid:
        for curvature in curvature_grid:
            trajectory = trajectory_decode.integrate_piecewise_controls(
                times,
                speeds_mps=np.full(len(times), speed, dtype=np.float64),
                curvatures_1pm=np.full(len(times), curvature, dtype=np.float64),
            )
            energy = _fixed_source_objective(
                trajectory,
                sampled_observed,
                sampled_weights,
                pixel_xy,
                kwargs["camera_to_ego"],
                kwargs["intrinsics"],
                kwargs["image_size"],
                kwargs.get("depths_m"),
                float(kwargs["minimum_flow_scale_px"]),
                adaptive_plane_params=kwargs.get("adaptive_plane_params"),
            )[0]
            candidates.append((float(energy), float(speed), float(curvature)))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    best_trajectory = np.zeros((len(times), 3), dtype=np.float64)
    best_energy = _fixed_source_objective(
        best_trajectory,
        sampled_observed,
        sampled_weights,
        pixel_xy,
        kwargs["camera_to_ego"],
        kwargs["intrinsics"],
        kwargs["image_size"],
        kwargs.get("depths_m"),
        float(kwargs["minimum_flow_scale_px"]),
        adaptive_plane_params=kwargs.get("adaptive_plane_params"),
    )[0]
    starts = candidates[: int(grid_seeds)]
    for _, speed, curvature in starts:
        trajectory, energy = trajectory_decode._fit_once(
            future_times_s=times,
            observed=sampled_observed,
            weights=sampled_weights,
            pixel_xy=pixel_xy,
            camera_to_ego=kwargs["camera_to_ego"],
            intrinsics=kwargs["intrinsics"],
            image_size=kwargs["image_size"],
            depths_m=kwargs.get("depths_m"),
            minimum_flow_scale_px=float(kwargs["minimum_flow_scale_px"]),
            initial_speed_mps=speed,
            max_iterations=int(local_iterations),
            initial_curvatures_1pm=np.full(len(times), curvature, dtype=np.float64),
            adaptive_plane_params=kwargs.get("adaptive_plane_params"),
        )
        if energy < best_energy:
            best_trajectory = trajectory
            best_energy = energy
    metrics = _flow_energy_rows(
        best_trajectory,
        sampled_observed,
        sampled_weights,
        pixel_xy,
        camera_to_ego=kwargs["camera_to_ego"],
        intrinsics=kwargs["intrinsics"],
        image_size=kwargs["image_size"],
        depths_m=kwargs.get("depths_m"),
        adaptive_plane_params=kwargs.get("adaptive_plane_params"),
        minimum_flow_scale_px=float(kwargs["minimum_flow_scale_px"]),
    )
    height, width = observed.shape[1:3]
    v = pixel_xy[:, 1] / max(height - 1, 1)
    u = pixel_xy[:, 0] / max(width - 1, 1)
    return {
        "protocol": "fixed-source-spatially-stratified-rigid-fit-v1",
        "effective_intrinsics": np.asarray(kwargs["intrinsics"], dtype=np.float64).tolist(),
        "sampling": {
            "selected_pixels": int(len(pixel_xy)),
            "v_range": [0.55, 0.85],
            "grid": [3, 6],
            "u_quantiles": [float(value) for value in np.quantile(u, [0.05, 0.5, 0.95])],
            "v_quantiles": [float(value) for value in np.quantile(v, [0.05, 0.5, 0.95])],
        },
        "search": {
            "constant_grid_candidates": len(candidates),
            "local_seeds": len(starts),
            "local_iterations": int(local_iterations),
            "best_grid_energy": float(candidates[0][0]),
            "best_grid_speed_mps": float(candidates[0][1]),
            "best_grid_curvature_1pm": float(candidates[0][2]),
        },
        "trajectory": best_trajectory.tolist(),
        **metrics,
    }


def _summary(rows: list[dict[str, Any]], *, practical_margin: float = 0.05) -> dict[str, Any]:
    values = [row["audit"] for row in rows]
    if not values:
        return {"rows": 0}
    fitted = np.asarray([row["fitted_energy"] for row in values], dtype=np.float64)
    zero = np.asarray([row["zero_flow_energy"] for row in values], dtype=np.float64)
    delta = fitted - zero
    delta[np.abs(delta) < 1e-9] = 0.0
    rng = np.random.default_rng(20260909)
    boot = np.mean(delta[rng.integers(0, len(delta), size=(2000, len(delta)))], axis=1)
    return {
        "rows": len(rows),
        "fitted_energy_mean": float(np.mean(fitted)),
        "fitted_energy_median": float(np.median(fitted)),
        "zero_flow_energy_mean": float(np.mean(zero)),
        "zero_flow_energy_median": float(np.median(zero)),
        "delta_mean": float(np.mean(delta)),
        "delta_median": float(np.median(delta)),
        "delta_mean_bootstrap_95ci": [float(value) for value in np.quantile(boot, [0.025, 0.975])],
        "practical_margin": float(practical_margin),
        "meaningfully_better_rows": int(np.sum(delta < -float(practical_margin))),
        "meaningfully_better_fraction": float(np.mean(delta < -float(practical_margin))),
        "primary_success": bool(np.quantile(boot, 0.975) < -float(practical_margin)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--grid-seeds", type=int, default=12)
    parser.add_argument("--local-iterations", type=int, default=12)
    parser.add_argument("--disable-fb", action="store_true")
    args = parser.parse_args()

    config = evaluator._json(args.config)
    if args.disable_fb:
        config["flow"]["forward_backward"] = False
    records = []
    for manifest in args.manifest:
        for raw in _read_jsonl(manifest):
            record = evaluator.validate_record(raw, manifest_root=manifest.parent)
            record["source_key"] = raw.get("source_key")
            record["branch_role"] = raw.get("branch_role")
            records.append(record)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    flow_cfg = config["flow"]
    checkpoint = flow_cfg.get("checkpoint")
    if checkpoint:
        checkpoint_path = Path(str(checkpoint))
        if not checkpoint_path.is_absolute():
            checkpoint_path = args.config.parent.parent / checkpoint_path
        checkpoint = str(checkpoint_path)
    extractor = evaluator.RaftFlowExtractor(
        model_size=str(flow_cfg["model"]),
        device=args.device,
        updates=int(flow_cfg["updates"]),
        batch_size=int(flow_cfg["batch_size"]),
        forward_backward=bool(flow_cfg["forward_backward"]),
        fb_abs_threshold_px=float(flow_cfg["fb_abs_threshold_px"]),
        fb_relative_threshold=float(flow_cfg["fb_relative_threshold"]),
        checkpoint=checkpoint,
    )
    perception = evaluator.build_perception(config, device=args.device)

    original_sample = trajectory_decode._sample_pixels
    original_objective = trajectory_decode._objective
    original_decode = evaluator.decode_continuous_trajectory
    trajectory_decode._sample_pixels = _stratified_sample_pixels
    trajectory_decode._objective = _fixed_source_objective
    current_audit: dict[str, Any] | None = None

    def wrapped_decode(**kwargs: Any) -> dict[str, Any]:
        nonlocal current_audit
        # A minimal ordinary decode keeps evaluate_record's output contract
        # intact; the independently recorded audit below is the experiment.
        result = original_decode(
            **{
                **kwargs,
                "initial_speeds_mps": (3.0,),
                "max_iterations": 1,
                "curvature_multistart": False,
            }
        )
        current_audit = _fit_audit(
            kwargs,
            grid_seeds=int(args.grid_seeds),
            local_iterations=int(args.local_iterations),
        )
        return result

    evaluator.decode_continuous_trajectory = wrapped_decode
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        for index, record in enumerate(records, start=1):
            try:
                current_audit = None
                evaluator.evaluate_record(record, extractor, config, perception)
                if current_audit is None:
                    raise RuntimeError("decoder audit wrapper was not called")
                rows.append({
                    "sample_id": record["sample_id"],
                    "source_key": record.get("source_key"),
                    "branch_role": record.get("branch_role"),
                    "stratum": (record.get("metadata") or {}).get("stratum"),
                    "audit": current_audit,
                })
            except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as error:
                errors.append({"sample_id": str(record.get("sample_id")), "error": str(error)})
            if index % 10 == 0 or index == len(records):
                print(f"processed {index}/{len(records)}; rows={len(rows)} errors={len(errors)}", flush=True)
    finally:
        trajectory_decode._sample_pixels = original_sample
        trajectory_decode._objective = original_objective
        evaluator.decode_continuous_trajectory = original_decode

    output = {
        "protocol": "fixed-source-spatially-stratified-rigid-fit-v1",
        "forward_backward": not args.disable_fb,
        "manifests": [str(path) for path in args.manifest],
        "summary": _summary(rows),
        "by_stratum": {
            str(stratum): _summary([row for row in rows if row["stratum"] == stratum])
            for stratum in sorted({row["stratum"] for row in rows}, key=lambda value: str(value))
        },
        "errors": errors,
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

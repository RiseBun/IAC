#!/usr/bin/env python3
"""Attach official NAVSIM PDM scores to IAC JSONL rows.

The previous PDMS proxy was geometry-only. This tool uses the official NAVSIM
metric stack: Scene -> NavSimScenario -> MetricCacheProcessor ->
PDMSimulator/PDMScorer -> navsim.evaluate.pdm_score.pdm_score.

Rows are scored by the NAVSIM token embedded in IAC sample_id:
    <log_name>__<token>__<source_type>
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import pickle
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NAVSIM_ROOT = PROJECT_ROOT / "third_party" / "navsim_official"
DEFAULT_NAVSIM_DEPS_ROOT = PROJECT_ROOT / "third_party" / "navsim_runtime_deps"

TRAINING_QUALITY_SOURCES = {
    "gt_pos",
    "perturb_speed",
    "perturb_lateral",
    "perturb_heading",
}

SCORE_COLUMNS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "pdm_score",
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def _source(row: Dict[str, Any]) -> str:
    return str(row.get("source_type") or row.get("sample_type") or "unknown")


def _parse_source_set(raw: str) -> set[str] | None:
    value = str(raw or "").strip()
    if not value or value.lower() in {"all", "*"}:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _group_id(row: Dict[str, Any]) -> str:
    value = row.get("group_id") or row.get("anchor_id")
    if value is not None:
        return str(value)
    sample_id = str(row.get("sample_id", ""))
    if "__" in sample_id:
        return sample_id.rsplit("__", 1)[0]
    return sample_id


def _token_from_row(row: Dict[str, Any]) -> str | None:
    for key in ("navsim_token", "token", "sample_token", "frame_token"):
        value = row.get(key)
        if value:
            return str(value)
    sample_id = str(row.get("sample_id", ""))
    parts = sample_id.split("__")
    if len(parts) >= 3 and parts[1]:
        return parts[1]
    return None


def _extract_frame_lists(obj: Any) -> List[List[Dict[str, Any]]]:
    if isinstance(obj, list):
        if not obj:
            return []
        if all(isinstance(item, dict) for item in obj):
            return [obj]
        lists: List[List[Dict[str, Any]]] = []
        for item in obj:
            lists.extend(_extract_frame_lists(item))
        return lists
    if isinstance(obj, dict):
        for key in ("frames", "scene_frames", "data", "logs", "samples"):
            if key in obj:
                return _extract_frame_lists(obj[key])
    return []


def _load_log_frames(log_path: Path) -> List[Dict[str, Any]]:
    obj = pickle.load(open(log_path, "rb"))
    lists = _extract_frame_lists(obj)
    if not lists:
        raise ValueError(f"no frame list found in {log_path}")
    if len(lists) > 1:
        raise ValueError(f"expected one frame list in {log_path}, got {len(lists)}")
    return lists[0]


def _setup_official_navsim(args: argparse.Namespace) -> Dict[str, Any]:
    navsim_root = Path(args.navsim_repo).resolve()
    if not navsim_root.exists():
        raise FileNotFoundError(
            f"NAVSIM official repo not found: {navsim_root}. "
            "Clone https://github.com/autonomousvision/navsim under IAC/third_party first."
        )
    deps_root = Path(args.navsim_deps).resolve()
    if deps_root.exists():
        sys.path.insert(0, str(deps_root))
    sys.path.insert(0, str(navsim_root))
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(Path(args.openscene_data_root).resolve()))
    os.environ["NUPLAN_MAPS_ROOT"] = str(Path(args.map_root).resolve())

    from nuplan.planning.simulation.trajectory.trajectory_sampling import (  # type: ignore
        TrajectorySampling,
    )
    from navsim.common.dataclasses import Scene, SensorConfig, Trajectory  # type: ignore
    from navsim.common.enums import SceneFrameType  # type: ignore
    from navsim.evaluate.pdm_score import (  # type: ignore
        get_trajectory_as_array,
        pdm_score,
        transform_trajectory,
    )
    from navsim.planning.metric_caching.metric_cache_processor import (  # type: ignore
        MetricCacheProcessor,
    )
    from navsim.planning.scenario_builder.navsim_scenario import (  # type: ignore
        NavSimScenario,
    )
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (  # type: ignore
        PDMScorer,
        PDMScorerConfig,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (  # type: ignore
        PDMSimulator,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (  # type: ignore
        WeightedMetricIndex,
    )
    from navsim.traffic_agents_policies.constant_velocity_traffic_agents import (  # type: ignore
        ConstantVelocityTrafficAgents,
    )
    from navsim.traffic_agents_policies.log_replay_traffic_agents import (  # type: ignore
        LogReplayTrafficAgents,
    )

    return {
        "TrajectorySampling": TrajectorySampling,
        "Scene": Scene,
        "SceneFrameType": SceneFrameType,
        "SensorConfig": SensorConfig,
        "Trajectory": Trajectory,
        "get_trajectory_as_array": get_trajectory_as_array,
        "pdm_score": pdm_score,
        "transform_trajectory": transform_trajectory,
        "MetricCacheProcessor": MetricCacheProcessor,
        "NavSimScenario": NavSimScenario,
        "PDMScorer": PDMScorer,
        "PDMScorerConfig": PDMScorerConfig,
        "PDMSimulator": PDMSimulator,
        "WeightedMetricIndex": WeightedMetricIndex,
        "ConstantVelocityTrafficAgents": ConstantVelocityTrafficAgents,
        "LogReplayTrafficAgents": LogReplayTrafficAgents,
    }


def _upsample_candidate_traj(
    candidate_traj: Sequence[Sequence[Any]],
    *,
    num_poses: int,
    interval_length: float,
    source_interval_length: float,
) -> np.ndarray:
    arr = np.asarray(candidate_traj, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("candidate_traj must be a non-empty 2D array")
    if arr.shape[1] < 3:
        pad = np.zeros((arr.shape[0], 3 - arr.shape[1]), dtype=np.float64)
        arr = np.concatenate([arr, pad], axis=1)
    arr = arr[:, :3]

    src_t = np.concatenate(
        [[0.0], np.arange(1, arr.shape[0] + 1, dtype=np.float64) * source_interval_length]
    )
    src = np.vstack([np.zeros((1, 3), dtype=np.float64), arr])
    src[:, 2] = np.unwrap(src[:, 2])
    dst_t = np.arange(1, num_poses + 1, dtype=np.float64) * interval_length
    out = np.zeros((num_poses, 3), dtype=np.float32)
    out[:, 0] = np.interp(dst_t, src_t, src[:, 0])
    out[:, 1] = np.interp(dst_t, src_t, src[:, 1])
    out[:, 2] = np.interp(dst_t, src_t, src[:, 2])
    out[:, 2] = np.arctan2(np.sin(out[:, 2]), np.cos(out[:, 2]))
    return out


def _pdm_frame_to_scores(score_frame: Any) -> Dict[str, float]:
    series = score_frame.iloc[0]
    return {
        key: float(series[key])
        for key in SCORE_COLUMNS
        if key in series and math.isfinite(float(series[key]))
    }


class OfficialPDMRunner:
    def __init__(self, args: argparse.Namespace, official: Dict[str, Any]) -> None:
        self.args = args
        self.official = official
        self.sampling = official["TrajectorySampling"](
            num_poses=int(args.num_poses),
            interval_length=float(args.interval_length),
        )
        self.processor = official["MetricCacheProcessor"](
            cache_path=str(args.metric_cache_path) if args.save_metric_cache else None,
            force_feature_computation=bool(args.force_metric_cache),
            proposal_sampling=self.sampling,
        )
        scorer_config = official["PDMScorerConfig"](
            human_penalty_filter=bool(args.human_penalty_filter)
        )
        self.simulator = official["PDMSimulator"](proposal_sampling=self.sampling)
        self.scorer = official["PDMScorer"](
            proposal_sampling=self.sampling,
            config=scorer_config,
        )
        if args.traffic_agents == "constant_velocity":
            self.traffic_agents_policy = official["ConstantVelocityTrafficAgents"](
                self.sampling
            )
        elif args.traffic_agents == "log_replay":
            self.traffic_agents_policy = official["LogReplayTrafficAgents"](
                self.sampling
            )
        else:
            raise ValueError(f"unknown traffic_agents: {args.traffic_agents}")

        self._frames_by_log: Dict[str, List[Dict[str, Any]]] = {}
        self._index_by_log: Dict[str, Dict[str, int]] = {}
        self._metric_cache_by_token: Dict[str, Any] = {}
        self.cache_paths: List[Path] = []

    def _log_path(self, log_name: str) -> Path:
        return Path(self.args.navsim_log_path) / f"{log_name}.pkl"

    def _frames_for_log(self, log_name: str) -> List[Dict[str, Any]]:
        if log_name not in self._frames_by_log:
            path = self._log_path(log_name)
            if not path.exists():
                raise FileNotFoundError(f"NAVSIM log missing: {path}")
            frames = _load_log_frames(path)
            self._frames_by_log[log_name] = frames
            self._index_by_log[log_name] = {
                str(frame["token"]): idx
                for idx, frame in enumerate(frames)
                if frame.get("token")
            }
        return self._frames_by_log[log_name]

    def _cache_path_for(self, log_name: str, token: str) -> Path:
        return Path(self.args.metric_cache_path) / log_name / "unknown" / token / "metric_cache.pkl"

    def _scene_for_token(self, log_name: str, token: str) -> Any:
        frames = self._frames_for_log(log_name)
        token_to_index = self._index_by_log[log_name]
        if token not in token_to_index:
            raise KeyError(f"token {token} not found in {log_name}")
        current_idx = token_to_index[token]
        history = int(self.args.num_history_frames)
        future = int(self.args.num_future_frames)
        start = current_idx - history + 1
        end = current_idx + future + 1
        if start < 0 or end > len(frames):
            raise IndexError(
                f"token {token} in {log_name} lacks official context window: "
                f"start={start}, end={end}, frames={len(frames)}"
            )
        frame_list = frames[start:end]
        if self.args.require_route and not frame_list[history - 1].get("roadblock_ids"):
            raise ValueError(f"token {token} in {log_name} has no route")
        return self.official["Scene"].from_scene_dict_list(
            frame_list,
            None,
            num_history_frames=history,
            num_future_frames=future,
            sensor_config=self.official["SensorConfig"].build_no_sensors(),
        )

    def _metric_cache_for(self, log_name: str, token: str) -> Any:
        if token in self._metric_cache_by_token:
            return self._metric_cache_by_token[token]
        cache_path = self._cache_path_for(log_name, token)
        if self.args.load_metric_cache and cache_path.exists():
            with lzma.open(cache_path, "rb") as f:
                metric_cache = pickle.load(f)
        else:
            scene = self._scene_for_token(log_name, token)
            scenario = self.official["NavSimScenario"](
                scene,
                map_root=str(Path(self.args.map_root)),
                map_version=str(self.args.map_version),
            )
            if self.args.save_metric_cache:
                meta = self.processor.compute_and_save_metric_cache(scenario)
                if meta is not None:
                    meta_path = (
                        getattr(meta, "file_name", None)
                        or getattr(meta, "file_path", None)
                        or getattr(meta, "cache_path", None)
                    )
                    if meta_path is not None:
                        self.cache_paths.append(Path(meta_path))
                with lzma.open(cache_path, "rb") as f:
                    metric_cache = pickle.load(f)
            else:
                metric_cache = self.processor.compute_metric_cache(scenario)
        self._metric_cache_by_token[token] = metric_cache
        return metric_cache

    def _apply_human_penalty_filter(
        self,
        metric_cache: Any,
        score_frames: Sequence[Any],
    ) -> None:
        if not self.scorer._config.human_penalty_filter:
            return
        if metric_cache.scene_type != self.official["SceneFrameType"].ORIGINAL:
            return
        initial_ego_state = metric_cache.ego_state
        human_trajectory = self.official["transform_trajectory"](
            metric_cache.human_trajectory,
            initial_ego_state,
        )
        human_states = self.official["get_trajectory_as_array"](
            human_trajectory,
            self.sampling,
            initial_ego_state.time_point,
        )
        human_simulated_states = self.simulator.simulate_proposals(
            human_states[None, ...],
            initial_ego_state,
        )
        human_simulated_agent_detections_tracks = (
            self.traffic_agents_policy.simulate_environment(
                human_simulated_states[0],
                metric_cache,
            )
        )
        human_pdm_result = self.scorer.score_proposals(
            human_simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            human_simulated_agent_detections_tracks,
        )[0]

        skip_columns = {
            "multiplicative_metrics_prod",
            "weighted_metrics",
            "weighted_metrics_array",
            "pdm_score",
        }
        weighted_metric_index = self.official["WeightedMetricIndex"]
        for score_frame in score_frames:
            modified_any = False
            for column in human_pdm_result.columns:
                if (
                    column not in skip_columns
                    and human_pdm_result[column].iloc[0] == 0
                ):
                    score_frame.at[0, column] = 1
                    modified_any = True
            if modified_any:
                score_frame.at[0, "multiplicative_metrics_prod"] = (
                    score_frame.at[0, "no_at_fault_collisions"]
                    * score_frame.at[0, "drivable_area_compliance"]
                    * score_frame.at[0, "driving_direction_compliance"]
                    * score_frame.at[0, "traffic_light_compliance"]
                )
                weighted_metrics = score_frame.at[0, "weighted_metrics"].copy()
                weighted_metrics[weighted_metric_index.PROGRESS] = score_frame.at[
                    0, "ego_progress"
                ]
                weighted_metrics[weighted_metric_index.TTC] = score_frame.at[
                    0, "time_to_collision_within_bound"
                ]
                weighted_metrics[weighted_metric_index.LANE_KEEPING] = score_frame.at[
                    0, "lane_keeping"
                ]
                weighted_metrics[weighted_metric_index.HISTORY_COMFORT] = score_frame.at[
                    0, "history_comfort"
                ]
                score_frame.at[0, "weighted_metrics"] = weighted_metrics

    def _pairwise_official_score_frames(
        self,
        score_frames: Sequence[Any],
        multi_metrics: np.ndarray,
        weighted_metrics: np.ndarray,
        progress_raw: np.ndarray,
    ) -> List[Any]:
        weighted_metric_index = self.official["WeightedMetricIndex"]
        metric_weights = self.scorer._config.weighted_metrics_array
        metric_mask = np.ones_like(metric_weights, dtype=bool)
        metric_mask[weighted_metric_index.TWO_FRAME_EXTENDED_COMFORT] = False
        multiplicative_scores = multi_metrics.prod(axis=0)

        adjusted: List[Any] = []
        for proposal_idx, score_frame in enumerate(score_frames, start=1):
            frame = score_frame.copy(deep=True)
            norm_constant = max(
                float(progress_raw[0] * multiplicative_scores[0]),
                float(progress_raw[proposal_idx] * multiplicative_scores[proposal_idx]),
            )
            if norm_constant > self.scorer._config.progress_distance_threshold:
                progress_score = float(
                    np.clip(progress_raw[proposal_idx] / norm_constant, 0.0, 1.0)
                )
            else:
                progress_score = 1.0
            proposal_weighted = weighted_metrics[:, proposal_idx].copy()
            proposal_weighted[weighted_metric_index.PROGRESS] = progress_score
            weighted_score = float(
                (proposal_weighted[metric_mask] * metric_weights[metric_mask]).sum()
                / metric_weights[metric_mask].sum()
            )
            final_score = float(multiplicative_scores[proposal_idx] * weighted_score)
            frame.at[0, "ego_progress"] = progress_score
            frame.at[0, "weighted_metrics"] = proposal_weighted
            frame.at[0, "pdm_score"] = final_score
            adjusted.append(frame)
        return adjusted

    def score_rows(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
        if not rows:
            return []
        pairs = {
            (
                str(row.get("scene_name") or row.get("log_name") or ""),
                _token_from_row(row),
            )
            for row in rows
        }
        if len(pairs) != 1:
            raise ValueError("score_rows expects one scene/token group")
        log_name, token = next(iter(pairs))
        if not log_name or token is None:
            raise ValueError("row must contain scene_name/log_name and token/sample_id")
        metric_cache = self._metric_cache_for(log_name, token)
        initial_ego_state = metric_cache.ego_state
        pdm_states = self.official["get_trajectory_as_array"](
            metric_cache.trajectory,
            self.sampling,
            initial_ego_state.time_point,
        )
        pred_states = []
        for row in rows:
            poses = _upsample_candidate_traj(
                row.get("candidate_traj") or [],
                num_poses=int(self.args.num_poses),
                interval_length=float(self.args.interval_length),
                source_interval_length=float(self.args.source_interval_length),
            )
            trajectory = self.official["Trajectory"](
                poses=poses,
                trajectory_sampling=self.sampling,
            )
            pred_trajectory = self.official["transform_trajectory"](
                trajectory,
                initial_ego_state,
            )
            pred_states.append(
                self.official["get_trajectory_as_array"](
                    pred_trajectory,
                    self.sampling,
                    initial_ego_state.time_point,
                )
            )
        trajectory_states = np.concatenate(
            [pdm_states[None, ...]] + [states[None, ...] for states in pred_states],
            axis=0,
        )
        simulated_states = self.simulator.simulate_proposals(
            trajectory_states,
            initial_ego_state,
        )
        simulated_agent_detections_tracks = (
            self.traffic_agents_policy.simulate_environment(
                simulated_states[1],
                metric_cache,
            )
        )
        if len(simulated_agent_detections_tracks) != trajectory_states.shape[1]:
            raise AssertionError(
                "Traffic agents trajectories must be of length "
                f"{trajectory_states.shape[1]}, got "
                f"{len(simulated_agent_detections_tracks)}"
            )

        score_frames = self.scorer.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            simulated_agent_detections_tracks,
            metric_cache.past_human_trajectory,
        )
        multi_metrics = np.array(self.scorer._multi_metrics, copy=True)
        weighted_metrics = np.array(self.scorer._weighted_metrics, copy=True)
        progress_raw = np.array(self.scorer._progress_raw, copy=True)
        adjusted_frames = self._pairwise_official_score_frames(
            score_frames[1:],
            multi_metrics,
            weighted_metrics,
            progress_raw,
        )
        self._apply_human_penalty_filter(metric_cache, adjusted_frames)
        return [_pdm_frame_to_scores(frame) for frame in adjusted_frames]

    def score_row(self, row: Dict[str, Any]) -> Dict[str, float]:
        return self.score_rows([row])[0]

    def write_cache_metadata(self) -> None:
        if not self.args.save_metric_cache or not self.cache_paths:
            return
        metadata_dir = Path(self.args.metric_cache_path) / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        path = metadata_dir / "iac_official_pdms_cache.csv"
        known: set[str] = set()
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            known.update(line.strip() for line in lines[1:] if line.strip())
        known.update(str(item) for item in self.cache_paths)
        with path.open("w", encoding="utf-8") as f:
            f.write("cache_path\n")
            for item in sorted(known):
                f.write(item)
                f.write("\n")


def _select_groups(
    rows: Sequence[Dict[str, Any]],
    max_groups: int,
) -> List[Dict[str, Any]]:
    if max_groups <= 0:
        return list(rows)
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        gid = _group_id(row)
        if gid not in seen:
            if len(seen) >= max_groups:
                continue
            seen.add(gid)
        if gid in seen:
            selected.append(row)
    return selected


def _shard_groups(
    rows: Sequence[Dict[str, Any]],
    num_shards: int,
    shard_index: int,
) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return list(rows)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )
    group_order: Dict[str, int] = {}
    for row in rows:
        gid = _group_id(row)
        if gid not in group_order:
            group_order[gid] = len(group_order)
    return [
        row
        for row in rows
        if group_order[_group_id(row)] % num_shards == shard_index
    ]


def enrich_rows(
    rows: Sequence[Dict[str, Any]],
    runner: OfficialPDMRunner,
    fail_on_error: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    score_sources = _parse_source_set(runner.args.score_sources)
    enriched: List[Dict[str, Any] | None] = [None] * len(rows)
    score_values: List[float] = []
    score_by_source: Dict[str, List[float]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    scored_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    rows_by_group: Dict[str, List[int]] = defaultdict(list)

    for idx, row in enumerate(rows):
        out = dict(row)
        source = _source(out)
        source_counts[source] += 1
        out["group_id"] = _group_id(out)
        token = _token_from_row(out)
        if token is not None:
            out["navsim_token"] = token
        enriched[idx] = out
        if score_sources is not None and source not in score_sources:
            skipped_counts[source] += 1
            continue
        rows_by_group[out["group_id"]].append(idx)

    score_total = sum(len(indices) for indices in rows_by_group.values())
    scored_attempts = 0
    for group_idx, indices in enumerate(rows_by_group.values(), 1):
        group_rows = [enriched[idx] for idx in indices]
        assert all(row is not None for row in group_rows)
        try:
            group_scores = runner.score_rows(group_rows)  # type: ignore[arg-type]
            for out, scores in zip(group_rows, group_scores):
                source = _source(out)
                pdms = float(scores["pdm_score"])
                out["official_pdm_score"] = pdms
                out["pdms_score"] = pdms
                out["planning_score"] = pdms
                if source in TRAINING_QUALITY_SOURCES:
                    out["candidate_quality_score"] = pdms
                for key, value in scores.items():
                    out[f"official_{key}"] = float(value)
                score_values.append(pdms)
                score_by_source[source].append(pdms)
                scored_counts[source] += 1
        except Exception as exc:
            if fail_on_error:
                raise
            for out in group_rows:
                source = _source(out)
                out["official_pdm_error"] = f"{type(exc).__name__}: {exc}"
                out["pdms_score"] = None
                out["planning_score"] = None
                if source in TRAINING_QUALITY_SOURCES:
                    out["candidate_quality_score"] = None
            failure_counts[type(exc).__name__] += 1
            if len(failure_counts) <= 5:
                print(
                    f"[WARN] failed group {group_idx}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc(limit=2)
        scored_attempts += len(indices)
        if scored_attempts == score_total or scored_attempts % 100 <= len(indices):
            print(
                f"[PDMS] scored rows={scored_attempts}/{score_total} "
                f"groups={group_idx}/{len(rows_by_group)}",
                flush=True,
            )

    summary = {
        "rows": len(rows),
        "groups": len({_group_id(row) for row in rows}),
        "tokens": len({str(_token_from_row(row)) for row in rows if _token_from_row(row)}),
        "source_counts": dict(source_counts),
        "scored_counts": dict(scored_counts),
        "skipped_counts": dict(skipped_counts),
        "failure_counts": dict(failure_counts),
        "mean_official_pdm_score": mean(score_values) if score_values else None,
        "mean_official_pdm_by_source": {
            key: mean(values)
            for key, values in sorted(score_by_source.items())
            if values
        },
        "official_navsim_repo": str(Path(runner.args.navsim_repo).resolve()),
        "navsim_log_path": str(Path(runner.args.navsim_log_path).resolve()),
        "map_root": str(Path(runner.args.map_root).resolve()),
        "traffic_agents": runner.args.traffic_agents,
        "num_poses": int(runner.args.num_poses),
        "interval_length": float(runner.args.interval_length),
        "score_sources": "all" if score_sources is None else sorted(score_sources),
    }
    return [row for row in enriched if row is not None], summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add official NAVSIM PDM scores to IAC JSONL rows."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--navsim-repo", type=Path, default=DEFAULT_NAVSIM_ROOT)
    parser.add_argument("--navsim-deps", type=Path, default=DEFAULT_NAVSIM_DEPS_ROOT)
    parser.add_argument(
        "--openscene-data-root",
        type=Path,
        default=Path("/mnt/slurmfs-3090node1_msp/public_data/download/navtrain"),
    )
    parser.add_argument(
        "--navsim-log-path",
        type=Path,
        default=Path("/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_navsim_logs/trainval"),
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=Path("/mnt/slurmfs-3090node3_msp/public_data/nuplan/dataset/maps"),
    )
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    parser.add_argument(
        "--metric-cache-path",
        type=Path,
        default=PROJECT_ROOT / "work_dirs" / "navsim_official_metric_cache_iac",
    )
    parser.add_argument("--save-metric-cache", action="store_true")
    parser.add_argument("--load-metric-cache", action="store_true")
    parser.add_argument("--force-metric-cache", action="store_true")
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--num-poses", type=int, default=40)
    parser.add_argument("--interval-length", type=float, default=0.1)
    parser.add_argument("--source-interval-length", type=float, default=0.5)
    parser.add_argument(
        "--traffic-agents",
        choices=["log_replay", "constant_velocity"],
        default="log_replay",
    )
    parser.add_argument("--human-penalty-filter", action="store_true", default=True)
    parser.add_argument("--no-human-penalty-filter", dest="human_penalty_filter", action="store_false")
    parser.add_argument("--require-route", action="store_true", default=True)
    parser.add_argument("--allow-no-route", dest="require_route", action="store_false")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--score-sources",
        default="all",
        help=(
            "Comma-separated source_type list to score with official PDM. "
            "Use 'all' to score every row."
        ),
    )
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official = _setup_official_navsim(args)
    rows = _select_groups(_read_jsonl(args.input), max_groups=int(args.max_groups))
    rows = _shard_groups(
        rows,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    runner = OfficialPDMRunner(args, official)
    enriched, summary = enrich_rows(
        rows,
        runner,
        fail_on_error=bool(args.fail_on_error),
    )
    summary["num_shards"] = int(args.num_shards)
    summary["shard_index"] = int(args.shard_index)
    _write_jsonl(args.output, enriched)
    runner.write_cache_metadata()
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

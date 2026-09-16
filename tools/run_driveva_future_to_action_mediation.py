#!/usr/bin/env python3
"""Run the four-condition future-to-action mediation probe for DriveVA.

This runner is intended for the DriveVA checkout on ``iac``.  It keeps the
trajectory noise fixed across conditions, changes only the video-latent seed
for the future perturbation, and uses the internal directional attention mask
for the blocked/fixed-action pathway controls.  It never injects an evaluator
trajectory as the native action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


CONDITIONS = (
    "baseline",
    "future_perturbed",
    "future_perturbed_pathway_blocked",
    "future_fixed_action_pathway_control",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        token = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if len(token) == 64:
            return token
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _history(row: dict[str, Any], size: tuple[int, int]) -> list[Image.Image]:
    paths = row.get("history_images") or row.get("history_frame_paths")
    if not paths or len(paths) != 4:
        raise ValueError("each source must provide exactly four history image paths")
    return [Image.open(str(path)).convert("RGB").resize(size, Image.BICUBIC) for path in paths]


def _speed(row: dict[str, Any]) -> float:
    history = np.asarray(row.get("history_ego_state") or [], dtype=np.float64)
    if history.ndim == 2 and history.shape[1] >= 4:
        return float(max(history[-1, 3], 0.0))
    return 5.0


def _build_pipeline(root: Path, device: str, checkpoint: Path):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "examples/wanvideo/driveva_infer"))
    from diffsynth import load_state_dict
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
    from eval_navsim_pdm import _normalize_checkpoint_keys

    model_dir = str(root / "models")
    configs = [
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device=device, local_model_path=model_dir, skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device=device, local_model_path=model_dir, skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device=device, local_model_path=model_dir, skip_download=True),
    ]
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=configs, use_trajectory=True)
    pipe.load_state_dict(_normalize_checkpoint_keys(load_state_dict(str(checkpoint))), strict=False)
    pipe.eval()
    pipe.target_fps = 2
    pipe.num_history_frames = 4
    pipe.trajectory_norm_mode = "driveva_odo"
    pipe.trajectory_use_relative = False
    pipe.trajectory_condition_mode = "velocity"
    if not getattr(pipe, "_iac_vram_patched", False):
        pipe.enable_vram_management(vram_limit=20)
    return pipe


def _command_and_prompt(root: Path, row: dict[str, Any], speed: float) -> tuple[str, str, str]:
    sys.path.insert(0, str(root / "examples/wanvideo/driveva_infer"))
    from navsim_dataset import _build_prompt_fixed

    mode = str(row.get("branch_mode") or "logged")
    command = {"left": [1.0, 0.0, 0.0], "logged": [0.0, 1.0, 0.0], "right": [0.0, 0.0, 1.0]}.get(mode, [0.0, 1.0, 0.0])
    prompt = _build_prompt_fixed(command, speed)
    return mode, prompt, _fingerprint({"command": command, "speed": speed, "mode": mode})


def _run_condition(
    pipe,
    root: Path,
    row: dict[str, Any],
    condition: str,
    args,
    model_revision: str,
    *,
    reference_cache: dict[str, Any] | None = None,
    capture_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    speed = _speed(row)
    mode, prompt, command_fingerprint = _command_and_prompt(root, row, speed)
    source_key = str(row.get("source_key") or row.get("sample_id"))
    group_id = str(row.get("counterfactual_group_id") or source_key)
    source_seed = int(row.get("nuisance_seed", args.seed))
    future_seed = args.seed + (1 if condition in {"future_perturbed", "future_perturbed_pathway_blocked"} else 0)
    blocked = condition in {"future_perturbed_pathway_blocked", "future_fixed_action_pathway_control"}
    future_fingerprint = _fingerprint({"source": source_key, "future_seed": future_seed, "model_revision": model_revision})
    hist = _history(row, (args.width, args.height))
    with torch.no_grad():
        video, pred_traj, _ = pipe(
            prompt=prompt,
            negative_prompt="worst quality, low quality, blurry",
            longcat_video=hist,
            height=args.height,
            width=args.width,
            num_frames=len(hist) + 8,
            cfg_scale=1.0,
            num_inference_steps=args.steps,
            trajectory_len=8,
            ego_vel=torch.tensor([speed, 0.0], dtype=torch.float32),
            seed=future_seed,
            trajectory_seed=args.trajectory_seed,
            future_to_action_blocked=blocked,
            future_to_action_reference_cache=reference_cache,
            future_to_action_capture=capture_cache,
            rand_device=args.device,
            tiled=True,
        )
    out_dir = args.output_dir / group_id.replace("/", "_") / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    for index, frame in enumerate(list(video)[-8:]):
        path = out_dir / f"future_{index:02d}.png"
        frame.save(path)
        image_paths.append(str(path))
    trajectory = pred_traj[0].detach().float().cpu().numpy().tolist()
    native_action = trajectory[-1]
    return {
        "source_key": source_key,
        "counterfactual_group_id": group_id,
        "sample_id": f"{row.get('sample_id', group_id)}::{condition}",
        "condition": condition,
        "history_fingerprint": row.get("history_fingerprint") or _fingerprint(row.get("history_frame_paths")),
        "command_fingerprint": row.get("command_fingerprint") or command_fingerprint,
        "nuisance_seed": source_seed,
        "model_revision": model_revision,
        "wam_model_id": "driveva_navsim",
        "native_action": native_action,
        "action_trajectory": trajectory,
        "action_source": "native_action_head",
        "native_action_source": "native_action_head",
        "action_is_native": True,
        "action_normalization_fingerprint": args.normalization_fingerprint,
        "action_normalization_scale": [float(value) for value in args.normalization_scale],
        "future_fingerprint": future_fingerprint,
        "future_fingerprint_kind": "deterministic_video_latent_seed",
        "pathway_state": "blocked" if condition == "future_perturbed_pathway_blocked" else "fixed_action" if condition == "future_fixed_action_pathway_control" else "normal",
        "future_images": image_paths,
        "future_images_source": "wam_generated",
        "future_times_s": [0.5 * (index + 1) for index in range(8)],
        "future_seed": future_seed,
        "trajectory_seed": args.trajectory_seed,
        "future_to_action_blocked": blocked,
        "candidate_bank_used_by_decoder": False,
        "action_vector_definition": "final predicted trajectory point [x,y,z]",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driveva-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--trajectory-seed", type=int, default=20260917)
    parser.add_argument("--nuisance-seed", type=int, default=20260918)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--normalization-fingerprint", default="driveva-calibration-v1")
    parser.add_argument("--normalization-scale", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.device = args.device or f"cuda:{args.gpu}"
    rows = _read_jsonl(args.input)
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped.setdefault(str(row.get("source_key") or row.get("sample_id")), row)
    rows = list(deduped.values())
    if args.max_sources > 0:
        rows = rows[: args.max_sources]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "future_to_action_mediation.jsonl"
    existing = {json.loads(line)["sample_id"] for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()} if manifest.exists() else set()
    model_revision = _sha256_file(args.checkpoint)
    torch.cuda.set_device(args.gpu)
    pipe = _build_pipeline(args.driveva_root, args.device, args.checkpoint)
    with manifest.open("a", encoding="utf-8") as handle:
        for row in rows:
            source_id = str(row.get("sample_id", row.get("source_key")))
            baseline_cache: dict[str, Any] = {}
            ordered_conditions = (
                ("baseline", None, baseline_cache),
                ("future_perturbed", None, None),
                ("future_perturbed_pathway_blocked", baseline_cache, None),
                ("future_fixed_action_pathway_control", baseline_cache, None),
            )
            for condition, reference_cache, capture_cache in ordered_conditions:
                sample_id = f"{source_id}::{condition}"
                if sample_id in existing:
                    continue
                record = _run_condition(
                    pipe,
                    args.driveva_root,
                    row,
                    condition,
                    args,
                    model_revision,
                    reference_cache=reference_cache,
                    capture_cache=capture_cache,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                existing.add(sample_id)
                print(json.dumps({"condition": condition, "source_key": record["source_key"], "completed": len(existing)}), flush=True)
                torch.cuda.empty_cache()
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "protocol": "iac-future-to-action-mediation-v1",
        "model": "DriveVA",
        "sources": len(rows),
        "conditions": list(CONDITIONS),
        "seed": args.seed,
        "trajectory_seed": args.trajectory_seed,
        "normalization_fingerprint": args.normalization_fingerprint,
        "normalization_scale": args.normalization_scale,
        "model_revision": model_revision,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(existing), "output": str(manifest)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate DriveVA counterfactual twins with a shared nuisance seed.

This runner intentionally lives in IAC rather than patching the external
DriveVA checkout.  The original runner advanced the seed by input-row index,
so the left and right members of a pair changed both navigation command and
diffusion noise.  That output cannot support a single-variable RCS estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DRIVEVA_ROOT = Path(
    "/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/DriveVA"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def branch_mode(row: dict) -> str:
    return str(row.get("branch_mode") or row.get("branch_role") or "logged")


def output_sample_id(row: dict) -> str:
    group = str(row.get("counterfactual_group_id") or row.get("source_key"))
    return str(row.get("branch_id") or f"{group}::{branch_mode(row)}")


def history(row: dict, size: tuple[int, int]) -> list[Image.Image]:
    paths = row.get("history_images") or row.get("history_frame_paths")
    return [Image.open(str(path)).convert("RGB").resize(size, Image.BICUBIC) for path in paths]


def history_speed(row: dict) -> float:
    states = np.asarray(row.get("history_ego_state") or [], dtype=np.float64)
    if states.ndim == 2 and states.shape[1] >= 4:
        return float(max(states[-1, 3], 0.0))
    return 5.0


def build_pipeline(device: str):
    sys.path.insert(0, str(DRIVEVA_ROOT))
    sys.path.insert(0, str(DRIVEVA_ROOT / "examples/wanvideo/driveva_infer"))
    from diffsynth import load_state_dict
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
    from eval_navsim_pdm import _normalize_checkpoint_keys

    local = str(DRIVEVA_ROOT / "models")
    configs = [
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            offload_device=device,
            local_model_path=local,
            skip_download=True,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            offload_device=device,
            local_model_path=local,
            skip_download=True,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="Wan2.2_VAE.pth",
            offload_device=device,
            local_model_path=local,
            skip_download=True,
        ),
    ]
    pipeline = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=configs,
        use_trajectory=True,
    )
    checkpoint = DRIVEVA_ROOT / "checkpoints/pdms90_9.safetensors"
    pipeline.load_state_dict(
        _normalize_checkpoint_keys(load_state_dict(str(checkpoint))), strict=False
    )
    pipeline.eval()
    pipeline.target_fps = 2
    pipeline.num_history_frames = 4
    pipeline.trajectory_norm_mode = "driveva_odo"
    pipeline.trajectory_use_relative = False
    pipeline.trajectory_condition_mode = "velocity"
    if not getattr(pipeline, "_iac_vram_patched", False):
        pipeline.enable_vram_management(vram_limit=20)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--fallback-seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    all_rows = read_jsonl(args.input)
    indexed_rows = [
        (index, row)
        for index, row in enumerate(all_rows)
        if index % args.num_shards == args.shard_index
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / f"driveva_native_joint_{args.shard_index:02d}.jsonl"
    existing: dict[str, dict] = {}
    if args.resume and manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[str(record["sample_id"])] = record

    pending = [
        (index, row)
        for index, row in indexed_rows
        if output_sample_id(row) not in existing
    ]
    print(
        json.dumps(
            {
                "all_rows": len(all_rows),
                "shard_rows": len(indexed_rows),
                "existing_rows": len(existing),
                "pending_rows": len(pending),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
        ),
        flush=True,
    )

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    pipeline = build_pipeline(device)
    from navsim_dataset import _build_prompt_fixed

    command_by_mode = {
        "left": [1.0, 0.0, 0.0],
        "logged": [0.0, 1.0, 0.0],
        "right": [0.0, 0.0, 1.0],
    }
    for completed, (global_index, row) in enumerate(pending, start=1):
        mode = branch_mode(row)
        pair_seed = int(row.get("nuisance_seed", args.fallback_seed + global_index))
        speed = history_speed(row)
        frames = history(row, (args.width, args.height))
        prompt = _build_prompt_fixed(command_by_mode.get(mode, command_by_mode["logged"]), speed)
        with torch.no_grad():
            video, predicted_trajectory, _ = pipeline(
                prompt=prompt,
                negative_prompt="worst quality, low quality, blurry",
                longcat_video=frames,
                height=args.height,
                width=args.width,
                num_frames=len(frames) + 8,
                cfg_scale=1.0,
                num_inference_steps=args.steps,
                trajectory_len=8,
                ego_vel=torch.tensor([speed, 0.0], dtype=torch.float32),
                seed=pair_seed,
                rand_device=device,
                tiled=True,
            )

        sample_id = output_sample_id(row)
        sample_dir = args.output_dir / f"sample_{global_index:04d}_{mode}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        future_paths = []
        for frame_index, frame in enumerate(list(video)[-8:]):
            path = sample_dir / f"future_{frame_index:02d}.png"
            frame.save(path)
            future_paths.append(str(path))
        trajectory = predicted_trajectory[0].detach().float().cpu().numpy().tolist()
        record = {
            "sample_id": sample_id,
            "source_key": row.get("source_key"),
            "counterfactual_group_id": row.get("counterfactual_group_id", row.get("source_key")),
            "branch_role": mode,
            "branch_mode": mode,
            "history_fingerprint": row.get("history_fingerprint"),
            "nuisance_seed": pair_seed,
            "actual_generation_seed": pair_seed,
            "intervention_type": "navigation_command_onehot",
            "branch_id": row.get("branch_id", sample_id),
            "future_images": future_paths,
            "future_images_source": "wam_generated",
            "future_times_s": [0.5 * (index + 1) for index in range(8)],
            "action_trajectory": trajectory,
            "action_trajectory_source": "wam_native_action_head",
            "action_injection_verified": False,
            "candidate_bank_used_by_decoder": False,
            "wam_model_id": "driveva_navsim",
        }
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        existing[sample_id] = record
        print(
            json.dumps(
                {
                    "completed": completed,
                    "pending": len(pending),
                    "global_index": global_index,
                    "sample_id": sample_id,
                    "actual_generation_seed": pair_seed,
                }
            ),
            flush=True,
        )
        torch.cuda.empty_cache()

    run_config = {
        "input": str(args.input),
        "all_rows": len(all_rows),
        "shard_rows": len(indexed_rows),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "fallback_seed": args.fallback_seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "model": "driveva_navsim",
        "native_action": True,
        "future_frames": 8,
        "horizon_s": 4.0,
        "seed_contract": "same counterfactual_group_id uses identical nuisance_seed",
    }
    (args.output_dir / f"run_config_{args.shard_index:02d}.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(existing), "output": str(manifest)}))


if __name__ == "__main__":
    main()

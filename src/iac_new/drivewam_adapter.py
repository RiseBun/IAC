"""External-action rollout adapter for the published DriveWAM NavSim code.

DriveWAM's public evaluator uses a zero action-conditioning chunk while it
generates future video and samples the action branch afterwards.  This
adapter exposes two auditable intervention variants without changing the
model weights:

``condition_chunk=0`` replaces the clean action context chunk.
``condition_chunk=1`` writes the supplied action into the future action chunk
before the future video rollout.  The second variant is intentionally marked
as an intervention probe because it is not the repository's default path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _joint_grid_id(evaluator: Any, shape: tuple[int, ...], *, action: bool) -> torch.Tensor:
    """Build the grid-id layout used by DriveWAM's training forward."""
    from utils import get_mesh_id

    f, h, w = shape[-3:]
    if action:
        patch_f = patch_h = patch_w = 1
        t = 1
    else:
        patch_f, patch_h, patch_w = evaluator.patch_size
        t = 0
    return get_mesh_id(
        f // patch_f, h // patch_h, w // patch_w,
        t=t, f_w=1, f_shift=0, action=action,
    ).to(evaluator.device)[None]


@torch.no_grad()
def rollout_joint_external_action(
    evaluator: Any,
    batch: dict[str, Any],
    action: torch.Tensor,
    *,
    num_video_steps: int = 4,
    seed: int = 0,
    video_denoise_fraction: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental joint sampler matching the training token layout.

    The external future action is held as a clean action token while future
    video latents are denoised through ``forward_train`` together with the
    action stream. This is a coupling probe, not official DriveWAM inference.
    """
    from utils import data_seq_to_patch

    images, first_frame_latent, prompt_emb, ego_cond = _single_sample_context(evaluator, batch)
    action_chunk, _ = _action_tensor_from_batch(evaluator, batch, action)
    zero_action = torch.zeros_like(action_chunk)
    full_action = torch.cat([zero_action, action_chunk], dim=2)
    action_mask = torch.zeros_like(batch["actions"][:1], device=evaluator.device).bool()
    action_mask[:, list(evaluator.config.used_action_channel_ids), 1:2] = True

    num_future_latents = (images.shape[1] - 1) // 4
    latent_shape = (
        first_frame_latent.shape[0], first_frame_latent.shape[1], num_future_latents + 1,
        first_frame_latent.shape[-2], first_frame_latent.shape[-1],
    )
    video_cond = torch.zeros(latent_shape, device=evaluator.device, dtype=evaluator.dtype)
    video_cond[:, :, :1] = first_frame_latent
    video_latents = torch.randn(latent_shape, device=evaluator.device, dtype=evaluator.dtype)
    video_latents[:, :, :1] = first_frame_latent

    latent_grid_id = _joint_grid_id(evaluator, latent_shape, action=False)
    action_grid_id = _joint_grid_id(evaluator, tuple(full_action.shape), action=True)
    video_timesteps = evaluator._prepare_video_timesteps(num_video_steps, video_denoise_fraction)
    action_timesteps = torch.zeros((1, full_action.shape[2]), device=evaluator.device, dtype=torch.float32)

    rng_devices = evaluator._rng_devices()
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        # ``forward_train`` unconditionally compiles the FlexAttention mask.
        # The evaluator is loaded with torch SDPA, whose attention path does
        # not consume that mask; skip only the mask compilation for this
        # experimental eager probe (never for official rollout).
        from modules.model import FlexAttnFunc
        original_init_mask = FlexAttnFunc.init_mask
        FlexAttnFunc.init_mask = staticmethod(lambda *args, **kwargs: None)
        try:
            for step_id, t in enumerate(video_timesteps):
                last_step = step_id == len(video_timesteps) - 1
                video_t = torch.zeros((1, latent_shape[2]), device=evaluator.device, dtype=torch.float32)
                video_t[:, 1:] = float(t.item())
                input_dict = {
                "latent_dict": {
                    "noisy_latents": video_latents,
                    "latent": video_cond,
                    "timesteps": video_t,
                    "cond_timesteps": torch.zeros_like(video_t),
                    "grid_id": latent_grid_id,
                    "text_emb": prompt_emb,
                    "ego_cond": ego_cond,
                },
                "action_dict": {
                    "noisy_latents": full_action,
                    "latent": full_action,
                    "timesteps": action_timesteps,
                    "cond_timesteps": torch.zeros_like(action_timesteps),
                    "grid_id": action_grid_id,
                    "text_emb": prompt_emb,
                    "actions_mask": action_mask,
                },
                "chunk_size": evaluator.config.frame_chunk_size,
                "window_size": evaluator.config.attn_window,
            }
                latent_pred_seq, _ = evaluator.transformer(input_dict, train_mode=True)
                latent_pred = data_seq_to_patch(
                    evaluator.patch_size, latent_pred_seq,
                    latent_shape[2], latent_shape[3], latent_shape[4], batch_size=1,
                )
                if not last_step:
                    video_latents = evaluator.video_scheduler.step(latent_pred, t, video_latents)
                    video_latents[:, :, :1] = first_frame_latent
        finally:
            FlexAttnFunc.init_mask = original_init_mask

    return torch.cat([zero_action, action_chunk], dim=2), video_latents


@dataclass(frozen=True)
class DriveWAMIntervention:
    condition_chunk: int
    action_source: str = "external_trajectory"

    def __post_init__(self) -> None:
        if self.condition_chunk not in (0, 1):
            raise ValueError("DriveWAM condition_chunk must be 0 or 1")


def _single_sample_context(evaluator: Any, batch: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    images = batch["images"][:1].to(evaluator.device)
    prompt = batch["prompt"][:1] if isinstance(batch["prompt"], list) else [batch["prompt"][0]]
    ego_cond = {
        "velocity_acc": batch["ego_velocity_acc"][:1].to(evaluator.device),
        "history_poses": batch["history_poses"][:1].to(evaluator.device),
        "command_ids": batch["command_id"][:1].to(evaluator.device),
    }
    first_frame_latent = evaluator._encode_images(images[:, :1]).to(evaluator.dtype)
    prompt_emb = evaluator._encode_prompt(prompt)
    return images, first_frame_latent, prompt_emb, ego_cond


def _action_tensor_from_batch(evaluator: Any, batch: dict[str, Any], action: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one clean action chunk and its mask in DriveWAM layout."""
    template = batch["actions"][:1, :, 0:1].to(evaluator.device, dtype=evaluator.dtype).clone()
    mask = batch["actions_mask"][:1, :, 0:1].to(evaluator.device).bool().clone()
    if action is None:
        return torch.zeros_like(template), mask
    value = torch.as_tensor(action, device=evaluator.device, dtype=evaluator.dtype)
    if value.ndim in (2, 3) and value.shape[-1] == 3:
        value = trajectory_to_drivewam_chunk(evaluator, value)
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.shape != template.shape:
        raise ValueError(f"external action must have shape {tuple(template.shape)}, got {tuple(value.shape)}")
    return value, mask


def trajectory_to_drivewam_chunk(evaluator: Any, trajectory: torch.Tensor) -> torch.Tensor:
    """Encode cumulative ``[x, y, yaw]`` poses as one DriveWAM action chunk."""
    value = torch.as_tensor(trajectory, device=evaluator.device, dtype=evaluator.dtype)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("trajectory must have shape [T,3] or [B,T,3]")
    if value.shape[1] != int(evaluator.config.action_chunk_steps):
        raise ValueError(
            f"trajectory length must be {evaluator.config.action_chunk_steps}, got {value.shape[1]}"
        )
    deltas = torch.cat([value[:, :1], value[:, 1:] - value[:, :-1]], dim=1)
    q01 = torch.as_tensor(evaluator.config.norm_stat["q01"], device=value.device, dtype=value.dtype)
    q99 = torch.as_tensor(evaluator.config.norm_stat["q99"], device=value.device, dtype=value.dtype)
    normalized = (deltas - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    out = torch.zeros(
        value.shape[0], int(evaluator.config.action_dim), 1,
        value.shape[1], 1, device=value.device, dtype=value.dtype
    )
    channels = list(evaluator.config.used_action_channel_ids)[:3]
    out[:, channels, 0, :, 0] = normalized.transpose(1, 2)
    return out


@torch.no_grad()
def rollout_external_action(
    evaluator: Any,
    batch: dict[str, Any],
    action: torch.Tensor | None,
    *,
    intervention: DriveWAMIntervention = DriveWAMIntervention(0),
    num_video_steps: int = 3,
    num_action_steps: int = 10,
    seed: int = 0,
    video_denoise_fraction: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate future latents under an externally supplied action condition.

    The returned pair matches ``NavSimEvaluator.predict``:
    ``(pred_actions, pred_latents)``.  This function is deliberately separate
    from the model repository so the benchmark can record the exact
    intervention variant in its manifest.
    """
    images, first_frame_latent, prompt_emb, ego_cond = _single_sample_context(evaluator, batch)
    action_cond, action_mask = _action_tensor_from_batch(evaluator, batch, action)
    zero_action = torch.zeros_like(action_cond)
    rng_devices = evaluator._rng_devices()
    predicted_latents = [first_frame_latent]

    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        evaluator._init_rollout_cache(first_frame_latent, zero_action)
        try:
            evaluator.transformer(
                evaluator._make_video_input(
                    first_frame_latent,
                    torch.zeros((1, 1), device=evaluator.device),
                    prompt_emb,
                    ego_cond,
                    frame_index=0,
                ),
                update_cache=2,
                cache_name=evaluator.cache_name,
                action_mode=False,
            )

            if intervention.condition_chunk == 0:
                cache_action = action_cond
                cache_chunk = 0
            else:
                # Preserve the repository's context action, then write the
                # supplied action in the future action position before video.
                cache_action = zero_action
                cache_chunk = 0
            evaluator.transformer(
                evaluator._make_action_input(
                    cache_action,
                    torch.zeros((1, 1), device=evaluator.device),
                    prompt_emb,
                    ego_cond,
                    chunk_index=cache_chunk,
                ),
                update_cache=2,
                cache_name=evaluator.cache_name,
                action_mode=True,
            )
            if intervention.condition_chunk == 1:
                evaluator.transformer(
                    evaluator._make_action_input(
                        action_cond,
                        torch.zeros((1, 1), device=evaluator.device),
                        prompt_emb,
                        ego_cond,
                        chunk_index=1,
                    ),
                    update_cache=2,
                    cache_name=evaluator.cache_name,
                    action_mode=True,
                )

            future_latent_shape = (
                first_frame_latent.shape[0], first_frame_latent.shape[1], 1,
                first_frame_latent.shape[-2], first_frame_latent.shape[-1],
            )
            num_future_latents = (images.shape[1] - 1) // 4
            for frame_index in range(1, 1 + num_future_latents):
                predicted_latents.append(
                    evaluator._rollout_video_frame(
                        frame_index=frame_index,
                        latent_shape=future_latent_shape,
                        prompt_emb=prompt_emb,
                        ego_cond=ego_cond,
                        num_steps=num_video_steps,
                        denoise_fraction=video_denoise_fraction,
                    )
                )

            # Action prediction is retained as a diagnostic output. It is not
            # used to define the external intervention.
            future_action = evaluator._rollout_action_chunk(
                chunk_index=1,
                action_shape=(
                    action_cond.shape[0], action_cond.shape[1], 1,
                    action_cond.shape[-2], action_cond.shape[-1],
                ),
                action_mask=batch["actions_mask"][:1, :, 1:2].bool().to(evaluator.device),
                prompt_emb=prompt_emb,
                ego_cond=ego_cond,
                num_steps=num_action_steps,
            )
        finally:
            evaluator.transformer.clear_cache(evaluator.cache_name)

    return torch.cat([zero_action, future_action], dim=2), torch.cat(predicted_latents, dim=2)

"""Load a trained student and generate reverse-time videos."""

from __future__ import annotations

from typing import Optional

import torch

from . import student as student_module
from . import teacher as teacher_module
from .methods import METHODS
from .utils import CheckpointManager, seed_everything


def load_student_for_inference(
    method_name: str,
    checkpoint_dir: str,
    hf_repo_id: Optional[str] = None,
    model_id: str = teacher_module.MODEL_ID,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[list] = None,
    rotate_layers: Optional[str] = None,
):
    """Load the teacher pipeline, student adapter, method, and checkpoint.

    `lora_rank`/`lora_alpha`/`target_modules`/`rotate_layers` default to
    whatever `scripts/train.py` saved into the checkpoint's metadata (its
    full training config), so an `attn_rotation_all` checkpoint is never
    accidentally loaded with the `attn_rotation` default (`up_attn1`) —
    explicit args here still override the saved config if passed.
    """
    checkpoint_manager = CheckpointManager(checkpoint_dir, repo_id=hf_repo_id)
    checkpoint_payload = checkpoint_manager.resume()
    if checkpoint_payload is None:
        raise RuntimeError(
            f"No checkpoint found for method '{method_name}' in "
            f"{checkpoint_dir} or HF repo {hf_repo_id}"
        )

    state_dict = checkpoint_payload.get("model")
    if state_dict is None:
        raise RuntimeError(
            f"Checkpoint payload for method '{method_name}' "
            "does not contain a 'model' key"
        )

    saved_config = checkpoint_payload.get("metadata", {}).get("config") or {}
    lora_rank = saved_config.get("lora_rank", lora_rank)
    lora_alpha = saved_config.get("lora_alpha", lora_alpha)
    target_modules = target_modules or saved_config.get("target_modules")
    rotate_layers = rotate_layers or saved_config.get("rotate_layers", "none")

    teacher_pipeline = teacher_module.load_teacher(model_id, device, dtype)
    student = student_module.build_student(
        teacher_pipeline,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    )
    student = METHODS[method_name]().apply(student, rotate_layers=rotate_layers)

    student_module.load_lora_state_dict(student, state_dict)
    student.eval()
    return teacher_pipeline, student


def generate(
    prompt: str,
    teacher_pipeline,
    student,
    negative_prompt: str = "",
    num_frames: int = 16,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 50,
    guidance_scale: float = 9.0,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> dict:
    """Generate a reverse-time video with prompt encoding, U-Net denoising, and VAE decoding."""
    if seed is not None:
        seed_everything(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    scheduler = teacher_pipeline.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Text transformation: prompt strings -> text embeddings.
    prompt_embeds, negative_prompt_embeds = teacher_module.encode_prompt(
        teacher_pipeline,
        prompt,
        negative_prompt=negative_prompt,
        device=device,
    )

    latent_dtype = teacher_pipeline.unet.dtype
    latents = torch.randn(
        (1, 4, num_frames, height // 8, width // 8),
        device=device,
        dtype=latent_dtype,
        generator=generator,
    )
    latents = latents * scheduler.init_noise_sigma

    do_cfg = guidance_scale > 1.0
    prompt_embeds = prompt_embeds.to(device=device, dtype=latent_dtype)
    if do_cfg:
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=latent_dtype)
        embeds_batch = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # U-Net transformation: noisy latents -> predicted noise -> denoised latents.
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            model_input = scheduler.scale_model_input(latents, timestep)

            if do_cfg:
                # Batch the conditional and unconditional passes into one
                # forward call instead of two sequential ones.
                model_input_batch = torch.cat([model_input, model_input], dim=0)
                timestep_batch = timestep.expand(model_input_batch.shape[0])
                noise_pred_batch = student_module.predict_noise(
                    student,
                    model_input_batch,
                    timestep_batch,
                    embeds_batch,
                )
                negative_noise_pred, noise_pred = noise_pred_batch.chunk(2)
                noise_pred = negative_noise_pred + guidance_scale * (
                    noise_pred - negative_noise_pred
                )
            else:
                timestep_batch = timestep.expand(model_input.shape[0])
                noise_pred = student_module.predict_noise(
                    student,
                    model_input,
                    timestep_batch,
                    prompt_embeds,
                )

            latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # VAE transformation: latent video -> pixel-space video frames.
    video = teacher_module.decode_latents_to_video(teacher_pipeline, latents)
    return {"latents": latents, "video": video}

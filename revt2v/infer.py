"""Load a trained student and generate reverse-time videos."""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Callable, Optional, Sequence

import torch

from . import student as student_module
from . import teacher as teacher_module
from .data import flip_latents_time_axis
from .methods import METHODS, attn_injection
from .methods.conv_mirror import ALL_BLOCKS, check_flip_request, set_flip_state
from .utils import CheckpointManager, seed_everything

logger = logging.getLogger(__name__)


def _resume_checkpoint(
    method_name: str,
    checkpoint_dir: str,
    hf_repo_id: Optional[str] = None,
    hub_subfolder: Optional[str] = None,
) -> Optional[dict]:
    """Load a method's checkpoint from `<hub_subfolder or method>/latest.pt`.
    Raises if the checkpoint found was saved by a different method."""
    candidates = [hub_subfolder or method_name]

    for subfolder in candidates:
        payload = CheckpointManager(checkpoint_dir, repo_id=hf_repo_id, hub_subfolder=subfolder).resume()
        if payload is None:
            continue
        saved_method = payload.get("metadata", {}).get("method")
        if saved_method not in (None, method_name):
            raise RuntimeError(
                f"Checkpoint at {subfolder or 'repo root'} was trained with "
                f"method '{saved_method}', not '{method_name}'"
            )
        return payload
    return None


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
    hub_subfolder: Optional[str] = None,
):
    """Load the teacher pipeline, student adapter, method, and checkpoint.

    `lora_rank`/`lora_alpha`/`target_modules`/`rotate_layers` default to
    whatever `scripts/train.py` saved into the checkpoint's metadata (its
    full training config), so a `rotate_layers: all` checkpoint is never
    accidentally loaded with a narrower default — explicit args here still
    override the saved config if passed.
    """
    checkpoint_payload = _resume_checkpoint(method_name, checkpoint_dir, hf_repo_id, hub_subfolder)
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
    student = METHODS[method_name]().apply(
        student,
        rotate_layers=rotate_layers,
        flip_blocks=check_flip_request(None, checkpoint_payload.get("metadata", {})),
    )

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
    latents: Optional[torch.Tensor] = None,
    on_step: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Generate a reverse-time video with prompt encoding, U-Net denoising, and VAE decoding.

    `latents` is optional initial noise of shape (1, 4, F, H/8, W/8), before
    `init_noise_sigma` scaling; pass the same tensor to compare methods from
    identical noise. If None, it's drawn from `seed`.

    `on_step(i, total_steps)`, if given, is called once per denoising step
    (i from 0) — e.g. to report progress to a caller that can't otherwise
    see inside this loop.
    """
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
    if latents is None:
        latents = torch.randn(
            (1, 4, num_frames, height // 8, width // 8),
            device=device,
            dtype=latent_dtype,
            generator=generator,
        )
    else:
        latents = latents.to(device=device, dtype=latent_dtype)
    latents = latents * scheduler.init_noise_sigma

    do_cfg = guidance_scale > 1.0
    prompt_embeds = prompt_embeds.to(device=device, dtype=latent_dtype)
    if do_cfg:
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=latent_dtype)
        embeds_batch = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # U-Net transformation: noisy latents -> predicted noise -> denoised latents.
    total_steps = len(scheduler.timesteps)
    with torch.no_grad():
        for i, timestep in enumerate(scheduler.timesteps):
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

            if on_step is not None:
                on_step(i, total_steps)

    # VAE transformation: latent video -> pixel-space video frames.
    video = teacher_module.decode_latents_to_video(teacher_pipeline, latents)
    return {"latents": latents, "video": video}


class MethodBank:
    """Every inference method on ONE shared U-Net (fits a 16 GB T4).

    Methods: "teacher", "conv_oracle", "attn_injection" (always available --
    the last one runs zero-shot until a checkpoint is loaded), and, if their
    checkpoints exist, "conv_lora" and "attn_lora". Switching enables/disables
    LoRA adapters, swaps attention processors, and flips/unflips temporal
    conv kernels; the clean teacher state is restored after every call.
    """

    def __init__(
        self,
        hf_repo_id: Optional[str] = None,
        checkpoint_root: str = "results/checkpoints",
        model_id: str = teacher_module.MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = device
        self.hf_repo_id = hf_repo_id
        self.checkpoint_root = checkpoint_root
        self.pipeline = teacher_module.load_teacher(model_id, device, dtype)
        self.student = None  # PeftModel once the first adapter is loaded
        self.adapters = {}  # method name -> checkpoint metadata
        self.methods = ["teacher", "conv_oracle", "attn_injection"]

    def load_adapter(self, method: str, hub_subfolder: Optional[str] = None) -> bool:
        """Load "conv_lora", "attn_lora" or "attn_injection" as a named LoRA
        adapter. Returns False (and skips it, leaving "attn_injection"
        running zero-shot) if no checkpoint exists -- or, for
        "attn_injection" specifically, if one exists but its metadata
        doesn't confirm it was trained with the injection mechanism active
        (e.g. someone pointed this at an old plain-LoRA checkpoint by
        mistake)."""
        training_method = {
            "conv_lora": "conv_lora",
            "attn_lora": "attn_lora",
            "attn_injection": "attn_injection",
        }[method]
        payload = _resume_checkpoint(
            training_method,
            f"{self.checkpoint_root}/{training_method}",
            self.hf_repo_id,
            hub_subfolder,
        )
        if payload is None:
            return False

        metadata = payload.get("metadata", {})
        if method == "attn_injection" and not metadata.get("injection"):
            logger.warning(
                "checkpoint at %s/latest.pt doesn't have metadata['injection']=True -- "
                "refusing to load it for attn_injection (it may have been trained without "
                "the injection mechanism active); staying zero-shot",
                hub_subfolder or training_method,
            )
            return False

        config = metadata.get("config") or {}
        host = SimpleNamespace(unet=self.student) if self.student is not None else self.pipeline
        self.student = student_module.build_student(
            host,
            lora_rank=config.get("lora_rank", 8),
            lora_alpha=config.get("lora_alpha", 16),
            target_modules=config.get("target_modules"),
            adapter_name=method,
        )
        student_module.load_lora_state_dict(self.student, payload["model"], adapter_name=method)
        self.student.eval()
        self.student.requires_grad_(False)
        self.adapters[method] = metadata
        if method not in self.methods:
            self.methods.append(method)
        return True

    @contextmanager
    def _method_state(self, method: str, flip_blocks: Optional[Sequence[str]] = None):
        if method not in self.methods:
            raise ValueError(f"Unknown or unloaded method {method!r}, available: {self.methods}")

        unet = self.pipeline.unet
        processors = dict(unet.attn_processors)
        try:
            if method in ("teacher", "conv_oracle"):
                adapters_off = self.student.disable_adapter() if self.student is not None else nullcontext()
                with adapters_off:
                    if method == "conv_oracle":
                        set_flip_state(unet, ALL_BLOCKS if flip_blocks is None else flip_blocks)
                    yield self.student if self.student is not None else unet
            else:
                metadata = self.adapters[method]
                self.student.set_adapter(method)
                set_flip_state(unet, check_flip_request(flip_blocks, metadata))
                yield self.student
        finally:
            set_flip_state(unet, ())
            unet.set_attn_processor(processors)

    def generate(
        self,
        prompt: str,
        method: str,
        seed: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        flip_blocks: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> dict:
        """Run `generate` with `method` active. `flip_blocks` overrides the
        oracle's flips (default: all) or must match a conv student's
        trained flips; other kwargs go to `generate`."""
        if method == "attn_injection":
            if flip_blocks:
                raise ValueError("attn_injection does not take conv flips")
            loaded = "attn_injection" in self.adapters
            model = self.student if loaded else self.pipeline.unet
            if loaded:
                self.student.set_adapter("attn_injection")
            rotate_layers = (self.adapters.get("attn_injection", {}).get("config") or {}).get("rotate_layers", "up_attn1")
            return attn_injection.generate(
                prompt, self.pipeline, model, rotate_layers=rotate_layers, seed=seed, device=self.device, latents=latents, **kwargs
            )
        with self._method_state(method, flip_blocks) as model:
            return generate(prompt, self.pipeline, model, seed=seed, device=self.device, latents=latents, **kwargs)

    def matched_target(self, prompt: str, z: torch.Tensor, **kwargs) -> dict:
        """reverse(teacher(latents=flip(z))): the per-sample reference a
        reverse generator started from noise `z` should reproduce."""
        result = self.generate(prompt, "teacher", latents=flip_latents_time_axis(z), **kwargs)
        return {
            "latents": flip_latents_time_axis(result["latents"]),
            "video": result["video"].flip(-4),  # (..., F, H, W, C)
        }

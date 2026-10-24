"""Load a trained student and generate reverse-time videos."""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from . import student as student_module
from . import teacher as teacher_module
from .data import flip_latents_time_axis
from .methods import METHODS, attn_injection
from .methods.attn_rotation import RotatedTemporalAttnProcessor
from .methods.conv_mirror import ALL_BLOCKS, check_flip_request, set_flip_state
from .utils import CheckpointManager, load_config, seed_everything

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


# ModelScope's U-Net: 9 up-block attn1 layers get rotated for "up_attn1"
# (module-level so tests can monkeypatch it against a toy U-Net with fewer
# up-blocks, same pattern as methods.attn_injection's _UP_ATTN1_COUNT).
_UP_ATTN1_ROTATION_COUNT = 9


def _rotation_processors(unet: Any) -> Dict[str, Any]:
    """Processor dict for attn_rotation's 9 up-block temporal attn1 layers.

    Mirrors `methods.attn_rotation.AttentionRotation.apply`'s own
    "up_attn1" layer selection exactly, reimplemented here because that
    method assumes a peft-wrapped student (it unwraps via `get_base_model`/
    `base_model.model`) -- MethodBank also needs to install/remove these
    processors on the plain teacher U-Net, for `attn_rotation_zeroshot`
    before any adapter has ever been loaded.
    """
    from diffusers import TransformerTemporalModel

    processors: Dict[str, Any] = {}
    for name, module in unet.named_modules():
        if not isinstance(module, TransformerTemporalModel):
            continue
        if not name.startswith("up_blocks."):
            continue
        for block_idx in range(len(module.transformer_blocks)):
            processors[f"{name}.transformer_blocks.{block_idx}.attn1.processor"] = RotatedTemporalAttnProcessor()
    if len(processors) != _UP_ATTN1_ROTATION_COUNT:
        raise RuntimeError(
            f"expected {_UP_ATTN1_ROTATION_COUNT} up-block attn1 layers for attn_rotation, found {len(processors)}"
        )
    return processors


def _infer_lora_config_from_state_dict(
    state_dict: Dict[str, Any], configured_alpha: Optional[int] = None
) -> Tuple[int, int, list]:
    """Recover (rank, alpha, target_modules) from LoRA tensor shapes when a
    checkpoint's metadata doesn't have them. Rank is read off `lora_A`'s
    shape; alpha can't be recovered from the weights themselves (it's only
    a loss-time/merge-time rescale, never stored), so this uses
    `configured_alpha` (e.g. configs/attn_rotation.yaml's `lora_alpha`) if
    given, else alpha=rank -- and always logs which one it assumed."""
    lora_a_keys = [k for k in state_dict if re.search(r"\.lora_A(\.[^.]+)?\.weight$", k)]
    if not lora_a_keys:
        raise RuntimeError("checkpoint has no lora_A tensors -- can't infer rank/target_modules")

    rank = None
    target_modules = set()
    for key in lora_a_keys:
        r = state_dict[key].shape[0]
        if rank is None:
            rank = r
        elif r != rank:
            raise RuntimeError(f"inconsistent LoRA rank across tensors ({rank} vs {r} at {key!r})")
        base = re.sub(r"\.lora_A(\.[^.]+)?\.weight$", "", key)
        base = re.sub(r"^base_model\.model\.", "", base)
        target_modules.add(base)

    alpha = configured_alpha if configured_alpha is not None else rank
    logger.warning(
        "attn_rotation checkpoint metadata is missing lora_rank/lora_alpha/target_modules -- "
        "inferred rank=%d from tensor shapes, assuming alpha=%d (%s), %d target module(s)",
        rank,
        alpha,
        "from configs/attn_rotation.yaml" if configured_alpha is not None else "alpha=rank default, no config found",
        len(target_modules),
    )
    return rank, alpha, sorted(target_modules)


class MethodBank:
    """Every inference method on ONE shared U-Net (fits a 16 GB T4).

    Methods: "teacher", "conv_oracle", "attn_injection" (always available --
    the last one runs zero-shot until a checkpoint is loaded), and, if their
    checkpoints exist, "conv_lora" and "attn_lora". Switching enables/disables
    LoRA adapters, swaps attention processors, and flips/unflips temporal
    conv kernels; the clean teacher state is restored after every call.

    Also "attn_rotation_zeroshot" (always available -- the original
    self-rotation mechanism, no LoRA, research-only: not in the server's
    public method list) and "attn_rotation" (only once
    `load_attn_rotation_adapter()` finds and verifies its checkpoint).
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
        self.methods = ["teacher", "conv_oracle", "attn_injection", "attn_rotation_zeroshot"]

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

    def load_attn_rotation_adapter(self, hub_subfolder: str = "attn_rotation") -> bool:
        """Load the original attn_rotation checkpoint: self-rotation (Q/K
        flipped on the frame axis) + LoRA on the 9 up-block temporal attn1
        layers' to_v/to_out.0. Checks `<hub_subfolder>/latest.pt` first,
        then the legacy repo-root `latest.pt` (where this checkpoint was
        actually saved, before the per-method subfolder convention
        existed) -- prefers the subfolder if both exist.

        Returns False only if no checkpoint is found at all. If one IS
        found but doesn't verify (wrong LoRA shape, an all-zero adapter, no
        measurable effect on the U-Net), this raises instead of silently
        continuing -- "attn_rotation" is never added to `self.methods` on
        a failed verification.
        """
        payload, found_at = None, None
        for subfolder, local_tag in ((hub_subfolder, hub_subfolder), (None, f"{hub_subfolder}_legacy_root")):
            candidate = CheckpointManager(
                f"{self.checkpoint_root}/{local_tag}", repo_id=self.hf_repo_id, hub_subfolder=subfolder
            ).resume()
            if candidate is not None:
                payload, found_at = candidate, (subfolder or "<repo root>")
                break

        if payload is None:
            logger.warning(
                "no attn_rotation checkpoint found (checked %s/latest.pt and the repo root "
                "latest.pt, locally under %s and on HF repo %s)",
                hub_subfolder,
                self.checkpoint_root,
                self.hf_repo_id,
            )
            return False

        state_dict = payload.get("model")
        if state_dict is None:
            raise RuntimeError(f"attn_rotation checkpoint at {found_at} has no 'model' key")

        metadata = payload.get("metadata", {}) or {}
        saved_method = metadata.get("method")
        if saved_method not in (None, "attn_rotation"):
            logger.warning(
                "attn_rotation checkpoint at %s has metadata['method']=%r, not 'attn_rotation' -- "
                "loading it anyway, but double check this is the right checkpoint",
                found_at,
                saved_method,
            )

        config = metadata.get("config") or {}
        saved_rotate_layers = config.get("rotate_layers", "up_attn1")
        if saved_rotate_layers != "up_attn1":
            logger.warning(
                "attn_rotation checkpoint at %s was trained with rotate_layers=%r, not "
                "'up_attn1' -- this MethodBank wiring only installs the 9-layer up_attn1 rotation",
                found_at,
                saved_rotate_layers,
            )

        rank, alpha, target_modules = config.get("lora_rank"), config.get("lora_alpha"), config.get("target_modules")
        if rank is None or alpha is None or not target_modules:
            configured_alpha = None
            config_path = Path("configs/attn_rotation.yaml")
            if config_path.exists():
                try:
                    configured_alpha = load_config(config_path).get("lora_alpha")
                except Exception:
                    configured_alpha = None
            inferred_rank, inferred_alpha, inferred_targets = _infer_lora_config_from_state_dict(
                state_dict, configured_alpha
            )
            rank = rank if rank is not None else inferred_rank
            alpha = alpha if alpha is not None else inferred_alpha
            target_modules = target_modules or inferred_targets

        host = SimpleNamespace(unet=self.student) if self.student is not None else self.pipeline
        self.student = student_module.build_student(
            host, lora_rank=rank, lora_alpha=alpha, target_modules=target_modules, adapter_name="attn_rotation"
        )

        from peft import set_peft_model_state_dict

        load_result = set_peft_model_state_dict(self.student, state_dict, adapter_name="attn_rotation")
        missing = [k for k in getattr(load_result, "missing_keys", []) if "attn_rotation" in k]
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if missing or unexpected:
            raise RuntimeError(
                f"attn_rotation checkpoint at {found_at} didn't load cleanly: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        self.student.eval()
        self.student.requires_grad_(False)

        from peft.tuners.lora import LoraLayer

        rotation_layers = [
            (name, module)
            for name, module in self.student.named_modules()
            if isinstance(module, LoraLayer) and "attn_rotation" in module.lora_A
        ]
        bad_names = [name for name, _ in rotation_layers if not re.search(r"up_blocks.*attn1\.(to_v|to_out\.0)$", name)]
        if len(rotation_layers) != 18 or bad_names:
            raise RuntimeError(
                "attn_rotation adapter verification failed: expected exactly 18 LoRA Linear "
                f"layers in up_blocks attn1 (to_v/to_out.0), found {len(rotation_layers)} "
                f"(unexpected names: {bad_names[:5]})"
            )

        zero_norm = [name for name, module in rotation_layers if module.lora_B["attn_rotation"].weight.norm().item() == 0.0]
        if zero_norm:
            raise RuntimeError(
                f"attn_rotation adapter verification failed: lora_B is all-zero for {zero_norm[:5]} "
                "-- looks untrained"
            )

        unet = self.pipeline.unet
        processors_before = dict(unet.attn_processors)
        try:
            unet.set_attn_processor(_rotation_processors(unet))
            fixed_emb, _ = teacher_module.encode_prompt(self.pipeline, "", device=self.device)
            fixed_emb = fixed_emb.to(unet.dtype)
            fixed_latents = torch.randn((1, 4, 16, 32, 32), device=self.device, dtype=unet.dtype)
            fixed_t = torch.tensor([500], device=self.device)
            with torch.no_grad():
                self.student.set_adapter("attn_rotation")
                out_on = self.student(fixed_latents, fixed_t, encoder_hidden_states=fixed_emb, return_dict=False)[0]
                with self.student.disable_adapter():
                    out_off = self.student(fixed_latents, fixed_t, encoder_hidden_states=fixed_emb, return_dict=False)[0]
            if torch.allclose(out_on, out_off, atol=1e-6):
                raise RuntimeError(
                    "attn_rotation adapter verification failed: U-Net output is identical with "
                    "the adapter enabled vs disabled on a fixed input -- the LoRA appears to have no effect"
                )
        finally:
            unet.set_attn_processor(processors_before)

        self.adapters["attn_rotation"] = metadata
        if "attn_rotation" not in self.methods:
            self.methods.append("attn_rotation")
        logger.info(
            "attn_rotation adapter loaded from %s: rank=%d alpha=%d, %d LoRA layers verified",
            found_at,
            rank,
            alpha,
            len(rotation_layers),
        )
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
        if method in ("attn_rotation", "attn_rotation_zeroshot"):
            if flip_blocks:
                raise ValueError(f"{method} does not take conv flips")
            if method == "attn_rotation" and "attn_rotation" not in self.adapters:
                raise ValueError(
                    "attn_rotation has no loaded checkpoint -- call load_attn_rotation_adapter() "
                    "first, or use 'attn_rotation_zeroshot' instead"
                )
            unet = self.pipeline.unet
            processors_before = dict(unet.attn_processors)
            try:
                unet.set_attn_processor(_rotation_processors(unet))
                if method == "attn_rotation":
                    self.student.set_adapter("attn_rotation")
                    return generate(prompt, self.pipeline, self.student, seed=seed, device=self.device, latents=latents, **kwargs)
                adapters_off = self.student.disable_adapter() if self.student is not None else nullcontext()
                with adapters_off:
                    model = self.student if self.student is not None else unet
                    return generate(prompt, self.pipeline, model, seed=seed, device=self.device, latents=latents, **kwargs)
            finally:
                unet.set_attn_processor(processors_before)
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

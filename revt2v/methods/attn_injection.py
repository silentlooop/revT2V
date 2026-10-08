"""Cross-model attention injection for reverse-time video generation.

Two forward passes through the SAME U-Net per denoising step:

  1. capture: run on flip(x_t) (frame axis), all LoRA adapters disabled,
     stock (un-rotated) attention. Store softmax(QK^T) for each selected
     temporal self-attention layer, plus the pass's own noise-prediction
     output (reused as the mirror-loss target during training -- free,
     since that forward pass already happened).
  2. inject: run on x_t (the real trajectory). In the same layers, ignore
     this pass's own Q/K and instead use the captured map rotated 180
     degrees (`.flip(-1).flip(-2)`) against THIS pass's own V/to_out -- so a
     LoRA on to_v/to_out.0 (trained with this mechanism active) still has
     something to learn, exactly like `AttnLoRA`'s target modules.

Important subtlety, verified in tests/test_attn_injection.py: for an
*isolated* attention layer (or `transformer_in`, the first temporal op in
the U-Net, with nothing direction-aware upstream of it), `to_q`/`to_k` are
per-frame Linears, so capturing on `flip(x_t)` gives exactly
`flip(Q(x_t))`/`flip(K(x_t))`, and the 180-degree rotation on inject exactly
undoes that flip -- the whole two-pass construction reduces algebraically to
PLAIN, un-rotated attention on `x_t`. That's expected, not a bug: it means
injection at `transformer_in` alone is a no-op. The method's real targets
(`rotate_layers="up_attn1"`) sit *downstream* of many direction-aware
temporal convs (unflipped kernels are not time-reversal-equivariant -- this
is the same asymmetry `conv_mirror.py`'s `flip_temporal_convs` exists to
correct for), so by the time a signal reaches an up-block `attn1`, the
capture pass's hidden state there is genuinely *not* the flip of the inject
pass's -- the identity above breaks, and the injected map is a real,
different attention pattern transplanted from "what the model's own
attention looks like on the reversed trajectory, this deep in the network"
onto the real trajectory's own V/to_out.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from diffusers import TransformerTemporalModel

from ..data import flip_latents_time_axis

# Layer counts for rotate_layers, selected by class (TransformerTemporalModel),
# never by name fragment. "up_attn1" and "all" are already-verified counts
# for ModelScope's U-Net; "up_both" (attn1+attn2, up_blocks only) is computed
# at call time instead of hand-guessed -- see temporal_self_attn_names.
_UP_ATTN1_COUNT = 9
_ALL_COUNT = 34
ROTATE_LAYER_CHOICES = ("up_attn1", "up_both", "all")


def _base_unet(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _normalize_rotate_layers(rotate_layers: Any) -> str:
    """scripts/train.py's shared call sites default missing config to
    "none" (meaningful for other methods that can disable their mechanism
    entirely); this one has no "no rotation" mode, so None/""/"none" all
    mean its own default."""
    return rotate_layers if rotate_layers and rotate_layers != "none" else "up_attn1"


def temporal_self_attn_names(unet: Any, rotate_layers: str = "up_attn1") -> List[str]:
    """Full module names (no `.processor` suffix) of the temporal
    self-attention layers selected by `rotate_layers`:
    - "up_attn1": attn1 only, up_blocks only (9 layers).
    - "up_both": attn1 + attn2, up_blocks only. Not a known constant -- counted
      by finding attn1-up and attn2-up separately and asserting they match
      (every TransformerTemporalModel block pairs attn1/attn2 symmetrically).
    - "all": attn1 + attn2, every TransformerTemporalModel incl. transformer_in
      (34 layers).
    """
    if rotate_layers not in ROTATE_LAYER_CHOICES:
        raise ValueError(f"rotate_layers must be one of {ROTATE_LAYER_CHOICES}, got {rotate_layers!r}")
    unet = _base_unet(unet)

    def find(attn_names: Tuple[str, ...], up_only: bool) -> List[str]:
        names = []
        for name, module in unet.named_modules():
            if not isinstance(module, TransformerTemporalModel):
                continue
            if up_only and not name.startswith("up_blocks."):
                continue
            for block_idx in range(len(module.transformer_blocks)):
                for attn_name in attn_names:
                    names.append(f"{name}.transformer_blocks.{block_idx}.{attn_name}")
        return names

    if rotate_layers == "up_attn1":
        names = find(("attn1",), up_only=True)
        if len(names) != _UP_ATTN1_COUNT:
            raise ValueError(f"Expected {_UP_ATTN1_COUNT} up-block temporal attn1 layers, found {len(names)}")
        return names

    if rotate_layers == "up_both":
        attn1_up = find(("attn1",), up_only=True)
        attn2_up = find(("attn2",), up_only=True)
        if len(attn1_up) != len(attn2_up):
            raise ValueError(
                f"Expected attn1/attn2 to pair up symmetrically in up_blocks, found "
                f"{len(attn1_up)} attn1 vs {len(attn2_up)} attn2"
            )
        return attn1_up + attn2_up

    names = find(("attn1", "attn2"), up_only=False)
    if len(names) != _ALL_COUNT:
        raise ValueError(f"Expected {_ALL_COUNT} temporal attention layers for rotate_layers='all', found {len(names)}")
    return names


def inject_layer_targets(unet: Any, rotate_layers: str = "up_attn1") -> List[str]:
    """to_v/to_out.0 full names for the layers `rotate_layers` selects --
    LoRA targets for AttnInjection: only the layers that receive an injected
    attention map get trainable V/output, Q/K stay frozen everywhere."""
    targets = []
    for name in temporal_self_attn_names(unet, rotate_layers):
        targets.append(f"{name}.to_v")
        targets.append(f"{name}.to_out.0")
    return targets


class _CapturingAttnProcessor:
    """Stock temporal self-attention; also stores softmax(QK^T) in `store[name]`."""

    def __init__(self, name: str, store: Dict[str, torch.Tensor]) -> None:
        self.name = name
        self.store = store

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise ValueError("Temporal attention processor expects self-attention")
        residual = hidden_states
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(hidden_states))

        attn_weights = attn.get_attention_scores(query, key, attention_mask)
        self.store[self.name] = attn_weights

        attn_output = attn.batch_to_head_dim(torch.matmul(attn_weights, value))
        attn_output = attn.to_out[1](attn.to_out[0](attn_output))
        if attn.residual_connection:
            attn_output = attn_output + residual
        return attn_output / attn.rescale_output_factor


class _InjectedAttnProcessor:
    """Uses a captured, optionally 180-degree-rotated attention map against
    THIS call's own V/to_out (so this is where a to_v/to_out.0 LoRA acts)."""

    def __init__(self, name: str, store: Dict[str, torch.Tensor], rotate: bool = True) -> None:
        self.name = name
        self.store = store
        self.rotate = rotate

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise ValueError("Temporal attention processor expects self-attention")
        if self.name not in self.store:
            raise RuntimeError(f"no captured attention map for {self.name!r} -- run the capture pass first")
        residual = hidden_states
        value = attn.head_to_batch_dim(attn.to_v(hidden_states))

        attn_weights = self.store[self.name]
        if self.rotate:
            attn_weights = attn_weights.flip(-1).flip(-2)
        if attn_weights.shape[0] != value.shape[0]:
            raise RuntimeError(
                f"batch mismatch at {self.name!r}: captured map batch {attn_weights.shape[0]} "
                f"!= this pass's batch {value.shape[0]} (capture and inject must share batch order)"
            )

        attn_output = attn.batch_to_head_dim(torch.matmul(attn_weights, value))
        attn_output = attn.to_out[1](attn.to_out[0](attn_output))
        if attn.residual_connection:
            attn_output = attn_output + residual
        return attn_output / attn.rescale_output_factor


def capture_and_inject(
    unet: Any,
    student_for_v: Any,
    batch: torch.Tensor,
    timestep_batch: torch.Tensor,
    embeds: torch.Tensor,
    store: Dict[str, torch.Tensor],
    layer_names: List[str],
    rotate: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the capture pass (flip(batch), LoRA disabled, stock attention,
    no_grad) then the inject pass (batch, whatever adapter is currently
    enabled on `student_for_v`, the just-captured maps rotated 180 degrees).
    Returns (inject_noise_pred, capture_noise_pred) -- the capture output is
    needed by the training mirror loss "for free" (no extra forward pass).

    `unet` and `student_for_v` are usually the same object (a peft-wrapped
    U-Net: `student_for_v` lets us call `.disable_adapter()`/`.set_adapter()`
    while `unet` is where processors get installed) -- `student_for_v` may
    lack `disable_adapter` entirely for the zero-shot (no LoRA at all) case.
    """
    stock = dict(unet.attn_processors)
    capture_processors = {f"{n}.processor": _CapturingAttnProcessor(n, store) for n in layer_names}
    inject_processors = {f"{n}.processor": _InjectedAttnProcessor(n, store, rotate=rotate) for n in layer_names}
    adapters_off = student_for_v.disable_adapter() if hasattr(student_for_v, "disable_adapter") else nullcontext()

    was_training = unet.training
    unet.eval()  # deterministic capture pass (matches conv_mirror.mirrored_teacher_noise)
    try:
        unet.set_attn_processor({**stock, **capture_processors})
        with torch.no_grad(), adapters_off:
            capture_noise_pred = unet(flip_latents_time_axis(batch), timestep_batch, encoder_hidden_states=embeds, return_dict=False)[0]
    finally:
        unet.train(was_training)

    unet.set_attn_processor({**stock, **inject_processors})
    inject_noise_pred = unet(batch, timestep_batch, encoder_hidden_states=embeds, return_dict=False)[0]

    return inject_noise_pred, capture_noise_pred


def generate(
    prompt: str,
    teacher_pipeline: Any,
    student: Any,
    rotate_layers: str = "up_attn1",
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
    """Zero-shot (or LoRA-on-top, via `student`) reverse-time generation
    through attention injection -- see the module docstring for the
    mechanism. `student` is the model whose V/to_out the inject pass uses:
    pass the plain teacher U-Net for zero-shot, or a peft-wrapped student
    with its "attn_injection" adapter set for the trained variant.
    """
    from .. import teacher as teacher_module
    from ..utils import seed_everything

    if seed is not None:
        seed_everything(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    scheduler = teacher_pipeline.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    prompt_embeds, negative_prompt_embeds = teacher_module.encode_prompt(
        teacher_pipeline, prompt, negative_prompt=negative_prompt, device=device,
    )

    latent_dtype = teacher_pipeline.unet.dtype
    if latents is None:
        latents = torch.randn((1, 4, num_frames, height // 8, width // 8), device=device, dtype=latent_dtype, generator=generator)
    else:
        latents = latents.to(device=device, dtype=latent_dtype)
    latents = latents * scheduler.init_noise_sigma

    do_cfg = guidance_scale > 1.0
    prompt_embeds = prompt_embeds.to(device=device, dtype=latent_dtype)
    if do_cfg:
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=latent_dtype)
        embeds_batch = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    unet = _base_unet(student)
    layer_names = temporal_self_attn_names(unet, rotate_layers)
    stock_processors = dict(unet.attn_processors)
    store: Dict[str, torch.Tensor] = {}

    total_steps = len(scheduler.timesteps)
    try:
        with torch.no_grad():
            for i, timestep in enumerate(scheduler.timesteps):
                model_input = scheduler.scale_model_input(latents, timestep)
                # Same tensor for both passes -> capture/inject batch order matches automatically.
                batch = torch.cat([model_input, model_input], dim=0) if do_cfg else model_input
                timestep_batch = timestep.expand(batch.shape[0])
                embeds = embeds_batch if do_cfg else prompt_embeds

                noise_pred, _ = capture_and_inject(unet, student, batch, timestep_batch, embeds, store, layer_names)
                store.clear()

                if do_cfg:
                    negative_noise_pred, cond_noise_pred = noise_pred.chunk(2)
                    noise_pred = negative_noise_pred + guidance_scale * (cond_noise_pred - negative_noise_pred)

                latents = scheduler.step(noise_pred, timestep, latents).prev_sample

                if on_step is not None:
                    on_step(i, total_steps)
    finally:
        unet.set_attn_processor(stock_processors)

    video = teacher_module.decode_latents_to_video(teacher_pipeline, latents)
    return {"latents": latents, "video": video}


class AttnInjection:
    """Trainable attention-injection method: capture the base U-Net's own
    attention on the time-flipped latent, inject it (rotated 180 degrees)
    into the real pass, and train a to_v/to_out.0 LoRA on top of that."""

    def lora_targets(self, unet: Any, rotate_layers: Any = None) -> List[str]:
        return inject_layer_targets(unet, _normalize_rotate_layers(rotate_layers))

    def apply(self, student: Any, rotate_layers: Any = "up_attn1", flip_blocks: Any = None, **kwargs: Any) -> Any:
        """Install the inject processors permanently (so the uniform
        `student_module.predict_noise()` call in scripts/train.py's loop
        naturally uses them) and remember the layer set for `prepare_step`.
        Never flips convs: refuse a student asked to."""
        if flip_blocks:
            raise ValueError(f"{type(self).__name__} takes no flip_blocks, got {flip_blocks}")
        rotate_layers = _normalize_rotate_layers(rotate_layers)
        unet = _base_unet(student)
        self._rotate_layers = rotate_layers
        self._layer_names = temporal_self_attn_names(unet, rotate_layers)
        self._store: Dict[str, torch.Tensor] = {}
        self._last_captured_noise_pred: Optional[torch.Tensor] = None
        stock = dict(unet.attn_processors)
        inject_processors = {f"{n}.processor": _InjectedAttnProcessor(n, self._store) for n in self._layer_names}
        unet.set_attn_processor({**stock, **inject_processors})
        return student

    def prepare_step(self, student: Any, noisy_latents: torch.Tensor, timesteps: torch.Tensor, encoder_hidden_states: torch.Tensor) -> None:
        """Called once per micro-batch, right before scripts/train.py's
        uniform `predict_noise()` call: runs the capture pass (populating
        `self._store`, which the already-installed inject processors read
        from) and stashes the capture pass's own output for `loss()`'s
        mirror term -- the inject-and-predict step itself happens via that
        following uniform `predict_noise()` call, not here."""
        unet = _base_unet(student)
        self._store.clear()
        stock = dict(unet.attn_processors)
        capture_processors = {f"{n}.processor": _CapturingAttnProcessor(n, self._store) for n in self._layer_names}
        was_training = unet.training
        unet.eval()
        try:
            unet.set_attn_processor({**stock, **capture_processors})
            with torch.no_grad(), student.disable_adapter():
                self._last_captured_noise_pred = unet(
                    flip_latents_time_axis(noisy_latents), timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False
                )[0]
        finally:
            unet.train(was_training)
            inject_processors = {f"{n}.processor": _InjectedAttnProcessor(n, self._store) for n in self._layer_names}
            unet.set_attn_processor({**stock, **inject_processors})

    def loss(
        self,
        predicted_noise: Any,
        target_noise: Any,
        mirror_loss_weight: float = 0.0,
        **extra: Any,
    ) -> Any:
        """eps-MSE + `mirror_loss_weight` * MSE to flip(capture_pass_output),
        in fp32 -- same formula as conv_lora.mirror_loss, but reusing
        prepare_step's capture output instead of a third forward pass."""
        loss = torch.nn.functional.mse_loss(predicted_noise.float(), target_noise.float())
        if mirror_loss_weight > 0:
            if self._last_captured_noise_pred is None:
                raise ValueError("mirror loss needs prepare_step to have run first")
            mirror_target = flip_latents_time_axis(self._last_captured_noise_pred)
            loss = loss + mirror_loss_weight * torch.nn.functional.mse_loss(predicted_noise.float(), mirror_target.float())
        return loss

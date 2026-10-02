"""Mirror ModelScope's temporal convolutions in time for reverse-time video generation.

Flipping every temporal Conv3d kernel (kernel (3, 1, 1), time = weight dim 2)
turns the U-Net into an exact time-mirror of the teacher:

    unet_flipped(z, t, emb) == flip(unet(flip(z), t, emb))

Temporal attention needs no change: it has no positional encoding, so it is
order-agnostic and already commutes with a frame flip. Rotating it on top of
flipped convs would be wrong.

Two modes (config `use_trained_weights`):
- false -> **oracle**: teacher weights, every temporal conv flipped, no LoRA,
  no training. Equivalent to reversing the teacher's output, so it's a
  reference/upper bound, not a learned method.
- true  -> **conv student**: LoRA on the temporal Conv3d layers of the
  (by default UNflipped) teacher, trained on reversed latents, optionally with
  a mirrored-teacher loss. `flip_blocks` flips some blocks before training
  ("partial flip + LoRA"); it is saved in the checkpoint and re-applied at
  inference.

Flips are all-or-nothing per layer: blends like 0.5*W + 0.5*flip(W) break
the model, so there's deliberately no blending parameter anywhere here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from diffusers.models.resnet import TemporalConvLayer

from ..data import flip_latents_time_axis

logger = logging.getLogger(__name__)

BLOCK_PREFIXES = {"down": "down_blocks.", "mid": "mid_block.", "up": "up_blocks."}
ALL_BLOCKS: Tuple[str, ...] = ("down", "mid", "up")

# ModelScope's U-Net: 22 TemporalConvLayers x 4 Conv3d each (32 down, 8 mid,
# 48 up). Sanity-checked like AttentionRotation.LAYER_COUNTS.
TEMPORAL_CONV_COUNT = 88

# Attribute on the base U-Net recording which blocks are currently flipped,
# so flips can be undone and double flips caught.
_FLIP_STATE_ATTR = "_revt2v_flipped_blocks"


def _base_unet(model: Any) -> Any:
    """Return the plain U-Net under a peft wrapper (or `model` itself)."""
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def _block_of(name: str) -> Optional[str]:
    for block, prefix in BLOCK_PREFIXES.items():
        if name.startswith(prefix):
            return block
    return None


def _check_blocks(blocks: Iterable[str]) -> Tuple[str, ...]:
    blocks = tuple(blocks or ())
    unknown = [b for b in blocks if b not in BLOCK_PREFIXES]
    if unknown:
        raise ValueError(f"flip blocks must be a subset of {list(ALL_BLOCKS)}, got {unknown}")
    return tuple(b for b in ALL_BLOCKS if b in blocks)


def find_temporal_convs(unet: Any) -> List[str]:
    """Return the names of every `nn.Conv3d` inside a `TemporalConvLayer`.

    Selected by class, not by name fragments, so GroupNorm/SiLU/Dropout in
    the same Sequential are never included. After peft wraps a conv, the
    weight-holding Conv3d is `<name>.base_layer`; peft's own `lora_A`/`lora_B`
    are also Conv3d (lora_A even has a (3, 1, 1) kernel) and are skipped.
    """
    unet = _base_unet(unet)
    names = []
    for layer_name, layer in unet.named_modules():
        if not isinstance(layer, TemporalConvLayer):
            continue
        for sub_name, module in layer.named_modules():
            if not isinstance(module, torch.nn.Conv3d):
                continue
            if any(part.startswith("lora_") for part in sub_name.split(".")):
                continue
            names.append(f"{layer_name}.{sub_name}")
    logger.info("Found %d temporal Conv3d layers", len(names))
    return names


def conv_lora_targets(unet: Any) -> List[str]:
    """Full module names of the temporal Conv3d layers, for peft's `target_modules`."""
    return [name.removesuffix(".base_layer") for name in find_temporal_convs(unet)]


def attention_lora_targets(unet: Any) -> List[str]:
    """Temporal attention Linears to also give LoRA in the conv student.

    TODO (you fill this in): return full module names of the temporal
    attention projections to train alongside the convs, e.g. the `to_v` and
    `to_out.0` of attn1/attn2 under each `TransformerTemporalModel`
    (selected by class, like `AttentionRotation.apply`). Returning [] keeps
    the conv student conv-only.
    """
    return []


def flipped_blocks(unet: Any) -> Tuple[str, ...]:
    """Blocks whose temporal convs are currently flipped."""
    return tuple(getattr(_base_unet(unet), _FLIP_STATE_ATTR, ()))


def flip_temporal_convs(unet: Any, blocks: Sequence[str] = ALL_BLOCKS) -> List[str]:
    """Flip temporal conv kernels in time for the selected blocks, in place.

    Each kernel becomes exactly `weight.flip(2)` — never a blend. Flipping
    the same blocks again restores the original weights. Returns the names
    of the flipped convs.
    """
    blocks = _check_blocks(blocks)
    unet = _base_unet(unet)
    modules = dict(unet.named_modules())
    flipped = []
    with torch.no_grad():
        for name in find_temporal_convs(unet):
            if _block_of(name) not in blocks:
                continue
            weight = modules[name].weight
            weight.copy_(weight.flip(2))
            flipped.append(name)

    current = set(getattr(unet, _FLIP_STATE_ATTR, ()))
    setattr(unet, _FLIP_STATE_ATTR, _check_blocks(current.symmetric_difference(blocks)))
    return flipped


def set_flip_state(unet: Any, blocks: Sequence[str]) -> None:
    """Flip/unflip so that exactly `blocks` end up flipped."""
    target = set(_check_blocks(blocks))
    toggle = target.symmetric_difference(flipped_blocks(unet))
    if toggle:
        flip_temporal_convs(unet, tuple(toggle))


def checkpoint_flip_blocks(metadata: Dict[str, Any]) -> Tuple[str, ...]:
    """The flip config a conv-student checkpoint was trained with."""
    if "flip_blocks" in metadata:
        return _check_blocks(metadata["flip_blocks"])
    return _check_blocks((metadata.get("config") or {}).get("flip_blocks") or ())


def check_flip_request(requested: Optional[Sequence[str]], metadata: Dict[str, Any]) -> Tuple[str, ...]:
    """Return the flips to apply for a trained conv student, refusing any
    flip that differs from what it was trained with.

    A student trained without flips already generates reverse time; flipping
    its convs on top would reverse it again (back to forward time).
    """
    trained = checkpoint_flip_blocks(metadata)
    if requested is None:
        return trained
    requested = _check_blocks(requested)
    if set(requested) != set(trained):
        raise ValueError(
            f"Refusing to flip blocks {list(requested)} on a conv student "
            f"trained with flip_blocks={list(trained)}: extra flips on a "
            "trained student double-reverse time. Use method 'conv_oracle' "
            "for the flipped teacher instead."
        )
    return trained


def require_conv3d_lora() -> None:
    """Fail clearly if the installed peft can't put LoRA on Conv3d."""
    from peft.tuners.lora import layer

    if not hasattr(layer, "Conv3d"):
        import peft

        raise ImportError(
            f"peft {peft.__version__} has no Conv3d LoRA; conv_mirror needs "
            "peft>=0.21 (pip install -U peft)"
        )


def mirrored_teacher_noise(
    student: Any,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """flip(eps_teacher(flip(x_t))): the teacher (LoRA disabled, all convs
    unflipped, eval mode) run on time-flipped input, output flipped back."""
    unet = _base_unet(student)
    flips = flipped_blocks(unet)
    was_training = unet.training
    unet.eval()  # TemporalConvLayer has Dropout(0.1); the target must be deterministic
    try:
        set_flip_state(unet, ())
        with torch.no_grad(), student.disable_adapter():
            teacher_noise = unet(
                flip_latents_time_axis(noisy_latents),
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
    finally:
        set_flip_state(unet, flips)
        unet.train(was_training)
    return flip_latents_time_axis(teacher_noise)


class ConvMirror:
    """Flip temporal conv kernels (oracle) or LoRA-tune them (conv student)."""

    def lora_targets(self, unet: Any) -> List[str]:
        """LoRA targets for the conv student: temporal Conv3d layers plus
        whatever `attention_lora_targets` returns."""
        require_conv3d_lora()
        targets = conv_lora_targets(unet)
        if len(targets) != TEMPORAL_CONV_COUNT:
            raise ValueError(f"Expected {TEMPORAL_CONV_COUNT} temporal Conv3d layers, found {len(targets)}")
        return targets + attention_lora_targets(unet)

    def apply(
        self,
        student: Any,
        flip_blocks: Optional[Sequence[str]] = None,
        use_trained_weights: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Flip temporal convs and return the student.

        - use_trained_weights=False (oracle): flip all blocks.
        - use_trained_weights=True (student): flip exactly `flip_blocks`
          (default none), the same set it was/will be trained with.
        """
        blocks = ALL_BLOCKS if not use_trained_weights else (flip_blocks or ())
        set_flip_state(student, blocks)
        return student

    def loss(
        self,
        predicted_noise: Any,
        target_noise: Any,
        mirror_loss_weight: float = 0.0,
        student: Any = None,
        noisy_latents: Any = None,
        timesteps: Any = None,
        encoder_hidden_states: Any = None,
        **extra: Any,
    ) -> Any:
        """eps-MSE plus `mirror_loss_weight` * MSE to the mirrored teacher."""
        loss = torch.nn.functional.mse_loss(predicted_noise.float(), target_noise.float())
        if mirror_loss_weight > 0:
            if student is None or encoder_hidden_states is None:
                raise ValueError("mirror loss needs student and encoder_hidden_states")
            mirror_target = mirrored_teacher_noise(student, noisy_latents, timesteps, encoder_hidden_states)
            loss = loss + mirror_loss_weight * torch.nn.functional.mse_loss(
                predicted_noise.float(), mirror_target.float()
            )
        return loss

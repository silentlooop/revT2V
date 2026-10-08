"""LoRA on the UNflipped teacher's temporal layers, trained toward reverse time.

The student starts as the forward teacher (frozen ModelScope weights + zero
LoRA) and never flips a conv kernel or rotates attention, in training or at
inference. It learns reverse time from

    eps-MSE on noisy reversed latents
    + mirror_loss_weight * MSE(eps_student(x_t), flip(eps_teacher(flip(x_t))))

where the teacher is the same U-Net with LoRA disabled, under no_grad
(`conv_mirror.mirrored_teacher_noise`). Frame axis = dim 2 of (B, 4, F, H, W).

Two configs share this loss so they're directly comparable:
- `conv_lora` (main): LoRA on every Conv3d inside every TemporalConvLayer
  (88) + to_v/to_out.0 of every temporal attention layer (68 Linears).
- `attn_lora` (baseline): the 68 temporal attention Linears only.

Q/K stay frozen in both.
"""

from __future__ import annotations

from typing import Any, List

import torch
from diffusers import TransformerTemporalModel

from .conv_mirror import (
    TEMPORAL_CONV_COUNT,
    _base_unet,
    conv_lora_targets,
    flipped_blocks,
    mirrored_teacher_noise,
    require_conv3d_lora,
)

# 17 TransformerTemporalModels (transformer_in + 16 temp_attentions) x
# (attn1, attn2) x (to_v, to_out.0).
TEMPORAL_ATTN_LINEAR_COUNT = 68
ATTN_PROJECTIONS = ("to_v", "to_out.0")


def temporal_attention_lora_targets(unet: Any) -> List[str]:
    """Full names of to_v/to_out.0 in attn1+attn2 of every
    TransformerTemporalModel (incl. `transformer_in`), selected by class."""
    unet = _base_unet(unet)
    names = []
    for name, module in unet.named_modules():
        if not isinstance(module, TransformerTemporalModel):
            continue
        for block_idx in range(len(module.transformer_blocks)):
            for attn_name in ("attn1", "attn2"):
                for proj in ATTN_PROJECTIONS:
                    names.append(f"{name}.transformer_blocks.{block_idx}.{attn_name}.{proj}")
    return names


def mirror_loss(
    predicted_noise: Any,
    target_noise: Any,
    mirror_loss_weight: float = 0.0,
    student: Any = None,
    noisy_latents: Any = None,
    timesteps: Any = None,
    encoder_hidden_states: Any = None,
) -> Any:
    """eps-MSE + `mirror_loss_weight` * MSE to flip(eps_teacher(flip(x_t))), in fp32."""
    loss = torch.nn.functional.mse_loss(predicted_noise.float(), target_noise.float())
    if mirror_loss_weight > 0:
        if student is None or encoder_hidden_states is None:
            raise ValueError("mirror loss needs student and encoder_hidden_states")
        mirror_target = mirrored_teacher_noise(student, noisy_latents, timesteps, encoder_hidden_states)
        loss = loss + mirror_loss_weight * torch.nn.functional.mse_loss(
            predicted_noise.float(), mirror_target.float()
        )
    return loss


class AttnLoRA:
    """Baseline: LoRA on temporal attention to_v/to_out.0 only."""

    def lora_targets(self, unet: Any, rotate_layers: Any = None) -> List[str]:
        targets = temporal_attention_lora_targets(unet)
        if len(targets) != TEMPORAL_ATTN_LINEAR_COUNT:
            raise ValueError(
                f"Expected {TEMPORAL_ATTN_LINEAR_COUNT} temporal attention Linears, found {len(targets)}"
            )
        return targets

    def apply(self, student: Any, **kwargs: Any) -> Any:
        """No flips, no processor swaps: refuse a student that has either."""
        if flipped_blocks(student):
            raise ValueError(f"{type(self).__name__} never flips convs, found flipped {flipped_blocks(student)}")
        if kwargs.get("flip_blocks"):
            raise ValueError(f"{type(self).__name__} takes no flip_blocks, got {kwargs['flip_blocks']}")
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
        return mirror_loss(
            predicted_noise,
            target_noise,
            mirror_loss_weight=mirror_loss_weight,
            student=student,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )


class ConvLoRA(AttnLoRA):
    """Main method: LoRA on the 88 temporal Conv3d + the 68 temporal attention Linears."""

    def lora_targets(self, unet: Any, rotate_layers: Any = None) -> List[str]:
        require_conv3d_lora()
        convs = conv_lora_targets(unet)
        if len(convs) != TEMPORAL_CONV_COUNT:
            raise ValueError(f"Expected {TEMPORAL_CONV_COUNT} temporal Conv3d layers, found {len(convs)}")
        return convs + super().lora_targets(unet)

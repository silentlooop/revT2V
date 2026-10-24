"""Rotate temporal self-attention for reverse-time video generation."""

from __future__ import annotations

from typing import Any, Optional

import torch
from diffusers import TransformerTemporalModel



class RotatedTemporalAttnProcessor:
    """Apply reversed queries and keys to temporal self-attention."""

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        temb: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Run temporal self-attention with reversed frame queries and keys."""

        residual = hidden_states
        if encoder_hidden_states is not None:
            raise ValueError("Temporal attention processor expects self-attention")

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        query = torch.flip(query, dims=[1])
        key = torch.flip(key, dims=[1])

        attn_weights = attn.get_attention_scores(query, key, attention_mask)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn.batch_to_head_dim(attn_output)
        attn_output = attn.to_out[0](attn_output)
        attn_output = attn.to_out[1](attn_output)

        if attn.residual_connection:
            attn_output = attn_output + residual
        attn_output = attn_output / attn.rescale_output_factor

        return attn_output


class AttentionRotation:
    """Install rotated processors on temporal self-attention layers."""

    # Expected processor counts for this UNet's architecture, used as a
    # sanity assert so a diffusers version bump or a wrong rotate_layers
    # value fails loudly instead of silently rotating the wrong set.
    LAYER_COUNTS = {"up_attn1": 9, "all": 34, "none": 0}

    def apply(self, student: Any, rotate_layers: str = "up_attn1", **kwargs: Any) -> Any:
        """Replace temporal self-attention processors and return the student.

        `rotate_layers` selects which temporal attention layers get rotated,
        identified by class (`TransformerTemporalModel`), not by name
        fragments:
        - "up_attn1": only attn1 (self-attention) in up_blocks (9 layers).
        - "all": attn1 + attn2 in every TransformerTemporalModel, including
          `transformer_in` (34 layers). attn2 here is a second temporal
          self-attention pass, not text cross-attention — ModelScope's
          temp_attentions are built with `double_self_attention=True`, which
          forces attn2's `cross_attention_dim` to None and the UNet always
          calls temp_attentions without `encoder_hidden_states`, so attn2
          runs as self-attention over the same frame-axis sequence as attn1.
        - "none": no rotation (matches Baseline.apply's behavior).
        """
        if rotate_layers not in self.LAYER_COUNTS:
            raise ValueError(f"rotate_layers must be one of {list(self.LAYER_COUNTS)}, got {rotate_layers!r}")

        if rotate_layers == "none":
            return student

        unet = (
            student.get_base_model()
            if hasattr(student, "get_base_model")
            else student.base_model.model
        )
        processors = dict(unet.attn_processors)
        attn_names = ("attn1",) if rotate_layers == "up_attn1" else ("attn1", "attn2")

        replaced = 0
        for name, module in unet.named_modules():
            if not isinstance(module, TransformerTemporalModel):
                continue
            if rotate_layers == "up_attn1" and not name.startswith("up_blocks."):
                continue
            for block_idx in range(len(module.transformer_blocks)):
                for attn_name in attn_names:
                    key = f"{name}.transformer_blocks.{block_idx}.{attn_name}.processor"
                    processors[key] = RotatedTemporalAttnProcessor()
                    replaced += 1

        expected = self.LAYER_COUNTS[rotate_layers]
        if replaced != expected:
            raise ValueError(
                f"Expected {expected} temporal attention layers for "
                f"rotate_layers={rotate_layers!r}, found {replaced}"
            )

        unet.set_attn_processor(processors)
        return student

    def loss(
        self,
        predicted_noise: Any,
        target_noise: Any,
        rotation_weight: float = 0.0,
        **extra: Any,
    ) -> Any:
        """Compute the noise-prediction MSE loss."""
        return torch.nn.functional.mse_loss(predicted_noise.float(), target_noise.float())

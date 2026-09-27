"""Rotate temporal self-attention for reverse-time video generation."""

from __future__ import annotations

from typing import Any, Optional

import torch



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

    def apply(self, student: Any) -> Any:
        """Replace temporal self-attention processors and return the student."""

        unet = (
            student.get_base_model()
            if hasattr(student, "get_base_model")
            else student.base_model.model
        )
        processors = dict(unet.attn_processors)
        replaced = 0
        for name in processors:
            lowered_name = name.lower()
            is_temporal = (
                "temporal" in lowered_name
                or "temp" in lowered_name
            )
            is_temporal_attention = name.endswith("attn1.processor") or name.endswith("attn2.processor")
            if is_temporal and is_temporal_attention:
                processors[name] = RotatedTemporalAttnProcessor()
                replaced += 1

        if replaced == 0:
            raise ValueError("No temporal self-attention processors were found")

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

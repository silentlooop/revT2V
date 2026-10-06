"""Expose the available distillation methods and their shared interface."""

from .attn_rotation import AttentionRotation
from .conv_lora import AttnLoRA, ConvLoRA
from .conv_mirror import ConvMirror

METHODS = {
    "attn_rotation": AttentionRotation,
    "conv_mirror": ConvMirror,
    "conv_lora": ConvLoRA,
    "attn_lora": AttnLoRA,
}

__all__ = ["AttentionRotation", "AttnLoRA", "ConvLoRA", "ConvMirror", "METHODS"]

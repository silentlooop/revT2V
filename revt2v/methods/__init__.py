"""Expose the available distillation methods and their shared interface."""

from .attn_injection import AttnInjection
from .attn_rotation import AttentionRotation
from .conv_lora import AttnLoRA, ConvLoRA

METHODS = {
    "conv_lora": ConvLoRA,
    "attn_lora": AttnLoRA,
    "attn_injection": AttnInjection,
    "attn_rotation": AttentionRotation,
}

__all__ = ["AttentionRotation", "AttnInjection", "AttnLoRA", "ConvLoRA", "METHODS"]

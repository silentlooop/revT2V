"""Expose the available distillation methods and their shared interface."""

from .attn_injection import AttnInjection
from .conv_lora import AttnLoRA, ConvLoRA

METHODS = {
    "conv_lora": ConvLoRA,
    "attn_lora": AttnLoRA,
    "attn_injection": AttnInjection,
}

__all__ = ["AttnInjection", "AttnLoRA", "ConvLoRA", "METHODS"]

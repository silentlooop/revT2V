"""Expose the available distillation methods and their shared interface."""

from .attn_rotation import AttentionRotation
from .conv_mirror import ConvMirror

METHODS = {
    "attn_rotation": AttentionRotation,
    "conv_mirror": ConvMirror,
}

__all__ = ["AttentionRotation", "ConvMirror", "METHODS"]

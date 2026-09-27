"""Expose the available distillation methods and their shared interface."""

from .attn_rotation import AttentionRotation
from .baseline import Baseline
from .motion_prior import MotionPrior

METHODS = {
    "baseline": Baseline,
    "attn_rotation": AttentionRotation,
    "motion_prior": MotionPrior,
}

__all__ = ["AttentionRotation", "Baseline", "MotionPrior", "METHODS"]

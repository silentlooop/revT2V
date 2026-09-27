"""Method 3/3 (secondary): `motion_prior` — reconcile forward and backward
motion priors, from Jeon et al., "Motion Prior Distillation" (ICLR 2026).

*** LEARNING SCAFFOLD — you write the bodies. See LEARNING.md step 6. ***

Plain-language idea: `baseline`'s epsilon loss only ever tells the student
"here is the noise you should have predicted at this one random timestep" —
it never directly says anything about *motion* (how content changes from
frame to frame). The teacher, however, is extremely good at forward motion
(that's what it was pretrained on at scale); its forward-direction motion
statistics are a rich prior we're otherwise throwing away when we only
train on a single flipped copy of its output. This method adds an
auxiliary loss term that explicitly matches the *frame-to-frame motion* of
the student's denoised prediction against the (reversed) teacher target's
motion, on top of the standard epsilon loss — nudging the student to
reproduce not just individual frames but plausible *dynamics*.

Why this should help beyond `baseline`: epsilon loss is evaluated at a
random, usually-high noise level where individual-frame detail is barely
recoverable, so its gradient signal about fine-grained motion is weak. A
motion term computed on the model's estimated clean latents (`x0`, derived
from the predicted noise) gives a more direct, lower-variance signal about
whether consecutive frames move consistently — this is exactly the kind of
"prior" the paper argues is worth distilling explicitly rather than hoping
epsilon loss discovers it implicitly.
"""

from __future__ import annotations

from typing import Any

import torch


class MotionPrior:
    """No architecture change (`apply` is a no-op, like baseline); the
    difference from baseline is entirely in `loss`."""

    def apply(self, student: Any) -> Any:
        """Return `student` unchanged.

        See `Baseline.apply` for why this is still a method rather than
        skipped.
        """
        return student

    def loss(
        self,
        predicted_noise: Any,
        target_noise: Any,
        *,
        predicted_x0: Any = None,
        target_x0: Any = None,
        prior_weight: float = 0.0,
        noisy_latents: Any = None,
        timesteps: Any = None,
        scheduler: Any = None,
        **extra: Any,
    ) -> Any:
        """Compute the epsilon MSE loss plus an optional motion-consistency term."""
        base_loss = torch.nn.functional.mse_loss(predicted_noise.float(), target_noise.float())

        if prior_weight == 0.0 or target_x0 is None:
            return base_loss

        if predicted_x0 is None:
            predicted_x0 = self._predict_x0(predicted_noise, noisy_latents, timesteps, scheduler)

        motion_loss = self.motion_consistency_penalty(predicted_x0, target_x0)
        return base_loss + prior_weight * motion_loss

    def _predict_x0(self, predicted_noise: Any, noisy_latents: Any, timesteps: Any, scheduler: Any) -> Any:
        """Derive the estimated clean latents from a predicted epsilon at
        each sample's timestep: ``x0 = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)``."""
        alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=torch.float32)
        alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_alpha_prod_t = alpha_prod_t.sqrt()
        sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t).sqrt()
        return (noisy_latents.float() - sqrt_one_minus_alpha_prod_t * predicted_noise.float()) / sqrt_alpha_prod_t

    def motion_consistency_penalty(self, predicted_x0: Any, target_x0: Any) -> Any:
        """Penalize mismatched frame-to-frame motion between two latent
        videos, using consecutive-frame differences as a cheap proxy for
        motion (full optical flow is reserved for `scripts/evaluate.py`'s
        eval-time motion-consistency metric — this needs to be fast and
        differentiable for every training step).

        Args:
            predicted_x0, target_x0: Shape ``(B, 4, F, 32, 32)`` each.

        Returns:
            A 0-d scalar tensor.
        """
        predicted_motion = predicted_x0[:, :, 1:] - predicted_x0[:, :, :-1]
        target_motion = target_x0[:, :, 1:] - target_x0[:, :, :-1]
        return torch.nn.functional.mse_loss(predicted_motion.float(), target_motion.float())

"""Method 1/3: `baseline` — naive LoRA fine-tune, no architecture changes.

*** LEARNING SCAFFOLD — you write the bodies. See LEARNING.md step 3. ***

Plain-language idea: don't do anything clever. Take the frozen teacher
U-Net, attach LoRA to its temporal attention layers (`student.build_student`
already does this), and fine-tune on reverse-time latents with standard MSE.
This is the control condition:
`attn_rotation` and `motion_prior` are only interesting if they beat this.

Why would this alone produce backward motion at all? Because the training
*data* (`data.py`'s `flip_latents_time_axis`) already encodes reverse
motion — the model can in principle learn it purely from supervision,
same as any fine-tune. Whether it generalizes as well/fast as the other two
methods is exactly the empirical question this project is comparing.
"""

from __future__ import annotations

from typing import Any

import torch


class Baseline:
    """No-op `apply`; standard epsilon loss."""

    def apply(self, student: Any) -> Any:
        """Return `student` unchanged — baseline has no architecture
        intervention, it's the identity in the method pipeline.

        Kept as a method (not skipped) so `train.py` can call
        `method.apply(student)` uniformly across all three methods without
        special-casing baseline.
        """
        return student

    def loss(self, predicted_noise: Any, target_noise: Any, **extra: Any) -> Any:
        """Compute the plain noise-prediction MSE loss."""
        return torch.nn.functional.mse_loss(predicted_noise.float(), target_noise.float())

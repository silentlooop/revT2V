"""Build a LoRA-wrapped student U-Net for reverse-time video diffusion."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from peft import LoraConfig, get_peft_model
from peft import get_peft_model_state_dict
from peft import set_peft_model_state_dict


def build_student(
    teacher_pipeline,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: Optional[Union[str, List[str]]] = None,
):
    """Wrap the teacher U-Net with trainable LoRA adapters.

    `target_modules` is matched by peft as a module-name *suffix* across the
    whole U-Net, so short names like "to_v" also match spatial and
    cross-attention layers, not just temporal ones. Scope it to temporal
    attention only, e.g. a regex string such as
    ``r".*temp_attentions.*\\.(to_q|to_k|to_v|to_out\\.0)$"``.
    """

    if not target_modules:
        raise ValueError(
            "target_modules must name the temporal attention projections "
            "to receive LoRA, scoped so they don't also match spatial/"
            "cross-attention, e.g. r'.*temp_attentions.*\\.(to_q|to_k|to_v|to_out\\.0)$'"
        )

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    student = get_peft_model(teacher_pipeline.unet, config)

    non_temporal = sorted(
        name
        for name in student.base_model.targeted_module_names
        if "temp" not in name.lower()
    )
    if non_temporal:
        warnings.warn(
            "target_modules matched layers outside temporal attention "
            f"(e.g. {non_temporal[:3]}) — LoRA will train on spatial/"
            "cross-attention too. Scope target_modules to temp_attentions.",
            stacklevel=2,
        )

    student.train()
    return student


def predict_noise(
    student,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Predict noise from noisy latents using the LoRA-wrapped U-Net."""

    predicted_noise = student(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=False,
    )[0]
    return predicted_noise  


def lora_state_dict(student) -> Dict[str, Any]:
    """Return only the trainable LoRA weights from the student."""

    return get_peft_model_state_dict(student)


def load_lora_state_dict(student, state_dict: Dict[str, Any]) -> None:
    """Load saved LoRA weights into the student model."""

    set_peft_model_state_dict(student, state_dict)

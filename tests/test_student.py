"""Test LoRA wiring with a small CPU-only toy U-Net."""

from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from revt2v.student import build_student


class _ToyTemporalAttn(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)


class _ToyUNet(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.spatial_proj = nn.Linear(dim, dim)  # not a LoRA target
        self.temp_attentions = nn.ModuleList([_ToyTemporalAttn(dim)])


def test_build_student_only_lora_params_require_grad():
    pipeline = SimpleNamespace(unet=_ToyUNet())

    student = build_student(
        pipeline,
        lora_rank=2,
        lora_alpha=4,
        target_modules=["to_q", "to_k", "to_v"],
    )

    trainable = [name for name, p in student.named_parameters() if p.requires_grad]
    frozen = [name for name, p in student.named_parameters() if not p.requires_grad]

    assert trainable, "expected some trainable (LoRA) parameters"
    assert all("lora_" in name for name in trainable)
    assert any("spatial_proj" in name for name in frozen)
    assert any("to_q" in name and "lora_" not in name for name in frozen)

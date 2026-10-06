"""Test conv_lora / attn_lora target selection, LoRA wiring, and the mirror loss.

CPU tests use a tiny dummy U-Net built from real `TemporalConvLayer`s and
`TransformerTemporalModel`s; the GPU test checks the real ModelScope counts
(88 Conv3d + 68 Linear).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from diffusers import TransformerTemporalModel
from diffusers.models.resnet import TemporalConvLayer
from peft.tuners.lora import LoraLayer

from revt2v.methods import conv_lora
from revt2v.methods.conv_lora import AttnLoRA, ConvLoRA, temporal_attention_lora_targets
from revt2v.methods.conv_mirror import TEMPORAL_CONV_COUNT, flipped_blocks, mirrored_teacher_noise
from revt2v.student import build_student

CHANNELS = 4
FRAMES = 5


def _temporal_attn() -> TransformerTemporalModel:
    return TransformerTemporalModel(num_attention_heads=1, attention_head_dim=CHANNELS, in_channels=CHANNELS, norm_num_groups=2)


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)])  # spatial, not a target
        self.temp_convs = nn.ModuleList([TemporalConvLayer(CHANNELS, CHANNELS, norm_num_groups=2)])
        self.temp_attentions = nn.ModuleList([_temporal_attn()])


class _ToyUNet(nn.Module):
    """UNet3DConditionModel naming (transformer_in, down/mid/up blocks) and call signature."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.spatial_proj = nn.Linear(CHANNELS, CHANNELS)  # a to_v-free Linear that must stay frozen
        self.transformer_in = _temporal_attn()
        self.down_blocks = nn.ModuleList([_ToyBlock()])
        self.mid_block = _ToyBlock()
        self.up_blocks = nn.ModuleList([_ToyBlock()])
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.normal_(module.weight, std=0.3)
        self.eval()

    def forward(self, x, timesteps=None, encoder_hidden_states=None, return_dict=True):
        batch, channels, frames, height, width = x.shape
        hidden = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        hidden = self.transformer_in(hidden, num_frames=frames, return_dict=False)[0]
        for block in [*self.down_blocks, self.mid_block, *self.up_blocks]:
            hidden = block.temp_convs[0](hidden, num_frames=frames)
            hidden = block.temp_attentions[0](hidden, num_frames=frames, return_dict=False)[0]
        out = hidden.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return (out,)


@pytest.fixture
def toy_counts(monkeypatch):
    # 3 TemporalConvLayers x 4 Conv3d; 4 TransformerTemporalModels x 2 attn x 2 projections.
    monkeypatch.setattr(conv_lora, "TEMPORAL_CONV_COUNT", 12)
    monkeypatch.setattr(conv_lora, "TEMPORAL_ATTN_LINEAR_COUNT", 16)


def _lora_kinds(student) -> dict:
    kinds = {}
    for module in student.modules():
        if isinstance(module, LoraLayer):
            kind = type(module.get_base_layer()).__name__
            kinds[kind] = kinds.get(kind, 0) + 1
    return kinds


def test_attention_targets_are_v_and_out_of_attn1_attn2_incl_transformer_in():
    names = temporal_attention_lora_targets(_ToyUNet())

    assert len(names) == 4 * 2 * 2
    assert any(name.startswith("transformer_in.") for name in names)
    assert {name.rsplit(".attn", 1)[1] for name in names} == {"1.to_v", "1.to_out.0", "2.to_v", "2.to_out.0"}
    assert not any(name.endswith(("to_q", "to_k")) for name in names)


def test_conv_lora_wraps_convs_and_attention_attn_lora_only_attention(toy_counts):
    for method, expected in [(ConvLoRA(), {"Conv3d": 12, "Linear": 16}), (AttnLoRA(), {"Linear": 16})]:
        unet = _ToyUNet().half()
        targets = method.lora_targets(unet)
        student = build_student(SimpleNamespace(unet=unet), lora_rank=2, lora_alpha=2, target_modules=targets)

        assert _lora_kinds(student) == expected
        trainable = {name: p for name, p in student.named_parameters() if p.requires_grad}
        assert trainable and all("lora_" in name for name in trainable)
        assert all(p.dtype == torch.float32 for p in trainable.values())  # fp32 LoRA on an fp16 base
        assert not any("to_q" in name or "to_k" in name or "spatial_proj" in name for name in trainable)


def test_lora_targets_assert_counts():
    with pytest.raises(ValueError, match=str(TEMPORAL_CONV_COUNT)):
        ConvLoRA().lora_targets(_ToyUNet())
    with pytest.raises(ValueError, match="68"):
        AttnLoRA().lora_targets(_ToyUNet())


def test_apply_refuses_flips(toy_counts):
    unet = _ToyUNet()
    student = build_student(SimpleNamespace(unet=unet), lora_rank=2, target_modules=ConvLoRA().lora_targets(unet))
    assert ConvLoRA().apply(student, flip_blocks=[], rotate_layers="none") is student
    with pytest.raises(ValueError):
        ConvLoRA().apply(student, flip_blocks=["down"])


def test_mirror_loss_target_is_flipped_teacher_on_flipped_input(toy_counts):
    unet = _ToyUNet()
    student = build_student(SimpleNamespace(unet=unet), lora_rank=2, target_modules=ConvLoRA().lora_targets(unet))
    # Non-zero LoRA so student != teacher.
    with torch.no_grad():
        for name, p in student.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.3)

    x = torch.randn(1, CHANNELS, FRAMES, 2, 2)
    t = torch.tensor([10])
    emb = torch.randn(1, 3, CHANNELS)
    with torch.no_grad(), student.disable_adapter():
        expected = unet(x.flip(2))[0].flip(2)

    student.train()
    target = mirrored_teacher_noise(student, x, t, emb)
    assert torch.allclose(target, expected, atol=1e-6)
    assert not target.requires_grad
    assert student.training and flipped_blocks(student) == ()  # state restored

    student.eval()  # deterministic: TemporalConvLayer has dropout
    predicted = student(x, t, encoder_hidden_states=emb, return_dict=False)[0]
    noise = torch.randn_like(x)
    loss = ConvLoRA().loss(predicted, noise, mirror_loss_weight=0.5, student=student, noisy_latents=x, timesteps=t, encoder_hidden_states=emb)
    manual = nn.functional.mse_loss(predicted, noise) + 0.5 * nn.functional.mse_loss(predicted, expected)
    assert torch.isfinite(loss) and loss.dtype == torch.float32
    assert torch.allclose(loss, manual, atol=1e-6)

    loss.backward()
    grads = {name for name, p in student.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert grads and all("lora_" in name for name in grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and the ModelScope weights")
def test_modelscope_lora_target_counts():
    from revt2v.teacher import load_teacher

    unet = load_teacher().unet
    targets = ConvLoRA().lora_targets(unet)
    assert len(targets) == 88 + 68
    assert len(AttnLoRA().lora_targets(unet)) == 68

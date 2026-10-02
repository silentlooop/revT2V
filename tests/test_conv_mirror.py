"""Test temporal-conv discovery, flipping, and the double-flip guard.

CPU tests use a tiny dummy U-Net built from real `TemporalConvLayer`s; the
GPU test checks the full ModelScope U-Net is an exact time-mirror when
flipped.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from diffusers.models.resnet import TemporalConvLayer

from revt2v.methods.conv_mirror import (
    ALL_BLOCKS,
    TEMPORAL_CONV_COUNT,
    check_flip_request,
    find_temporal_convs,
    flip_temporal_convs,
    flipped_blocks,
    set_flip_state,
)

CHANNELS = 4
FRAMES = 5


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)])  # spatial, not a target
        self.temp_convs = nn.ModuleList([TemporalConvLayer(CHANNELS, CHANNELS, norm_num_groups=2)])


class _ToyUNet(nn.Module):
    """Same top-level naming as UNet3DConditionModel (down_blocks/mid_block/up_blocks)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.stray_conv3d = nn.Conv3d(CHANNELS, CHANNELS, (3, 1, 1))  # Conv3d outside a TemporalConvLayer
        self.down_blocks = nn.ModuleList([_ToyBlock()])
        self.mid_block = _ToyBlock()
        self.up_blocks = nn.ModuleList([_ToyBlock(), _ToyBlock()])
        # conv4 is zero-initialised (identity layer); randomise so flips matter.
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, F, H, W), like the real U-Net's latents."""
        batch, channels, frames, height, width = x.shape
        hidden = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        for block in [*self.down_blocks, self.mid_block, *self.up_blocks]:
            hidden = block.temp_convs[0](hidden, num_frames=frames)
        return hidden.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def _weights(unet: nn.Module) -> dict:
    modules = dict(unet.named_modules())
    return {name: modules[name].weight.detach().clone() for name in find_temporal_convs(unet)}


def test_find_temporal_convs_selects_only_conv3d_inside_temporal_conv_layers():
    names = find_temporal_convs(_ToyUNet())

    assert len(names) == 4 * 4  # 4 TemporalConvLayers x conv1..conv4
    assert "stray_conv3d" not in names
    assert all(".temp_convs." in name for name in names)
    # Index 2 (conv1) / 3 (conv2-4) of each Sequential: the Conv3d, not GroupNorm/SiLU/Dropout.
    assert {".".join(name.split(".")[-2:]) for name in names} == {
        "conv1.2",
        "conv2.3",
        "conv3.3",
        "conv4.3",
    }


def test_find_temporal_convs_skips_peft_lora_convs():
    from peft import LoraConfig, get_peft_model

    unet = _ToyUNet()
    targets = find_temporal_convs(unet)
    student = get_peft_model(unet, LoraConfig(r=2, lora_alpha=2, target_modules=targets))

    wrapped = find_temporal_convs(student)
    assert len(wrapped) == len(targets)
    assert all(name.endswith(".base_layer") for name in wrapped)
    assert not any("lora_" in name for name in wrapped)


def test_flip_twice_is_identity():
    unet = _ToyUNet()
    original = _weights(unet)

    flip_temporal_convs(unet)
    assert flipped_blocks(unet) == ALL_BLOCKS
    flip_temporal_convs(unet)

    assert flipped_blocks(unet) == ()
    for name, weight in _weights(unet).items():
        assert torch.equal(weight, original[name])


def test_flip_is_exact_and_per_block_no_blending():
    unet = _ToyUNet()
    original = _weights(unet)

    flip_temporal_convs(unet, blocks=("down",))

    for name, weight in _weights(unet).items():
        expected = original[name].flip(2) if name.startswith("down_blocks.") else original[name]
        assert torch.equal(weight, expected), name
    with pytest.raises(ValueError):
        flip_temporal_convs(unet, blocks=("sideways",))


def test_set_flip_state_reaches_exact_target():
    unet = _ToyUNet()
    original = _weights(unet)

    set_flip_state(unet, ("down", "up"))
    set_flip_state(unet, ("mid",))
    assert flipped_blocks(unet) == ("mid",)
    set_flip_state(unet, ())

    for name, weight in _weights(unet).items():
        assert torch.equal(weight, original[name])


def test_flipped_toy_net_is_time_mirror():
    unet = _ToyUNet()
    x = torch.randn(1, CHANNELS, FRAMES, 2, 2)

    with torch.no_grad():
        mirrored = unet(x.flip(2)).flip(2)
        flip_temporal_convs(unet)
        flipped = unet(x)

    assert torch.allclose(flipped, mirrored, atol=1e-5)


def test_double_flip_guard():
    unflipped_student = {"method": "conv_mirror", "flip_blocks": []}
    with pytest.raises(ValueError, match="double-reverse"):
        check_flip_request(ALL_BLOCKS, unflipped_student)

    assert check_flip_request(None, unflipped_student) == ()

    partial_student = {"config": {"flip_blocks": ["down"]}}  # older/config-only metadata
    assert check_flip_request(None, partial_student) == ("down",)
    assert check_flip_request(["down"], partial_student) == ("down",)
    with pytest.raises(ValueError):
        check_flip_request(["down", "up"], partial_student)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and the ModelScope weights")
def test_modelscope_unet_flipped_is_exact_time_mirror():
    from revt2v.teacher import load_teacher

    unet = load_teacher().unet
    names = find_temporal_convs(unet)
    print(f"{len(names)} temporal Conv3d, e.g. {names[:2]} ... {names[-1]}")
    assert len(names) == TEMPORAL_CONV_COUNT

    generator = torch.Generator(device="cuda").manual_seed(0)
    z = torch.randn((1, 4, 16, 32, 32), generator=generator, device="cuda", dtype=unet.dtype)
    t = torch.tensor([500], device="cuda")
    emb = torch.randn((1, 77, 1024), generator=generator, device="cuda", dtype=unet.dtype)

    try:
        with torch.no_grad():
            mirrored = unet(z.flip(2), t, encoder_hidden_states=emb, return_dict=False)[0].flip(2)
            flip_temporal_convs(unet)
            flipped = unet(z, t, encoder_hidden_states=emb, return_dict=False)[0]
    finally:
        set_flip_state(unet, ())

    max_diff = (flipped.float() - mirrored.float()).abs().max().item()
    print(f"max |unet_flipped(z) - flip(unet(flip(z)))| = {max_diff:.3e}")
    assert max_diff < 1e-2  # fp16 rounding only

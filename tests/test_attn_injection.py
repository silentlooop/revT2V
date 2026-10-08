"""Test attention injection: the processor math on a toy Attention (CPU),
and the full apply/prepare_step/loss integration + gradient isolation on a
toy U-Net (CPU, mirrors tests/test_conv_lora.py). The GPU test checks the
real ModelScope counts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from diffusers import TransformerTemporalModel
from diffusers.models.attention_processor import Attention
from diffusers.models.resnet import TemporalConvLayer
from peft.tuners.lora import LoraLayer

from revt2v.methods.attn_injection import (
    AttnInjection,
    _CapturingAttnProcessor,
    _InjectedAttnProcessor,
    inject_layer_targets,
    temporal_self_attn_names,
)
from revt2v.student import build_student

CHANNELS = 4
FRAMES = 5


# --------------------------------------------------------------------------
# Processor-level tests: toy Attention, no U-Net involved.
# --------------------------------------------------------------------------


def _toy_attention(dim: int = 8) -> Attention:
    torch.manual_seed(0)
    return Attention(query_dim=dim, heads=1, dim_head=dim, bias=False)


def _capture(attn: Attention, hidden_states: torch.Tensor) -> dict:
    store: dict = {}
    attn.processor = _CapturingAttnProcessor("layer0", store)
    attn(hidden_states)
    return store


def test_capture_then_inject_without_rotation_reproduces_plain_attention():
    """Point 5's required check: capturing on x and injecting that exact
    (un-rotated) map back against x's own V must be lossless."""
    attn = _toy_attention()
    hidden_states = torch.randn(2, 5, 8)

    store = _capture(attn, hidden_states)
    attn.processor = _InjectedAttnProcessor("layer0", store, rotate=False)
    actual = attn(hidden_states)

    from diffusers.models.attention_processor import AttnProcessor

    attn.processor = AttnProcessor()
    expected = attn(hidden_states)

    assert torch.allclose(actual, expected, atol=1e-4)


def test_capture_on_flipped_input_then_rotated_inject_is_identity_for_an_isolated_layer():
    """`to_q`/`to_k` are per-frame Linears with nothing direction-aware
    upstream of this bare layer, so capturing on flip(x) gives exactly
    flip(Q(x))/flip(K(x)), and rotating 180 degrees on inject exactly undoes
    that flip -- capture+rotate reduces algebraically to PLAIN attention on
    x. This is expected, not a bug: it's the same regime as `transformer_in`
    (the first temporal op in the real U-Net, nothing upstream of it either),
    so injection there is a no-op. The method's real targets
    (rotate_layers="up_attn1") sit downstream of direction-aware temporal
    convs, where this identity does NOT hold -- see the asymmetric-features
    test below, and the GPU test comparing injection at up_attn1 vs
    transformer_in on the real model."""
    attn = _toy_attention()
    hidden_states = torch.randn(1, 6, 8)

    store = _capture(attn, hidden_states.flip(1))
    attn.processor = _InjectedAttnProcessor("layer0", store, rotate=True)
    injected = attn(hidden_states)

    from diffusers.models.attention_processor import AttnProcessor

    attn.processor = AttnProcessor()
    plain = attn(hidden_states)

    assert torch.allclose(injected, plain, atol=1e-3)


def _asymmetric_temporal_mix(x: torch.Tensor) -> torch.Tensor:
    """A tiny direction-aware stand-in for a temporal conv: each frame's
    features depend asymmetrically on the PREVIOUS frame (not symmetric
    under time-reversal), so mix(flip(x)) != flip(mix(x)) -- unlike a
    per-frame Linear, this is not equivariant to flipping, matching what a
    real (unflipped) temporal conv kernel does deeper in the U-Net."""
    shifted = torch.roll(x, shifts=1, dims=1)
    shifted[:, 0] = 0  # no wraparound
    return x + 0.5 * shifted


def test_capture_on_asymmetric_features_then_rotated_inject_differs_from_plain():
    """Once there's a direction-aware op (standing in for the real network's
    temporal convs) between the raw input and this attention layer, the
    identity above no longer holds: capturing on the flipped trajectory's
    *features* (not just the flipped raw input) and injecting rotated 180
    degrees against the real trajectory's own V genuinely differs from plain
    attention. This is the regime attn_injection's real target layers
    (up_attn1, deep in the U-Net) actually operate in."""
    attn = _toy_attention()
    raw = torch.randn(1, 6, 8)
    h = _asymmetric_temporal_mix(raw)
    h_from_flipped_raw = _asymmetric_temporal_mix(raw.flip(1))

    # Sanity: the stand-in really is direction-aware (not just testing a no-op setup).
    assert not torch.allclose(h_from_flipped_raw, h.flip(1), atol=1e-4)

    store = _capture(attn, h_from_flipped_raw)
    attn.processor = _InjectedAttnProcessor("layer0", store, rotate=True)
    injected = attn(h)

    from diffusers.models.attention_processor import AttnProcessor

    attn.processor = AttnProcessor()
    plain = attn(h)

    assert not torch.allclose(injected, plain, atol=1e-3)


def test_inject_raises_on_batch_mismatch():
    attn = _toy_attention()
    store = {"layer0": torch.randn(3, 5, 5)}  # batch 3, but hidden_states below is batch 1
    attn.processor = _InjectedAttnProcessor("layer0", store)
    with pytest.raises(RuntimeError, match="layer0"):
        attn(torch.randn(1, 5, 8))


def test_inject_raises_without_a_capture_pass():
    attn = _toy_attention()
    attn.processor = _InjectedAttnProcessor("layer0", {})
    with pytest.raises(RuntimeError, match="layer0"):
        attn(torch.randn(1, 5, 8))


# --------------------------------------------------------------------------
# U-Net-level tests: toy U-Net with real TransformerTemporalModel layers
# (same construction as tests/test_conv_lora.py's _ToyUNet).
# --------------------------------------------------------------------------


def _temporal_attn() -> TransformerTemporalModel:
    return TransformerTemporalModel(num_attention_heads=1, attention_head_dim=CHANNELS, in_channels=CHANNELS, norm_num_groups=2)


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temp_convs = nn.ModuleList([TemporalConvLayer(CHANNELS, CHANNELS, norm_num_groups=2)])
        self.temp_attentions = nn.ModuleList([_temporal_attn()])


class _AttnProcessorMixin:
    """`attn_processors`/`set_attn_processor`, copied from diffusers'
    UNet3DConditionModel (a plain nn.Module doesn't have these -- they're
    per-class, not provided generically by nn.Module)."""

    @property
    def attn_processors(self) -> dict:
        processors: dict = {}

        def recurse(name, module, acc):
            if hasattr(module, "get_processor"):
                acc[f"{name}.processor"] = module.get_processor()
            for sub_name, child in module.named_children():
                recurse(f"{name}.{sub_name}", child, acc)
            return acc

        for name, module in self.named_children():
            recurse(name, module, processors)
        return processors

    def set_attn_processor(self, processor) -> None:
        def recurse(name, module, proc):
            if hasattr(module, "set_processor"):
                module.set_processor(proc if not isinstance(proc, dict) else proc.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                recurse(f"{name}.{sub_name}", child, proc)

        for name, module in self.named_children():
            recurse(name, module, processor)


class _ToyUNet(_AttnProcessorMixin, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
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
def toy_up_attn1(monkeypatch):
    """This toy UNet has one TransformerTemporalModel in up_blocks (1 block
    x 1 attn1) -- patch the real count (9) down to match it."""
    import revt2v.methods.attn_injection as attn_injection

    monkeypatch.setattr(attn_injection, "_UP_ATTN1_COUNT", 1)


def test_temporal_self_attn_names_up_attn1(toy_up_attn1):
    names = temporal_self_attn_names(_ToyUNet(), "up_attn1")
    assert names == ["up_blocks.0.temp_attentions.0.transformer_blocks.0.attn1"]


def test_inject_layer_targets_are_v_and_out_only(toy_up_attn1):
    targets = inject_layer_targets(_ToyUNet(), "up_attn1")
    assert targets == [
        "up_blocks.0.temp_attentions.0.transformer_blocks.0.attn1.to_v",
        "up_blocks.0.temp_attentions.0.transformer_blocks.0.attn1.to_out.0",
    ]


def test_apply_refuses_flip_blocks(toy_up_attn1):
    unet = _ToyUNet()
    student = build_student(SimpleNamespace(unet=unet), lora_rank=2, target_modules=inject_layer_targets(unet))
    with pytest.raises(ValueError):
        AttnInjection().apply(student, flip_blocks=["down"])


def test_prepare_step_and_loss_gradients_land_only_on_lora_params(toy_up_attn1):
    """Point 10's required check: one training step's gradients populate
    only the LoRA params this method targets, never Q/K or base weights."""
    unet = _ToyUNet()
    method = AttnInjection()
    student = build_student(SimpleNamespace(unet=unet), lora_rank=2, target_modules=inject_layer_targets(unet))
    student = method.apply(student, rotate_layers="up_attn1")
    # Non-zero LoRA so predictions actually depend on it.
    with torch.no_grad():
        for name, p in student.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.3)

    x = torch.randn(1, CHANNELS, FRAMES, 2, 2)
    t = torch.tensor([10])
    emb = torch.randn(1, 3, CHANNELS)
    noise = torch.randn_like(x)

    student.train()
    method.prepare_step(student, x, t, emb)
    predicted = student(x, t, encoder_hidden_states=emb, return_dict=False)[0]
    loss = method.loss(predicted, noise, mirror_loss_weight=0.5)

    assert torch.isfinite(loss) and loss.dtype == torch.float32
    loss.backward()

    grads = {name for name, p in student.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert grads and all("lora_" in name for name in grads)
    assert not any("to_q" in name or "to_k" in name for name in grads)

    kinds = {type(m.get_base_layer()).__name__ for m in student.modules() if isinstance(m, LoraLayer)}
    assert kinds == {"Linear"}  # only to_v/to_out.0 Linears got LoRA, never Conv3d/Q/K


def test_prepare_step_restores_training_mode_and_leaves_inject_processors_active(toy_up_attn1):
    unet = _ToyUNet()
    student = build_student(SimpleNamespace(unet=unet), lora_rank=2, target_modules=inject_layer_targets(unet))
    method = AttnInjection()
    student = method.apply(student, rotate_layers="up_attn1")

    x = torch.randn(1, CHANNELS, FRAMES, 2, 2)
    t = torch.tensor([10])
    emb = torch.randn(1, 3, CHANNELS)

    student.train()
    method.prepare_step(student, x, t, emb)
    assert student.training  # restored, not left in .eval() from the capture pass

    # The just-installed processors must be inject (not capture) processors,
    # i.e. a normal forward call must succeed and consume method._store.
    out = student(x, t, encoder_hidden_states=emb, return_dict=False)[0]
    assert out.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and the ModelScope weights")
def test_modelscope_layer_counts():
    from revt2v.teacher import load_teacher

    unet = load_teacher().unet
    assert len(temporal_self_attn_names(unet, "up_attn1")) == 9
    assert len(temporal_self_attn_names(unet, "all")) == 34
    both = temporal_self_attn_names(unet, "up_both")
    assert len(both) % 2 == 0 and len(both) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and the ModelScope weights")
def test_injection_differs_from_plain_at_up_attn1_but_not_at_transformer_in():
    """Confirms the mechanism actually does something at its real target
    layers (up_attn1, downstream of many direction-aware temporal convs),
    while reproducing the isolated-layer identity at transformer_in (the
    first temporal op in the U-Net, nothing direction-aware upstream of it
    -- same regime as the CPU toy test above)."""
    from revt2v.teacher import load_teacher

    unet = load_teacher().unet
    z = torch.randn(1, 4, 16, 32, 32, device="cuda", dtype=unet.dtype)
    t = torch.tensor([500], device="cuda")
    emb = torch.randn(1, 77, 1024, device="cuda", dtype=unet.dtype)

    with torch.no_grad():
        plain = unet(z, t, encoder_hidden_states=emb, return_dict=False)[0]

    def injected_output(layer_names: list) -> torch.Tensor:
        store: dict = {}
        stock = dict(unet.attn_processors)
        capture_processors = {f"{n}.processor": _CapturingAttnProcessor(n, store) for n in layer_names}
        inject_processors = {f"{n}.processor": _InjectedAttnProcessor(n, store) for n in layer_names}
        try:
            with torch.no_grad():
                unet.set_attn_processor({**stock, **capture_processors})
                unet(z.flip(2), t, encoder_hidden_states=emb, return_dict=False)
                unet.set_attn_processor({**stock, **inject_processors})
                out = unet(z, t, encoder_hidden_states=emb, return_dict=False)[0]
        finally:
            unet.set_attn_processor(stock)
        return out

    up_attn1_out = injected_output(temporal_self_attn_names(unet, "up_attn1"))
    transformer_in_out = injected_output(["transformer_in.transformer_blocks.0.attn1"])

    up_attn1_diff = (up_attn1_out.float() - plain.float()).abs().max().item()
    transformer_in_diff = (transformer_in_out.float() - plain.float()).abs().max().item()
    print(f"max diff vs plain: up_attn1={up_attn1_diff:.4f} transformer_in={transformer_in_diff:.6f}")

    assert transformer_in_diff < 1e-2  # fp16 tolerance: isolated-layer identity holds
    assert up_attn1_diff > 1e-2  # deep layer: injection genuinely changes the output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and the ModelScope weights")
def test_two_pass_training_step_on_real_model():
    from revt2v.data import flip_latents_time_axis
    from revt2v.teacher import load_teacher

    pipeline = load_teacher()
    unet = pipeline.unet
    method = AttnInjection()
    student = build_student(SimpleNamespace(unet=unet), lora_rank=4, target_modules=inject_layer_targets(unet))
    student = method.apply(student, rotate_layers="up_attn1")

    x = torch.randn(1, 4, 16, 32, 32, device="cuda", dtype=unet.dtype)
    t = torch.tensor([500], device="cuda")
    emb = torch.randn(1, 77, 1024, device="cuda", dtype=unet.dtype)
    noise = torch.randn_like(x)

    student.train()
    method.prepare_step(student, x, t, emb)
    predicted = student(x, t, encoder_hidden_states=emb, return_dict=False)[0]
    loss = method.loss(predicted, noise, mirror_loss_weight=0.5)
    loss.backward()

    grads = {name for name, p in student.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert grads and all("lora_" in name for name in grads)

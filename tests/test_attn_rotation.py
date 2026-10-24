"""Test temporal attention rotation on a small CPU-only attention layer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers.models.attention_processor import Attention

from revt2v.methods.attn_rotation import RotatedTemporalAttnProcessor


def _reference_attention(attn: Attention, hidden_states: torch.Tensor, rotate: bool) -> torch.Tensor:
    """Independent from-scratch reference: standard scaled-dot-product
    self-attention using `attn`'s own weights, optionally with q/k flipped
    along the frame axis (dim=1) and v/output left untouched."""
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    if rotate:
        query = torch.flip(query, dims=[1])
        key = torch.flip(key, dims=[1])

    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, value)
    return attn.to_out[0](out)


def _toy_attention(dim: int = 8) -> Attention:
    torch.manual_seed(0)
    attn = Attention(query_dim=dim, heads=1, dim_head=dim, bias=False)
    attn.processor = RotatedTemporalAttnProcessor()
    return attn


def test_rotated_attention_matches_flip_q_k_reference():
    attn = _toy_attention()
    hidden_states = torch.randn(2, 5, 8)  # (B*H*W, F, C)

    actual = attn(hidden_states)
    expected = _reference_attention(attn, hidden_states, rotate=True)

    assert actual.shape == hidden_states.shape
    assert torch.allclose(actual, expected, atol=1e-4)


def test_rotated_attention_is_not_plain_attention():
    """Guards against the classic mistake: flipping the *whole input*
    before attention and flipping the *whole output* back afterward is
    mathematically the identity (ordinary self-attention is
    permutation-equivariant), which would silently reduce to plain,
    unrotated attention instead of actually rotating anything."""
    attn = _toy_attention()
    hidden_states = torch.randn(1, 6, 8)

    rotated = attn(hidden_states)
    plain = _reference_attention(attn, hidden_states, rotate=False)

    assert not torch.allclose(rotated, plain, atol=1e-3)


def test_rotated_attention_output_position_zero_uses_reversed_query():
    """The output at frame 0 should be driven by the query for the *last*
    frame (that's what "flip q/k, don't flip the output" means) — spot
    check a single position against manual single-row attention. The
    weights for output position 0 come from query[F-1] against the
    *original* key (an ordinary attention row), but they get applied to
    the *reversed* value sequence — that reversal is what makes this
    genuinely different from ordinary attention rather than just relabeling
    which query row is "row 0"."""
    attn = _toy_attention()
    hidden_states = torch.randn(1, 4, 8)

    full = attn(hidden_states)

    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    last_frame_query = query[:, -1:, :]  # query for frame F-1
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(last_frame_query, key.transpose(-1, -2)) * scale
    weights = torch.softmax(scores, dim=-1)
    reversed_value = torch.flip(value, dims=[1])
    expected_row = attn.to_out[0](torch.matmul(weights, reversed_value))

    assert torch.allclose(full[:, 0:1, :], expected_row, atol=1e-4)


# --------------------------------------------------------------------------
# revt2v.infer wiring (CPU-only, no model download): layer selection and
# the zero-shot path's adapter-disable guarantee.
# --------------------------------------------------------------------------

import torch.nn as nn
from diffusers import TransformerTemporalModel

from revt2v.infer import _rotation_processors
from revt2v.student import build_student

_CHANNELS = 4


def _temporal_attn() -> TransformerTemporalModel:
    return TransformerTemporalModel(num_attention_heads=1, attention_head_dim=_CHANNELS, in_channels=_CHANNELS, norm_num_groups=2)


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temp_attentions = nn.ModuleList([_temporal_attn()])


class _AttnProcessorMixin:
    """Same `attn_processors`/`set_attn_processor` shim as
    tests/test_attn_injection.py's _ToyUNet -- a plain nn.Module doesn't
    have these (they're per-class on diffusers' real UNet3DConditionModel)."""

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
    """One TransformerTemporalModel in up_blocks, one in down_blocks (the
    down one must NOT be selected by rotate_layers="up_attn1")."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.down_blocks = nn.ModuleList([_ToyBlock()])
        self.up_blocks = nn.ModuleList([_ToyBlock()])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.3)
        self.eval()

    def forward(self, x, timesteps=None, encoder_hidden_states=None, return_dict=True):
        batch, channels, frames, height, width = x.shape
        hidden = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        hidden = self.down_blocks[0].temp_attentions[0](hidden, num_frames=frames, return_dict=False)[0]
        hidden = self.up_blocks[0].temp_attentions[0](hidden, num_frames=frames, return_dict=False)[0]
        out = hidden.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return (out,)


def test_rotation_processors_selects_only_up_block_attn1(monkeypatch):
    import revt2v.infer as infer

    monkeypatch.setattr(infer, "_UP_ATTN1_ROTATION_COUNT", 1)  # toy U-Net has 1 up-block attn1, not 9
    processors = infer._rotation_processors(_ToyUNet())
    assert list(processors) == ["up_blocks.0.temp_attentions.0.transformer_blocks.0.attn1.processor"]
    assert isinstance(processors["up_blocks.0.temp_attentions.0.transformer_blocks.0.attn1.processor"], RotatedTemporalAttnProcessor)


def test_rotation_processors_wrong_count_raises(monkeypatch):
    import revt2v.infer as infer

    monkeypatch.setattr(infer, "_UP_ATTN1_ROTATION_COUNT", 9)  # toy U-Net only has 1, not the real model's 9
    with pytest.raises(RuntimeError, match="found 1"):
        infer._rotation_processors(_ToyUNet())


def test_attn_rotation_zeroshot_disables_every_adapter():
    """The exact code path MethodBank.generate uses for
    "attn_rotation_zeroshot" (`student.disable_adapter()`, no adapter set
    active first) must leave the student computing EXACTLY the plain base
    model's output -- that's what "zero-shot" is supposed to guarantee, not
    just an assumption that disable_adapter() worked."""
    unet = _ToyUNet()
    targets = [
        "up_blocks.0.temp_attentions.0.transformer_blocks.0.attn1.to_v",
        "up_blocks.0.temp_attentions.0.transformer_blocks.0.attn1.to_out.0",
    ]
    student = build_student(SimpleNamespace(unet=unet), lora_rank=2, lora_alpha=2, target_modules=targets, adapter_name="attn_rotation")
    for name, param in student.named_parameters():
        if "lora_B" in name:  # peft zero-inits lora_B so a fresh adapter is a no-op; give it a real effect to test against
            nn.init.normal_(param, std=0.5)
    student.eval()  # build_student() leaves it in train mode; temporal attention has dropout, which must not add noise here

    x = torch.randn(1, 4, 3, 4, 4)
    with torch.no_grad():
        # unet and student share the same underlying LoraLayer-wrapped
        # modules -- "disabled" is a global flag on them, not something
        # that depends on which top-level reference forward() is called
        # through, so plain_out must be computed in the same context.
        with student.disable_adapter():
            disabled_out = student(x, return_dict=False)[0]
            plain_out = unet(x, return_dict=False)[0]
        enabled_out = student(x, return_dict=False)[0]

    assert torch.allclose(disabled_out, plain_out, atol=1e-5)
    assert not torch.allclose(enabled_out, plain_out, atol=1e-4)  # LoRA must actually do something when enabled

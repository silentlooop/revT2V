"""Test temporal attention rotation on a small CPU-only attention layer."""

from __future__ import annotations

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

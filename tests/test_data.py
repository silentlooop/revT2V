"""CPU-only, no model download: `flip_latents_time_axis` and `LatentDataset`."""

from __future__ import annotations

import torch
import pytest

from revt2v.data import LatentDataset, flip_latents_time_axis


def test_flip_latents_time_axis_unbatched_shape_and_values():
    # (C, F, H, W) — unbatched record convention, C=4.
    latents = torch.randn(4, 5, 3, 3)

    flipped = flip_latents_time_axis(latents)

    assert flipped.shape == latents.shape
    assert torch.equal(flipped[:, 0], latents[:, -1])
    assert torch.equal(flipped[:, -1], latents[:, 0])
    assert torch.equal(flipped, torch.flip(latents, dims=[1]))


def test_flip_latents_time_axis_batched_shape_and_values():
    # (B, C, F, H, W) — the DataLoader-batched convention.
    latents = torch.randn(2, 4, 5, 3, 3)

    flipped = flip_latents_time_axis(latents)

    assert flipped.shape == latents.shape
    assert torch.equal(flipped, torch.flip(latents, dims=[2]))


def test_flip_latents_time_axis_is_an_involution():
    latents = torch.randn(4, 6, 3, 3)
    assert torch.equal(flip_latents_time_axis(flip_latents_time_axis(latents)), latents)


def _make_record(num_frames: int = 6) -> dict:
    return {
        "prompt": "a test prompt",
        "latents": torch.randn(4, num_frames, 8, 8, dtype=torch.float16),
        "reversed_latents": torch.randn(4, num_frames, 8, 8, dtype=torch.float16),
        "prompt_embeds": torch.randn(77, 1024, dtype=torch.float16),
        "negative_prompt_embeds": torch.randn(77, 1024, dtype=torch.float16),
    }


def test_latent_dataset_item_shapes(tmp_path):
    torch.save(_make_record(), tmp_path / "00000_aaaaaaaa.pt")
    torch.save(_make_record(), tmp_path / "00001_bbbbbbbb.pt")

    dataset = LatentDataset(tmp_path)
    assert len(dataset) == 2

    item = dataset[0]
    assert item["latents"].shape == (4, 6, 8, 8)
    assert item["reversed_latents"].shape == (4, 6, 8, 8)
    assert item["prompt_embeds"].shape == (77, 1024)
    assert item["negative_prompt_embeds"].shape == (77, 1024)
    assert item["latents"].dtype == torch.float16
    assert item["prompt"] == "a test prompt"


def test_latent_dataset_raises_on_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        LatentDataset(tmp_path)

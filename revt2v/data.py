"""Build and load reverse-time latent dataset records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import Dataset
from . import teacher


def flip_latents_time_axis(latents: torch.Tensor) -> torch.Tensor:
    """Reverse a latent video along its frame axis."""

    reversed_latents = torch.flip(latents, dims=[-3])
    
    return reversed_latents


def build_dataset_record(
    prompt: str,
    teacher_pipeline,
    num_frames: int = 16,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 50,
    guidance_scale: float = 9.0,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Generate and package one training record for a prompt."""
    
    forward_latents = teacher.generate_forward_video(
        prompt=prompt,
        pipeline=teacher_pipeline,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed
    )
    
    reversed_latents = flip_latents_time_axis(forward_latents["latents"])
    
    prompt_embeds, negative_prompt_embeds = teacher.encode_prompt(prompt=prompt, pipeline=teacher_pipeline)

    # Dataset records are stored without the leading batch dimension:
    # (1, C, F, H, W) -> (C, F, H, W), and prompt embeddings likewise.
    return {
        "prompt": prompt,
        "latents": forward_latents["latents"].squeeze(0).cpu(),
        "reversed_latents": reversed_latents.squeeze(0).cpu(),
        "prompt_embeds": prompt_embeds.squeeze(0).cpu(),
        "negative_prompt_embeds": negative_prompt_embeds.squeeze(0).cpu()
    }



class LatentDataset(Dataset):
    """Load saved latent dataset records."""

    def __init__(self, root: str | Path) -> None:
        """Initialize the dataset from a directory of saved records."""
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.pt"))
        
        if not self.files:
            raise FileNotFoundError(
                f"No .pt files found in {self.root}"
            )

    def __len__(self) -> int:
        """Return the number of saved records."""
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load and return one saved record."""

        return torch.load(self.files[index], map_location="cpu")
        

"""Build and load reverse-time latent dataset records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    negative_prompt: str | None = None,
    return_video: bool = False,
) -> Dict[str, Any]:
    """Generate and package one training record for a prompt.

    `negative_prompt` only steers generation; the stored
    `negative_prompt_embeds` stay the empty-prompt (unconditional) embedding
    that CFG dropout trains on. `return_video=True` adds the decoded
    (F, H, W, C) clip under "video" (pop it before saving).
    """

    forward_latents = teacher.generate_forward_video(
        prompt=prompt,
        pipeline=teacher_pipeline,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        negative_prompt=negative_prompt,
    )
    
    reversed_latents = flip_latents_time_axis(forward_latents["latents"])
    
    prompt_embeds, negative_prompt_embeds = teacher.encode_prompt(prompt=prompt, pipeline=teacher_pipeline)

    # Dataset records are stored without the leading batch dimension:
    # (1, C, F, H, W) -> (C, F, H, W), and prompt embeddings likewise.
    record = {
        "prompt": prompt,
        "latents": forward_latents["latents"].squeeze(0).cpu(),
        "reversed_latents": reversed_latents.squeeze(0).cpu(),
        "prompt_embeds": prompt_embeds.squeeze(0).cpu(),
        "negative_prompt_embeds": negative_prompt_embeds.squeeze(0).cpu()
    }
    if return_video:
        record["video"] = forward_latents["video"]
    return record


MANIFEST_NAME = "manifest.jsonl"


def read_manifest(root: str | Path) -> List[Dict[str, Any]]:
    """Entries of `<root>/manifest.jsonl` (one JSON object per line), or []
    if there is none. Written by scripts/build_dataset.py; each entry has
    "shard", "prompt", "seed", "motion_score" and "kept"."""
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]



class LatentDataset(Dataset):
    """Load saved latent dataset records."""

    def __init__(self, root: str | Path, min_motion_score: Optional[float] = None) -> None:
        """Initialize the dataset from a directory of saved records.

        With a manifest, only its kept shards with motion_score >=
        `min_motion_score` (if set) are used; without one, every *.pt is.
        """
        self.root = Path(root)
        manifest = read_manifest(self.root)
        if manifest:
            entries = [e for e in manifest if e.get("kept", True)]
            if min_motion_score is not None:
                entries = [e for e in entries if e.get("motion_score", 0.0) >= min_motion_score]
            self.files = sorted(self.root / e["shard"] for e in entries if (self.root / e["shard"]).exists())
        else:
            if min_motion_score is not None:
                raise ValueError(f"min_motion_score needs a {MANIFEST_NAME} in {self.root}")
            self.files = sorted(self.root.glob("*.pt"))
        
        if not self.files:
            raise FileNotFoundError(
                f"No .pt files found in {self.root}"
                + (f" with motion_score >= {min_motion_score}" if min_motion_score is not None else "")
            )

    def __len__(self) -> int:
        """Return the number of saved records."""
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load and return one saved record."""

        return torch.load(self.files[index], map_location="cpu")
        

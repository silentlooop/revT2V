"""Load and run the frozen ModelScope text-to-video teacher pipeline."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from diffusers import DiffusionPipeline

MODEL_ID = "ali-vilab/text-to-video-ms-1.7b"


def load_teacher(
    model_id: str = MODEL_ID,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load and freeze the ModelScope text-to-video pipeline."""
    
    pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")
    pipeline.to(device)
        
    # freezing the model
    for module in (pipeline.unet, pipeline.vae, pipeline.text_encoder):
        module.requires_grad_(False)
        module.eval()

    return pipeline


def encode_prompt(
    pipeline,
    prompt: str,
    negative_prompt: str = "",
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode positive and negative prompts into text embeddings."""
    
    tokens = pipeline.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    tokens_negative = pipeline.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        text_embeddings = pipeline.text_encoder(
            tokens.input_ids.to(device)
        )[0]

        text_embeddings_negative = pipeline.text_encoder(
            tokens_negative.input_ids.to(device)
        )[0]
        
    
    return text_embeddings,text_embeddings_negative 
    


def generate_forward_video(
    pipeline,
    prompt: str,
    num_frames: int = 16,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 50,
    guidance_scale: float = 9.0,
    seed: Optional[int] = None,
) -> dict:
    """Generate a forward-time video and return its latents and decoded frames."""
       
    generator = None
    if seed is not None :
        generator = torch.Generator(device = pipeline.device).manual_seed(seed)
    
    with torch.no_grad():
        result = pipeline(
            prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
        )
    
    # With output_type="latent", TextToVideoSDPipeline returns `frames`
    # already in the U-Net's native (B, C, F, H/8, W/8) = (B, 4, F, 32, 32)
    # layout (verified against diffusers' pipeline_text_to_video_synth.py) —
    # no permute needed.
    latents = result.frames

    video = decode_latents_to_video(pipeline, latents)

    return {
        "latents": latents,
        "video": video,
    }
    


def decode_latents_to_video(
    pipeline,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Decode latent video tensors into video frames in the [0, 1] range."""

    batch_size, latent_channels, num_frames, latent_height, latent_width = (
        latents.shape
    )

    # Convert (B, C, F, H, W) to (B*F, C, H, W) for the image VAE.
    # Move frames next to the batch, then flatten B and F for the image VAE:
    # (B, C, F, H, W) -> (B*F, C, H, W).
    vae_latents = latents.permute(
        0, 2, 1, 3, 4
    ).reshape(
        batch_size * num_frames,
        latent_channels,
        latent_height,
        latent_width,
    )

    # Undo the scaling applied to latents before they enter the U-Net.
    vae_latents = vae_latents / pipeline.vae.config.scaling_factor

    with torch.no_grad():
        decoded = pipeline.vae.decode(vae_latents).sample

    # Convert VAE output from [-1, 1] to [0, 1].
    decoded = (decoded / 2 + 0.5).clamp(0, 1)

    _, video_channels, frame_height, frame_width = decoded.shape

    # Convert (B*F, C, H, W) to (B, F, H, W, C).
    # Restore the frame axis and move channels last for video consumers:
    # (B*F, C, H, W) -> (B, F, H, W, C).
    video = decoded.reshape(
        batch_size,
        num_frames,
        video_channels,
        frame_height,
        frame_width,
    ).permute(0, 1, 3, 4, 2)

    # Return (F, H, W, C) for a single video.
    if batch_size == 1:
        return video[0]

    return video


def encode_video_to_latents(
    pipeline,
    video: torch.Tensor,
) -> torch.Tensor:
    """Encode pixel-space video frames in [0, 1] into latent video tensors.

    Inverse of `decode_latents_to_video`: accepts the same ``(F, H, W, C)``
    or ``(B, F, H, W, C)`` layout and range it returns, and produces latents
    in the U-Net's ``(B, C, F, H, W)`` layout.
    """
    if video.ndim == 4:
        video = video.unsqueeze(0)

    batch_size, num_frames, frame_height, frame_width, video_channels = video.shape

    # (B, F, H, W, C) -> (B*F, C, H, W) for the image VAE.
    pixels = video.permute(0, 1, 4, 2, 3).reshape(
        batch_size * num_frames,
        video_channels,
        frame_height,
        frame_width,
    )

    # Convert [0, 1] to [-1, 1], the VAE's expected input range.
    pixels = pixels.to(pipeline.vae.dtype) * 2 - 1

    with torch.no_grad():
        latents = pipeline.vae.encode(pixels).latent_dist.sample()

    # Apply the same scaling used before latents enter the U-Net.
    latents = latents * pipeline.vae.config.scaling_factor

    _, latent_channels, latent_height, latent_width = latents.shape

    # (B*F, C, H, W) -> (B, C, F, H, W), the U-Net's latent layout.
    latents = latents.reshape(
        batch_size,
        num_frames,
        latent_channels,
        latent_height,
        latent_width,
    ).permute(0, 2, 1, 3, 4)

    return latents

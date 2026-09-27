"""Train LoRA weights for a student U-Net on reverse-time latent data."""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from revt2v import data, teacher  # noqa: E402
from revt2v import student as Student  # noqa: E402
from revt2v.methods import METHODS  # noqa: E402
from revt2v.utils import (  # noqa: E402
    CheckpointManager,
    add_config_args,
    default_checkpoint_dir,
    default_data_root,
    load_config,
    seed_everything,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--method", type=str, default=None, choices=list(METHODS), help="Overrides config's 'method'")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--hf-repo", type=str, default=None, help="Private HF Hub model repo id for checkpoint backup/resume")
    parser.add_argument("--push-every-seconds", type=float, default=600.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    method_name = args.method or config.get("method", "baseline")
    seed_everything(config.get("seed", 42))

    data_root = args.data_root or Path(config.get("data_root", default_data_root()))
    checkpoint_dir = args.checkpoint_dir or Path(config.get("checkpoint_dir", default_checkpoint_dir())) / method_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Method=%s device=%s data_root=%s", method_name, device, data_root)

    # --- Model setup (each call below is a scaffolded function you implement) ---
    pipeline = teacher.load_teacher(device=device)
    # pipeline.scheduler is whatever fast sampler the pipeline ships for
    # inference (e.g. DPM-Solver); its `add_noise` indexes into a short
    # `set_timesteps()`-populated schedule, not the full training range.
    # Training noise uses a DDPM scheduler built from the same config
    # instead, since alphas_cumprod only depends on the beta schedule, not
    # which scheduler class computes it.
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    student = Student.build_student(
        pipeline,
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.0),
        target_modules=config.get("target_modules"),
    )
    method = METHODS[method_name]()
    student = method.apply(student)

    optimizer = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=config.get("learning_rate", 1e-4),
    )

    # --- Resume ---
    ckpt = CheckpointManager(checkpoint_dir, repo_id=args.hf_repo, push_every_seconds=args.push_every_seconds)
    start_step = 0
    payload = ckpt.resume()
    if payload is not None:
        Student.load_lora_state_dict(student, payload["model"])
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        start_step = payload.get("step", 0) + 1
        logger.info("Resumed from step %d", payload.get("step", 0))
    else:
        logger.info("No checkpoint found, starting from scratch")

    # --- Data ---
    dataset = data.LatentDataset(data_root)
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 1), shuffle=True)
    batches = itertools.cycle(loader)

    num_steps = config.get("num_steps", 1000)
    progress = tqdm(range(start_step, num_steps), initial=start_step, total=num_steps, desc=f"train[{method_name}]")

    for step in progress:
        batch = next(batches)
        reversed_latents = batch["reversed_latents"].to(device=device, dtype=pipeline.unet.dtype)
        prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=pipeline.unet.dtype)

        noise = torch.randn(
            reversed_latents.shape,
            device=reversed_latents.device,
            dtype=reversed_latents.dtype,
        )
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (reversed_latents.shape[0],),
            device=reversed_latents.device,
            dtype=torch.long,
        )
        noisy_latents = noise_scheduler.add_noise(
            reversed_latents,
            noise,
            timesteps,
        )
        predicted_noise = Student.predict_noise(student, noisy_latents, timesteps, prompt_embeds)
        # Extra kwargs beyond (predicted_noise, target_noise) are method-specific
        # (e.g. rotation_weight, prior_weight) — every method accepts **extra so
        # this call is uniform across baseline/attn_rotation/motion_prior. A
        # method whose loss needs more context (e.g. motion_prior deriving
        # predicted_x0) can read noisy_latents/timesteps/scheduler from here too.
        loss = method.loss(
            predicted_noise,
            noise,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            scheduler=noise_scheduler,
            target_x0=reversed_latents,
            rotation_weight=config.get("rotation_weight", 0.0),
            prior_weight=config.get("prior_weight", 0.0),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            progress.set_postfix(loss=float(loss.detach().cpu()))

        if step % args.save_every == 0 or step == num_steps - 1:
            ckpt.save(
                step,
                Student.lora_state_dict(student),
                optimizer.state_dict(),
                force_push=(step == num_steps - 1),
                method=method_name,
            )

    logger.info("Training complete: %d steps, checkpoint at %s", num_steps, ckpt.local_path)


if __name__ == "__main__":
    main()

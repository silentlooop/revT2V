"""Train LoRA weights for a student U-Net on reverse-time latent data."""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from revt2v import data, infer, teacher  # noqa: E402
from revt2v import student as Student  # noqa: E402
from revt2v.methods import METHODS  # noqa: E402
from revt2v.utils import (  # noqa: E402
    CheckpointManager,
    add_config_args,
    default_checkpoint_dir,
    default_data_root,
    get_hf_token,
    list_hub_files,
    load_config,
    push_file_to_hub,
    save_video,
    seed_everything,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


def _ensure_dataset(data_root: Path, hf_dataset_repo: str | None, data_version: str | None = None) -> None:
    """Pull the shards (and manifest) of `data_version` from `hf_dataset_repo`
    if `data_root` has none locally.

    `data_version` (e.g. "v2") is the Hub folder `scripts/build_dataset.py`
    wrote to; None means the original shards at the repo root (files in
    version folders are never mixed in). For the case where a Kaggle session
    was wiped and `build_dataset.py` wasn't re-run before `train.py`.
    """
    if data_root.exists() and any(data_root.glob("*.pt")):
        return
    if not hf_dataset_repo:
        return

    from huggingface_hub import hf_hub_download

    prefix = f"{data_version.strip('/')}/" if data_version else ""
    wanted = [
        f
        for f in list_hub_files(hf_dataset_repo, repo_type="dataset")
        if f.startswith(prefix)
        and "/" not in f[len(prefix):]
        and (f.endswith(".pt") or f.endswith(data.MANIFEST_NAME))
    ]
    data_root.mkdir(parents=True, exist_ok=True)
    logger.info("data_root is empty; pulling %d files from %s/%s", len(wanted), hf_dataset_repo, prefix or "<root>")
    for name in tqdm(wanted, desc="pulling dataset"):
        downloaded = hf_hub_download(repo_id=hf_dataset_repo, filename=name, repo_type="dataset")
        (data_root / name[len(prefix):]).write_bytes(Path(downloaded).read_bytes())


def _log_lora_layers(student) -> None:
    """Log how many Conv3d/Linear layers got LoRA and the trainable param count."""
    from peft.tuners.lora import LoraLayer

    kinds = {}
    for module in student.modules():
        if isinstance(module, LoraLayer):
            kind = type(module.get_base_layer()).__name__
            kinds[kind] = kinds.get(kind, 0) + 1
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    dtypes = sorted({str(p.dtype) for p in student.parameters() if p.requires_grad})
    logger.info(
        "LoRA-wrapped layers: Conv3d=%d Linear=%d (all: %s) | trainable params=%d (%s)",
        kinds.get("Conv3d", 0),
        kinds.get("Linear", 0),
        kinds,
        trainable,
        ", ".join(dtypes),
    )


def _enable_gradient_checkpointing(unet) -> None:
    try:
        unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled on the U-Net")
    except ValueError as exc:
        logger.warning(
            "gradient_checkpointing=true but %s does not support it in this "
            "diffusers version (%s) — continuing without it",
            type(unet).__name__,
            exc,
        )


def _run_sampling(pipeline, student, config, step, sample_folder, hf_repo, results_dir, device) -> None:
    """Generate config['sample_prompts'] with the CURRENT student (rotation +
    LoRA active, CFG with uncond) and upload them, then restore train mode."""
    prompts = config.get("sample_prompts") or []
    if not prompts:
        return

    student.eval()
    try:
        for i, prompt in enumerate(prompts):
            with torch.no_grad():
                result = infer.generate(
                    prompt,
                    pipeline,
                    student,
                    num_frames=config.get("num_frames", 16),
                    height=config.get("height", 256),
                    width=config.get("width", 256),
                    num_inference_steps=config.get("sample_inference_steps", 25),
                    guidance_scale=config.get("guidance_scale", 9.0),
                    seed=config.get("sample_seed", 0),
                    device=device,
                )
            video_path = results_dir / sample_folder / "samples" / f"step{step:06d}_prompt{i}.mp4"
            save_video(result["video"], video_path)
            if hf_repo:
                push_file_to_hub(
                    video_path,
                    hf_repo,
                    f"samples/{sample_folder}/step{step:06d}_prompt{i}.mp4",
                    token=get_hf_token(),
                )
    finally:
        student.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--method", type=str, default=None, choices=list(METHODS), help="Overrides config's 'method'")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--hf-repo", type=str, default=None, help="Private HF Hub model repo id for checkpoint/sample backup/resume")
    parser.add_argument("--hf-dataset-repo", type=str, default=None, help="Private HF Hub dataset repo id to pull shards from if data_root is empty")
    parser.add_argument("--push-every-seconds", type=float, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Overrides config's num_steps, for smoke tests")
    parser.add_argument("--hf-subfolder", type=str, default=None, help="Overrides config's hf_subfolder (e.g. smoke_test/conv_lora)")
    parser.add_argument("--fresh", action="store_true", help="Ignore any existing checkpoint and start from step 0")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    method_name = args.method or config.get("method", "attn_rotation")
    seed_everything(config.get("seed", 42))

    data_root = args.data_root or Path(config.get("data_root", default_data_root()))
    checkpoint_dir = args.checkpoint_dir or Path(config.get("checkpoint_dir", default_checkpoint_dir())) / method_name
    results_dir = Path(config.get("results_dir", "results"))

    hf_repo = args.hf_repo or config.get("hf_repo")
    hf_dataset_repo = args.hf_dataset_repo or config.get("hf_dataset_repo")
    # None -> checkpoint pushed to the HF repo root (original layout); set
    # per method so different methods never overwrite each other's latest.pt.
    hf_subfolder = args.hf_subfolder or config.get("hf_subfolder")
    if method_name != "attn_rotation" and not (hf_subfolder or "").strip("/"):
        raise ValueError(f"{method_name} must set hf_subfolder so it never overwrites the root (attn_rotation) checkpoint")
    if method_name == "conv_mirror":
        if not config.get("use_trained_weights", True):
            raise ValueError("use_trained_weights=false is the conv oracle (flipped teacher); it needs no training")
    push_every_seconds = args.push_every_seconds if args.push_every_seconds is not None else config.get("push_every_seconds", 600.0)
    log_every_steps = args.log_every_steps if args.log_every_steps is not None else config.get("log_every_steps", 20)
    save_every_steps = args.save_every_steps if args.save_every_steps is not None else config.get("save_every_steps", 200)
    sample_every_steps = config.get("sample_every_steps", 250)
    grad_accum_steps = config.get("grad_accum_steps", 4)
    uncond_prob = config.get("uncond_prob", 0.1)
    num_steps = args.max_steps if args.max_steps is not None else config.get("num_steps", 1000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Method=%s device=%s data_root=%s", method_name, device, data_root)

    _ensure_dataset(data_root, hf_dataset_repo, config.get("data_version"))

    # --- Model setup (each call below is a scaffolded function you implement) ---
    pipeline = teacher.load_teacher(device=device)
    if config.get("gradient_checkpointing", False):
        _enable_gradient_checkpointing(pipeline.unet)
    # pipeline.scheduler is whatever fast sampler the pipeline ships for
    # inference (e.g. DPM-Solver); its `add_noise` indexes into a short
    # `set_timesteps()`-populated schedule, not the full training range.
    # Training noise uses a DDPM scheduler built from the same config
    # instead, since alphas_cumprod only depends on the beta schedule, not
    # which scheduler class computes it.
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    method = METHODS[method_name]()
    if hasattr(method, "lora_targets") and not config.get("target_modules"):
        # Saved into the checkpoint's config so inference rebuilds the same adapter.
        config["target_modules"] = method.lora_targets(pipeline.unet)
    student = Student.build_student(
        pipeline,
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.0),
        target_modules=config.get("target_modules"),
    )
    _log_lora_layers(student)
    student = method.apply(
        student,
        rotate_layers=config.get("rotate_layers", "none"),
        flip_blocks=config.get("flip_blocks") or [],
    )

    optimizer = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=config.get("learning_rate", 1e-4),
    )

    # --- Resume ---
    ckpt = CheckpointManager(checkpoint_dir, repo_id=hf_repo, push_every_seconds=push_every_seconds, hub_subfolder=hf_subfolder)
    start_step = 0
    payload = None if args.fresh else ckpt.resume()
    if payload is not None:
        Student.load_lora_state_dict(student, payload["model"])
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        rng_state = payload.get("metadata", {}).get("rng_state")
        if rng_state is not None:
            try:
                torch.set_rng_state(rng_state)
            except Exception:
                logger.warning("Could not restore RNG state from checkpoint; continuing with the fresh seed")
        start_step = payload.get("step", 0) + 1
        logger.info("Resumed from step %d", payload.get("step", 0))
    else:
        logger.info("No checkpoint found (or --fresh passed), starting from scratch")

    # --- Data ---
    dataset = data.LatentDataset(data_root, min_motion_score=config.get("min_motion_score"))
    logger.info("Dataset: %d clips from %s (min_motion_score=%s)", len(dataset), data_root, config.get("min_motion_score"))
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 1), shuffle=True)
    batches = itertools.cycle(loader)

    progress = tqdm(range(start_step, num_steps), initial=start_step, total=num_steps, desc=f"train[{method_name}]")

    for step in progress:
        step_start = time.time()
        optimizer.zero_grad()
        accumulated_loss = 0.0
        skipped_micro_batches = 0

        for _ in range(grad_accum_steps):
            batch = next(batches)
            reversed_latents = batch["reversed_latents"].to(device=device, dtype=pipeline.unet.dtype)
            prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=pipeline.unet.dtype)

            if uncond_prob > 0:
                negative_prompt_embeds = batch["negative_prompt_embeds"].to(device=device, dtype=pipeline.unet.dtype)
                uncond_mask = (torch.rand(prompt_embeds.shape[0], device=device) < uncond_prob).view(-1, 1, 1)
                prompt_embeds = torch.where(uncond_mask, negative_prompt_embeds, prompt_embeds)

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
            # this call is uniform across attn_rotation/conv_mirror. A method
            # whose loss needs more context (e.g. conv_mirror's mirrored-teacher
            # term) can read noisy_latents/timesteps/student/etc. from here too.
            loss = method.loss(
                predicted_noise,
                noise,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                scheduler=noise_scheduler,
                target_x0=reversed_latents,
                rotation_weight=config.get("rotation_weight", 0.0),
                prior_weight=config.get("prior_weight", 0.0),
                mirror_loss_weight=config.get("mirror_loss_weight", 0.0),
                student=student,
                encoder_hidden_states=prompt_embeds,
            )

            if not torch.isfinite(loss):
                logger.warning("Non-finite loss at step %d, skipping micro-batch", step)
                skipped_micro_batches += 1
                continue

            (loss / grad_accum_steps).backward()
            accumulated_loss += loss.detach().item()

        if skipped_micro_batches < grad_accum_steps:
            optimizer.step()

        if step % log_every_steps == 0:
            denom = grad_accum_steps - skipped_micro_batches
            avg_loss = accumulated_loss / denom if denom > 0 else float("nan")
            step_time = time.time() - step_start
            progress.set_postfix(loss=avg_loss, step_time=f"{step_time:.2f}s")
            logger.info("step=%d loss=%.4f step_time=%.2fs", step, avg_loss, step_time)

        if step % save_every_steps == 0 or step == num_steps - 1:
            ckpt.save(
                step,
                Student.lora_state_dict(student),
                optimizer.state_dict(),
                force_push=(step == num_steps - 1),
                method=method_name,
                config=config,
                flip_blocks=list(config.get("flip_blocks") or []),
                rng_state=torch.get_rng_state(),
            )

        if sample_every_steps and (step % sample_every_steps == 0 or step == num_steps - 1):
            # Samples go under hf_subfolder too, so a smoke test never mixes into a real run's.
            _run_sampling(pipeline, student, config, step, hf_subfolder or method_name, hf_repo, results_dir, device)

    logger.info("Training complete: %d steps, checkpoint at %s", num_steps, ckpt.local_path)


if __name__ == "__main__":
    main()

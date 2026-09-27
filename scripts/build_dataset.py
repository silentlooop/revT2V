"""Build a resumable reverse-time latent dataset from training prompts."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from revt2v import data, teacher  # noqa: E402
from revt2v.utils import (  # noqa: E402
    add_config_args,
    default_data_root,
    list_hub_files,
    load_config,
    push_file_to_hub,
    seed_everything,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_dataset")


def shard_name(index: int, prompt: str) -> str:
    """Deterministic filename for a prompt's record: an index (for stable
    ordering/inspection) plus a short hash of the prompt text (so re-running
    with a reordered/edited prompts file doesn't silently collide two
    different prompts onto the same shard)."""
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{index:05d}_{digest}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--prompts", type=Path, default=Path("prompts/train.txt"))
    parser.add_argument("--output", type=Path, default=None, help="Defaults to the config's data_root")
    parser.add_argument("--hf-repo", type=str, default=None, help="Private HF Hub dataset repo id for shard backup/resume")
    parser.add_argument("--limit", type=int, default=None, help="Only build the first N prompts (debugging)")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    seed_everything(config.get("seed", 42))

    output_dir = args.output or Path(config.get("data_root", default_data_root()))
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    if args.limit:
        prompts = prompts[: args.limit]
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompts)

    hub_files = set(list_hub_files(args.hf_repo, repo_type="dataset")) if args.hf_repo else set()
    if hub_files:
        logger.info("Found %d shards already on the Hub", len(hub_files))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading teacher on %s ...", device)
    pipeline = teacher.load_teacher(device=device)

    built, skipped = 0, 0
    for index, prompt in enumerate(tqdm(prompts, desc="building dataset")):
        name = shard_name(index, prompt)
        local_path = output_dir / name

        if local_path.exists():
            skipped += 1
            continue

        if name in hub_files:
            # Already built in a previous session; pull it down instead of
            # burning GPU time regenerating it.
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(repo_id=args.hf_repo, filename=name, repo_type="dataset")
            local_path.write_bytes(Path(downloaded).read_bytes())
            skipped += 1
            continue

        record = data.build_dataset_record(
            prompt,
            pipeline,
            num_frames=config.get("num_frames", 16),
            height=config.get("height", 256),
            width=config.get("width", 256),
            num_inference_steps=config.get("num_inference_steps", 50),
            guidance_scale=config.get("guidance_scale", 9.0),
            seed=config.get("seed", 42) + index,
        )
        torch.save(record, local_path)
        built += 1

        if args.hf_repo:
            push_file_to_hub(local_path, args.hf_repo, name, repo_type="dataset")

    logger.info("Done. Built %d new shards, skipped %d already present. Output: %s", built, skipped, output_dir)


if __name__ == "__main__":
    main()

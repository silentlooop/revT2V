"""Build a resumable reverse-time latent dataset from training prompts.

Each prompt is generated with `--seeds-per-prompt` seeds. Every clip gets a
`motion_score` (mean Farneback optical-flow magnitude on a 128px grayscale
preview of the decoded clip); clips below `--min-motion` are not saved. Every
clip, kept or not, is recorded in `manifest.jsonl`, so a resumed run never
regenerates it. With `--data-version v2`, shards and manifest live under
`v2/` in the HF dataset repo, leaving older data at the repo root intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from revt2v import data, teacher  # noqa: E402
from revt2v.utils import (  # noqa: E402
    add_config_args,
    default_data_root,
    list_hub_files,
    load_config,
    push_files_to_hub,
    seed_everything,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_dataset")

MOTION_PREVIEW_SIZE = 128


def shard_name(index: int, prompt: str, seed_index: int = 0) -> str:
    """Deterministic filename for a (prompt, seed) record: an index (for stable
    ordering/inspection), the seed slot, plus a short hash of the prompt text
    (so re-running with a reordered/edited prompts file doesn't silently
    collide two different prompts onto the same shard)."""
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{index:05d}_s{seed_index}_{digest}.pt"


def read_prompts(path: Path) -> list:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def check_no_overlap(train_prompts: list, test_prompts: list) -> None:
    overlap = {p.lower() for p in train_prompts} & {p.lower() for p in test_prompts}
    assert not overlap, f"train/test prompt overlap: {sorted(overlap)}"


def motion_score(video) -> float:
    """Mean Farneback optical-flow magnitude (pixels at 128x128) over all
    consecutive frame pairs of an (F, H, W, C) clip in [0, 1]."""
    import cv2

    frames = (video.float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    gray = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (MOTION_PREVIEW_SIZE, MOTION_PREVIEW_SIZE), interpolation=cv2.INTER_AREA)
        for f in frames
    ]
    magnitudes = []
    for prev, nxt in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitudes.append(np.linalg.norm(flow, axis=-1).mean())
    return float(np.mean(magnitudes))


def log_motion_distribution(entries: list, min_motion: float) -> None:
    scores = np.array([e["motion_score"] for e in entries])
    if not len(scores):
        return
    percentiles = np.percentile(scores, [0, 10, 25, 50, 75, 90, 100])
    logger.info(
        "motion_score over %d clips: min=%.3f p10=%.3f p25=%.3f median=%.3f p75=%.3f p90=%.3f max=%.3f",
        len(scores),
        *percentiles,
    )
    logger.info("below --min-motion %.3f (skipped): %d", min_motion, int((scores < min_motion).sum()))
    counts, edges = np.histogram(scores, bins=10)
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        logger.info("  [%.2f, %.2f) %4d %s", lo, hi, count, "#" * int(40 * count / max(counts.max(), 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--prompts", type=Path, default=Path("prompts/train.txt"))
    parser.add_argument("--test-prompts", type=Path, default=Path("prompts/test.txt"), help="Asserted disjoint from --prompts")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to the config's data_root")
    parser.add_argument("--hf-repo", type=str, default=None, help="Private HF Hub dataset repo id for shard backup/resume")
    parser.add_argument("--data-version", type=str, default=None, help="Hub folder for this build, e.g. v2 (default: config's data_version; none = repo root)")
    parser.add_argument("--seeds-per-prompt", type=int, default=3)
    parser.add_argument("--min-motion", type=float, default=0.15, help="Skip clips whose motion_score is below this")
    parser.add_argument("--negative-prompt", type=str, default="watermark, text")
    parser.add_argument("--push-every", type=int, default=10, help="Clips per Hub commit (shards + manifest together)")
    parser.add_argument("--limit", type=int, default=None, help="Only build the first N prompts (debugging)")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    seed_everything(config.get("seed", 42))

    output_dir = args.output or Path(config.get("data_root", default_data_root()))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_version = args.data_version if args.data_version is not None else config.get("data_version")
    hub_prefix = f"{data_version.strip('/')}/" if data_version else ""

    prompts = read_prompts(args.prompts)
    if args.test_prompts.exists():
        check_no_overlap(prompts, read_prompts(args.test_prompts))
    if args.limit:
        prompts = prompts[: args.limit]
    logger.info("Loaded %d prompts from %s x %d seeds", len(prompts), args.prompts, args.seeds_per_prompt)

    hub_files = set(list_hub_files(args.hf_repo, repo_type="dataset")) if args.hf_repo else set()
    manifest_path = output_dir / data.MANIFEST_NAME
    hub_manifest = f"{hub_prefix}{data.MANIFEST_NAME}"

    # Resume: local manifest, else the Hub's. Shards themselves aren't pulled
    # here; scripts/train.py pulls them when it needs them.
    if not manifest_path.exists() and hub_manifest in hub_files:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=args.hf_repo, filename=hub_manifest, repo_type="dataset")
        manifest_path.write_bytes(Path(downloaded).read_bytes())
    entries = {e["shard"]: e for e in data.read_manifest(output_dir)}
    logger.info("Manifest has %d clips already (%s/%s on the Hub)", len(entries), args.hf_repo, hub_prefix or "<root>")

    def write_manifest() -> None:
        manifest_path.write_text("".join(json.dumps(e) + "\n" for e in entries.values()))

    # Kept shards from an earlier run of this session that never reached the Hub.
    pending = [
        name
        for name, e in entries.items()
        if e["kept"] and f"{hub_prefix}{name}" not in hub_files and (output_dir / name).exists()
    ]

    def flush() -> None:
        if not args.hf_repo:
            pending.clear()
            return
        files = [(output_dir / name, f"{hub_prefix}{name}") for name in pending]
        push_files_to_hub(
            files + [(manifest_path, hub_manifest)],
            args.hf_repo,
            repo_type="dataset",
            commit_message=f"build_dataset: {len(entries)} clips in manifest",
        )
        pending.clear()

    jobs = [(i, p, k) for i, p in enumerate(prompts) for k in range(args.seeds_per_prompt)]
    todo = [job for job in jobs if shard_name(job[0], job[1], job[2]) not in entries]
    logger.info("%d clips to build, %d already in the manifest", len(todo), len(jobs) - len(todo))

    pipeline = None
    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading teacher on %s ...", device)
        pipeline = teacher.load_teacher(device=device)
        pipeline.set_progress_bar_config(disable=True)

    built, skipped_motion, since_push = 0, 0, 0
    for index, prompt, seed_index in tqdm(todo, desc="building dataset"):
        name = shard_name(index, prompt, seed_index)
        seed = config.get("seed", 42) + index * args.seeds_per_prompt + seed_index
        record = data.build_dataset_record(
            prompt,
            pipeline,
            num_frames=config.get("num_frames", 16),
            height=config.get("height", 256),
            width=config.get("width", 256),
            num_inference_steps=config.get("num_inference_steps", 50),
            guidance_scale=config.get("guidance_scale", 9.0),
            seed=seed,
            negative_prompt=args.negative_prompt,
            return_video=True,
        )
        score = motion_score(record.pop("video"))
        kept = score >= args.min_motion
        record["motion_score"] = score
        if kept:
            torch.save(record, output_dir / name)
            pending.append(name)
            built += 1
        else:
            skipped_motion += 1
            logger.info("skip (motion %.3f < %.3f): %s [seed %d]", score, args.min_motion, prompt, seed)
        entries[name] = {
            "shard": name,
            "prompt": prompt,
            "prompt_index": index,
            "seed": seed,
            "motion_score": score,
            "kept": kept,
            "negative_prompt": args.negative_prompt,
        }
        write_manifest()

        since_push += 1
        if since_push >= args.push_every:
            flush()
            since_push = 0

    write_manifest()
    if pending or since_push:
        flush()

    log_motion_distribution(list(entries.values()), args.min_motion)
    kept_total = sum(e["kept"] for e in entries.values())
    logger.info(
        "Done. Built %d new shards, skipped %d for low motion. Manifest: %d clips, %d kept. Output: %s",
        built,
        skipped_motion,
        len(entries),
        kept_total,
        output_dir,
    )


if __name__ == "__main__":
    main()

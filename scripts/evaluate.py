"""Evaluate a trained method against the reversed teacher on held-out
prompts: FVD, CLIP text-video score, temporal consistency, motion
consistency, plus side-by-side comparison videos.

Usage:
    python scripts/evaluate.py --config configs/attn_rotation.yaml \\
        --prompts prompts/test.txt --hf-repo <username>/revt2v-checkpoints

The CLI/orchestration below (arg parsing, loading models, looping over
prompts, saving results) is fully implemented. The four `compute_*` metric
functions are a learning scaffold — see LEARNING.md step 4.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from revt2v import data, infer, teacher  # noqa: E402
from revt2v.utils import (  # noqa: E402
    add_config_args,
    default_checkpoint_dir,
    default_results_dir,
    load_config,
    save_comparison_video,
    seed_everything,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")


# --------------------------------------------------------------------------
# Metric functions — LEARNING SCAFFOLD, you write the bodies.
# --------------------------------------------------------------------------


def compute_fvd(real_videos: "np.ndarray", fake_videos: "np.ndarray") -> float:
    """Frechet Video Distance between two sets of videos: how close the
    distribution of generated (student) videos is to the distribution of
    reference (reversed teacher) videos, in a pretrained video-classifier
    feature space.

    Args:
        real_videos: ``(N, F, H, W, C)`` uint8 array — reversed-teacher
            reference videos on the held-out prompts.
        fake_videos: ``(N, F, H, W, C)`` uint8 array — student outputs on
            the same prompts, same N and F.

    Returns:
        A single float distance (lower = more similar distributions).

    # TODO(1): Load a pretrained video feature extractor (classically an
    #   I3D network trained on Kinetics). Check what's easiest to obtain in
    #   your Kaggle/Colab environment — you may need an extra package or a
    #   downloaded checkpoint; this is the most infrastructure-heavy metric
    #   here, budget time for it.
    # TODO(2): Extract features for every video in both sets (batch it —
    #   N held-out videos at once may not fit in memory).
    # TODO(3): Fit a Gaussian (mean, covariance) to each feature set, then
    #   compute the Frechet distance between the two Gaussians (same formula
    #   as FID, just on video features instead of image features).
    #
    # Hint: with only ~30 held-out prompts, FVD's covariance estimate will
    # be noisy — treat this number as directional, not authoritative, and
    # say so when you report it.
    """
    raise NotImplementedError


def compute_clip_score(video: "np.ndarray", prompt: str) -> float:
    """Average CLIP image-text similarity between each frame and the
    prompt: does the generated video actually depict the prompt?

    Args:
        video: ``(F, H, W, C)`` uint8 array.
        prompt: The text prompt used to generate `video`.

    Returns:
        Mean cosine similarity across frames, roughly in ``[0, 1]``
        (CLIP similarities are usually reported *100 or left as cosine sim
        — pick one and document it).

    # TODO(1): Load a pretrained CLIP model + processor (e.g. via
    #   `transformers.CLIPModel`/`CLIPProcessor`, already a project
    #   dependency).
    # TODO(2): For each frame, get the image embedding and the prompt's text
    #   embedding, cosine-similarity them.
    # TODO(3): Average over frames.
    #
    # Hint: load the CLIP model once outside any per-prompt loop (in
    # `main()`, not inside this function) and pass it in — reloading a CLIP
    # checkpoint per prompt would dominate your eval runtime. Consider
    # adding a `model=None` parameter here that `main()` populates.
    """
    raise NotImplementedError


def compute_temporal_consistency(video: "np.ndarray") -> float:
    """How smoothly content changes frame-to-frame: mean CLIP image-image
    cosine similarity between consecutive frames. High for a
    smooth/coherent video, low for a flickery/incoherent one.

    Args:
        video: ``(F, H, W, C)`` uint8 array.

    Returns:
        Mean cosine similarity between consecutive-frame CLIP embeddings.

    # TODO(1): Embed every frame with a CLIP image encoder (reuse the model
    #   from `compute_clip_score` — consider a shared `model=None` param).
    # TODO(2): Cosine-similarity each `embedding[i]` with `embedding[i+1]`.
    # TODO(3): Average over the F-1 consecutive pairs.
    """
    raise NotImplementedError


def compute_motion_consistency(student_video: "np.ndarray", reference_video: "np.ndarray") -> float:
    """How similar the *motion* (not just per-frame content) is between the
    student's output and the reversed-teacher reference, via optical flow.

    Args:
        student_video: ``(F, H, W, C)`` uint8 array.
        reference_video: ``(F, H, W, C)`` uint8 array, same F.

    Returns:
        A single float (pick a direction — e.g. mean endpoint error between
        flow fields, lower = better — and document it).

    # TODO(1): Compute per-frame-pair optical flow for both videos (e.g.
    #   `cv2.calcOpticalFlowFarneback` if OpenCV is available in your
    #   environment, or another flow method of your choice).
    # TODO(2): Compare the two videos' flow fields frame-pair by frame-pair
    #   (e.g. mean absolute/squared difference between flow vectors).
    # TODO(3): Average over all F-1 frame pairs.
    #
    # Hint: OpenCV isn't in this project's dependencies — if you use it,
    # note that in README.md's "versions used" section so future-you (or a
    # grader) knows to install it.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# Orchestration — fully implemented.
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--prompts", type=Path, default=Path("prompts/test.txt"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    method_name = args.method or config.get("method", "baseline")
    seed_everything(config.get("seed", 42))

    checkpoint_dir = args.checkpoint_dir or Path(config.get("checkpoint_dir", default_checkpoint_dir())) / method_name
    output_dir = args.output or Path(default_results_dir()) / method_name
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    if args.limit:
        prompts = prompts[: args.limit]
    logger.info("Evaluating method=%s on %d held-out prompts", method_name, len(prompts))

    teacher_pipeline, student = infer.load_student_for_inference(
        method_name,
        checkpoint_dir=str(checkpoint_dir),
        hf_repo_id=args.hf_repo,
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=config.get("target_modules"),
    )

    per_prompt: list[dict[str, Any]] = []
    student_videos, reference_videos = [], []

    for i, prompt in enumerate(prompts):
        forward = teacher.generate_forward_video(teacher_pipeline, prompt, seed=config.get("seed", 42) + i)
        reference_latents = data.flip_latents_time_axis(forward["latents"])
        reference_video = teacher.decode_latents_to_video(teacher_pipeline, reference_latents)

        result = infer.generate(prompt, teacher_pipeline, student, seed=config.get("seed", 42) + i)
        student_video = result["video"]

        save_comparison_video(student_video, reference_video, output_dir / f"{i:03d}_compare.mp4")
        student_videos.append(student_video)
        reference_videos.append(reference_video)

        per_prompt.append(
            {
                "prompt": prompt,
                "clip_score": compute_clip_score(student_video, prompt),
                "temporal_consistency": compute_temporal_consistency(student_video),
                "motion_consistency": compute_motion_consistency(student_video, reference_video),
            }
        )
        logger.info("[%d/%d] %s", i + 1, len(prompts), prompt)

    # Combine the per-prompt videos into (N, F, H, W, C) metric batches.
    fvd = compute_fvd(np.stack(reference_videos), np.stack(student_videos))

    summary = {
        "method": method_name,
        "num_prompts": len(prompts),
        "fvd": fvd,
        "mean_clip_score": float(np.mean([p["clip_score"] for p in per_prompt])),
        "mean_temporal_consistency": float(np.mean([p["temporal_consistency"] for p in per_prompt])),
        "mean_motion_consistency": float(np.mean([p["motion_consistency"] for p in per_prompt])),
        "per_prompt": per_prompt,
    }
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Wrote %s", output_dir / "metrics.json")


if __name__ == "__main__":
    main()

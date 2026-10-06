"""How much LoRA rank does the time-flip of each temporal Conv3d need?

For a temporal conv with weight W (out, in, 3, 1, 1), flipping time changes
tap 0 by D = W[:, :, 2] - W[:, :, 0], tap 1 by 0 and tap 2 by -D. peft's
Conv3d LoRA is lora_B (1x1x1, out x r) after lora_A (same 3x1x1 kernel,
r x in), i.e. a per-tap update B @ A_k with a shared B, so rank r can express
the flip exactly when r >= rank(D). This prints the singular-value energy of
D captured by the top 8/16/32/64 components per layer, a summary, and a
recommended `lora_rank`. Needs only the U-Net weights; CPU is fine.

    python scripts/conv_rank_check.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from revt2v.methods.conv_mirror import find_temporal_convs  # noqa: E402
from revt2v.teacher import MODEL_ID  # noqa: E402

RANKS = (8, 16, 32, 64)


def flip_delta_spectrum(weight: torch.Tensor) -> dict:
    """Singular values of D = W[:, :, 2] - W[:, :, 0] for one (out, in, 3, 1, 1) kernel."""
    w = weight.detach().float().reshape(weight.shape[0], weight.shape[1], weight.shape[2])
    delta = w[:, :, 2] - w[:, :, 0]
    s = torch.linalg.svdvals(delta).double().numpy()
    energy = np.cumsum(s**2) / max((s**2).sum(), 1e-30)
    return {
        "shape": tuple(delta.shape),
        "rel_norm": float(delta.norm() / max(w.norm(), 1e-30)),
        "delta_sq": float((s**2).sum()),
        "energy": {r: float(energy[min(r, len(s)) - 1]) for r in RANKS},
        "rank90": int(np.searchsorted(energy, 0.90) + 1),
        "rank99": int(np.searchsorted(energy, 0.99) + 1),
    }


def rank_report(unet) -> dict:
    """Per-layer spectra plus the summary used for the recommendation."""
    modules = dict(unet.named_modules())
    layers = {name: flip_delta_spectrum(modules[name].weight) for name in find_temporal_convs(unet)}
    total = sum(l["delta_sq"] for l in layers.values()) or 1e-30
    summary = {}
    for r in RANKS:
        per_layer = np.array([l["energy"][r] for l in layers.values()])
        summary[r] = {
            "median": float(np.median(per_layer)),
            "min": float(per_layer.min()),
            # Energy over all layers together, weighted by how big each D is.
            "weighted": float(sum(l["energy"][r] * l["delta_sq"] for l in layers.values()) / total),
        }
    return {"layers": layers, "summary": summary}


def recommend_rank(summary: dict, target: float = 0.9) -> int:
    """Smallest rank whose median AND norm-weighted energy reach `target`."""
    for r in RANKS:
        if summary[r]["median"] >= target and summary[r]["weighted"] >= target:
            return r
    return RANKS[-1]


def print_report(report: dict, target: float) -> int:
    header = f"{'layer':<58} {'D shape':>11} {'|D|/|W|':>8} " + " ".join(f"{'top' + str(r):>6}" for r in RANKS) + f" {'r90':>5} {'r99':>5}"
    print(header)
    print("-" * len(header))
    for name, l in report["layers"].items():
        short = name.replace(".temp_convs.", ".tc.")
        print(
            f"{short:<58} {str(l['shape']):>11} {l['rel_norm']:8.3f} "
            + " ".join(f"{l['energy'][r]:6.3f}" for r in RANKS)
            + f" {l['rank90']:5d} {l['rank99']:5d}"
        )

    layers = list(report["layers"].values())
    print(f"\n{len(layers)} temporal Conv3d layers")
    print(f"rank for 90% energy: median {np.median([l['rank90'] for l in layers]):.0f}, max {max(l['rank90'] for l in layers)}")
    print(f"rank for 99% energy: median {np.median([l['rank99'] for l in layers]):.0f}, max {max(l['rank99'] for l in layers)}")
    print(f"\n{'rank':>5} {'median':>8} {'min':>8} {'weighted':>9}   (fraction of |D|^2 captured)")
    for r, s in report["summary"].items():
        print(f"{r:>5} {s['median']:8.3f} {s['min']:8.3f} {s['weighted']:9.3f}")

    rank = recommend_rank(report["summary"], target)
    print(
        f"\nRecommended lora_rank: {rank} (smallest of {list(RANKS)} with median and "
        f"norm-weighted energy >= {target:.0%}; "
        + ("reached" if min(report["summary"][rank]["median"], report["summary"][rank]["weighted"]) >= target
           else f"NOT reached even at {RANKS[-1]}")
        + ")"
    )
    print(f"Set it in configs/conv_lora.yaml and configs/attn_lora.yaml, or pass --set lora_rank={rank}")
    return rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--target", type=float, default=0.9, help="Energy fraction the recommended rank must capture")
    args = parser.parse_args()

    from diffusers import UNet3DConditionModel

    unet = UNet3DConditionModel.from_pretrained(args.model_id, subfolder="unet", variant="fp16", torch_dtype=torch.float16)
    print_report(rank_report(unet), args.target)


if __name__ == "__main__":
    main()

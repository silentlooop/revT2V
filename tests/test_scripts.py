"""CPU-only checks for the prompt files, the dataset motion filter, and the
conv rank check (scripts/ isn't a package, so the files are loaded directly)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from diffusers.models.resnet import TemporalConvLayer

ROOT = Path(__file__).resolve().parent.parent

TEST_PROMPTS = [
    "white smoke rising from a burning incense stick, black background",
    "red ink dropping into a clear glass of water, close-up",
    "orange juice being poured into a glass, close-up",
    "a red ball falling onto a wooden floor, static camera",
    "a man walking away from the camera down a long corridor, static camera",
]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"revt2v_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_files():
    build = _load_script("build_dataset")
    train = build.read_prompts(ROOT / "prompts" / "train.txt")
    test = build.read_prompts(ROOT / "prompts" / "test.txt")

    assert test == TEST_PROMPTS
    assert 140 <= len(train) <= 160 and len(set(train)) == len(train)
    build.check_no_overlap(train, test)


def test_motion_score_static_vs_moving():
    build = _load_script("build_dataset")
    frames, size = 8, 64
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    moving = torch.zeros(frames, size, size, 3)
    for f in range(frames):
        blob = ((yy - 32) ** 2 + (xx - 12 - 5 * f) ** 2 < 64).float()
        moving[f] = blob[..., None]
    static = moving[:1].expand(frames, -1, -1, -1)

    assert build.motion_score(static) < 0.01
    assert build.motion_score(moving) > 0.1


def test_rank_check_finds_the_rank_of_the_flip_delta():
    rank_check = _load_script("conv_rank_check")
    torch.manual_seed(0)
    layer = TemporalConvLayer(64, 64, norm_num_groups=8)
    conv = layer.conv1[2]
    with torch.no_grad():
        side = torch.randn(64, 64)
        delta = torch.randn(64, 4) @ torch.randn(4, 64)  # rank 4
        conv.weight[:, :, 0, 0, 0] = side
        conv.weight[:, :, 2, 0, 0] = side + delta

    spectrum = rank_check.flip_delta_spectrum(conv.weight)
    assert spectrum["rank99"] <= 4
    assert spectrum["energy"][8] > 0.999

    unet = nn.Module()
    unet.down_blocks = nn.ModuleList([nn.Module()])
    unet.down_blocks[0].temp_convs = nn.ModuleList([layer])
    report = rank_check.rank_report(unet)
    assert len(report["layers"]) == 4
    assert rank_check.recommend_rank({r: {"median": 0.95, "weighted": 0.95} for r in rank_check.RANKS}) == 8
    assert rank_check.recommend_rank({r: {"median": 0.5, "weighted": 0.5} for r in rank_check.RANKS}) == 64

"""Test evaluate.py's metric math on synthetic data (CPU-only, no downloads)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "revt2v_evaluate", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


def _moving_square(frames: int = 8, size: int = 64, step: int = 2) -> np.ndarray:
    """A bright square sliding right by `step` pixels per frame, (F, H, W, 3) uint8."""
    video = np.zeros((frames, size, size, 3), dtype=np.uint8)
    for f in range(frames):
        x = 8 + f * step
        video[f, 24:40, x : x + 16] = 255
    return video


def test_frechet_distance_zero_for_identical_and_grows_with_shift():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(64, 8))

    assert evaluate.frechet_distance(feats, feats) == pytest.approx(0.0, abs=1e-6)
    # Shifting the mean by 3 in each of 8 dims adds 8 * 3^2 = 72.
    assert evaluate.frechet_distance(feats, feats + 3) == pytest.approx(72.0, rel=1e-3)


def test_frechet_distance_needs_two_samples():
    pytest.importorskip("scipy")
    assert np.isnan(evaluate.frechet_distance(np.ones((1, 4)), np.ones((3, 4))))


def test_flow_metrics_detect_same_and_reversed_motion():
    pytest.importorskip("cv2")
    video = _moving_square()

    same = evaluate.compute_flow_metrics(video, video)
    reversed_ = evaluate.compute_flow_metrics(video[::-1].copy(), video)
    frozen = evaluate.compute_flow_metrics(np.repeat(video[:1], len(video), axis=0), video)

    assert same["flow_cosine"] > 0.9
    assert reversed_["flow_cosine"] < -0.5
    assert frozen["frozen"] and not same["frozen"]


def test_decord_shim_reads_frames(tmp_path):
    imageio = pytest.importorskip("imageio")
    path = tmp_path / "clip.mp4"
    imageio.mimwrite(path, list(_moving_square()), fps=8)

    evaluate._install_decord_shim()
    import decord

    reader = decord.VideoReader(str(path), num_threads=1)
    assert len(reader) == 8
    assert reader.get_batch(range(3)).asnumpy().shape == (3, 64, 64, 3)

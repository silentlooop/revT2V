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


def _sliding_texture(frames: int = 8, size: int = 64, step: int = 1) -> np.ndarray:
    """A smooth random texture sliding horizontally by `step` px per frame
    (negative = left), (F, H, W, 3) uint8. Full-frame, so flows of opposite
    directions overlap pixel-for-pixel like method vs matched_target do."""
    import cv2

    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (size, size)).astype(np.uint8), (0, 0), 2)
    return np.stack([np.repeat(np.roll(texture, f * step, axis=1)[..., None], 3, axis=-1) for f in range(frames)])


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
    video = _sliding_texture(step=1)

    same = evaluate.compute_flow_metrics(video, video)
    reversed_ = evaluate.compute_flow_metrics(_sliding_texture(step=-1), video)

    assert set(same) == {"flow_cosine"}
    assert same["flow_cosine"] > 0.9
    assert reversed_["flow_cosine"] < -0.5


def test_decord_shim_reads_frames(tmp_path):
    imageio = pytest.importorskip("imageio")
    path = tmp_path / "clip.mp4"
    pytest.importorskip("cv2")
    imageio.mimwrite(path, list(_sliding_texture()), fps=8)

    evaluate._install_decord_shim()
    import decord

    reader = decord.VideoReader(str(path), num_threads=1)
    assert len(reader) == 8
    assert reader.get_batch(range(3)).asnumpy().shape == (3, 64, 64, 3)


def test_frechet_distance_finite_with_singular_covariances():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    # 3 samples in 16 dims: both covariances are rank-deficient.
    assert np.isfinite(evaluate.frechet_distance(rng.normal(size=(3, 16)), rng.normal(size=(3, 16))))


def test_compute_fvd_rejects_mismatched_sets():
    video = np.zeros((16, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="equal set sizes"):
        evaluate.compute_fvd([video, video], [video], detector=None)
    with pytest.raises(ValueError, match="resolution"):
        evaluate.compute_fvd([video], [video[:8]], detector=None)


def test_score_videos_dir_columns_and_reference_nan(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("diffusers")
    video = _sliding_texture().astype(np.float32) / 255
    evaluate.save_video_set(
        {"matched_target": {"p0_s0": video}, "teacher": {"p0_s0": video[::-1]}},
        {"p0_s0": "a texture"},
        tmp_path / "videos",
    )

    result = evaluate.score_videos_dir(tmp_path / "videos", metrics=[])

    assert all(tuple(row) == evaluate.PER_VIDEO_COLUMNS for row in result["rows"])
    reference = next(s for s in result["summary"] if s["method"] == "matched_target")
    assert np.isnan(reference["fvd"]) and np.isnan(reference["flow_cosine"])
    assert all(tuple(entry) == evaluate.SUMMARY_COLUMNS for entry in result["summary"])
